"""
Causal Multiscale Linear Dynamical System (CMSS) - Versión estable para auditorías.

Pipeline causal: x_t -> Phi_t (R^10) -> Online LDS (k=4) -> Mahalanobis -> PercentileCalibrator
"""

from collections import deque
import math
from typing import Optional
import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.detectors.seasonal import DailySeasonalRobustZ
from sixeyes.detectors.gram_tracker import GramSubspaceTracker
from sixeyes.detectors.ramp_gradient import CausalRampGradientTracker
from sixeyes.detectors.active_drift import ActiveCentroidDriftTracker
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator


class CausalFeatureExtractor:
    """Extractor causal de características multiescala Phi_t en R^10."""

    def __init__(
        self,
        seasonal_bucket_minutes: int = 15,
        seasonal_window: int = 20,
        gram_dim: int = 16,
        gram_k: int = 2,
        gram_stride: int = 1,
        ramp_gamma: float = 0.8,
        ramp_w_scale: int = 1000,
        active_w: int = 30,
        eps_base: float = 1e-4,
        ema_fast_alpha: float = 0.3,
        ema_slow_alpha: float = 0.02,
        median_window: int = 50,
        mad_window: int = 100,
    ) -> None:
        self.eps_base = eps_base
        self.ema_fast_alpha = ema_fast_alpha
        self.ema_slow_alpha = ema_slow_alpha
        self.median_window = median_window
        self.mad_window = mad_window

        self.seasonal = DailySeasonalRobustZ(
            bucket_size_minutes=seasonal_bucket_minutes,
            window_size=seasonal_window,
            min_observations=5,
            trend_window_size=0,
        )
        self.gram = GramSubspaceTracker(
            dimension=gram_dim, k_subspace=gram_k, stride=gram_stride,
            alpha=0.05, alpha_slow=0.01, update_interval=5, min_samples=32,
            beta=1.0, history_size=200,
        )
        self.ramp = CausalRampGradientTracker(gamma=ramp_gamma, w_scale=ramp_w_scale)
        self.active_drift = ActiveCentroidDriftTracker(w_active=active_w, eps_base=eps_base)

        self._median_buffer: deque[float] = deque(maxlen=median_window)
        self._mad_buffer: deque[float] = deque(maxlen=mad_window)
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._prev_x: Optional[float] = None

    def update(self, value: float, timestamp: str = "") -> np.ndarray:
        if math.isnan(value) or math.isinf(value):
            return np.zeros(10, dtype=np.float64)

        if not timestamp:
            timestamp = "2026-10-12T00:00:00"

        # 1. x_t crudo
        x_raw = value

        # 2. Derivada normalizada
        if self._prev_x is not None and len(self._mad_buffer) >= 10:
            dx = value - self._prev_x
            mad_vals = np.array(self._mad_buffer)
            med_mad = float(np.median(mad_vals))
            mad_est = float(np.median(np.abs(mad_vals - med_mad))) * 1.4826
            dx_norm = dx / max(mad_est, 1e-8)
        else:
            dx_norm = 0.0

        # 3. Desviación de tendencia
        if len(self._median_buffer) >= 5:
            dev_trend = value - float(np.median(self._median_buffer))
        else:
            dev_trend = 0.0

        # 4. Momentum dual
        if self._ema_fast is None:
            self._ema_fast = value
            self._ema_slow = value
        else:
            self._ema_fast = (1 - self.ema_fast_alpha) * self._ema_fast + self.ema_fast_alpha * value
            self._ema_slow = (1 - self.ema_slow_alpha) * self._ema_slow + self.ema_slow_alpha * value
        momentum = self._ema_fast - self._ema_slow

        # 5. Residuo estacional
        r_seasonal = self.seasonal.update(value, timestamp)

        # 6. Residuo ortogonal Canal A
        _, gram_ready = self.gram._embedder.update(value)
        if gram_ready and self.gram._U_k is not None:
            vec = self.gram._embedder.update(value)[0]
            if vec is not None:
                vc = vec - self.gram._mu
                coeff = vc @ self.gram._U_k
                proj = coeff @ self.gram._U_k.T
                r_A = float(np.linalg.norm(vc - proj))
            else:
                r_A = 0.0
        else:
            self.gram.update(value, timestamp)
            r_A = 0.0

        # 7. Colapso traza Canal B
        s_collapse = 0.0
        if self.gram._mean_trace > 1e-8 and self.gram._S is not None:
            current_trace = float(np.trace(self.gram._S))
            if current_trace >= 0:
                ratio = (self.gram._mean_trace + 1e-8) / (current_trace + 1e-8)
                s_collapse = max(0.0, float(math.log(ratio)))

        # 8. Persistencia rampa
        d_t = self.ramp.update(value, timestamp)

        # 9. Shift centroide activo
        delta_active = self.active_drift.update(value, timestamp)

        # 10. Indicador reposo
        is_rest = 1.0 if abs(value) <= self.eps_base else 0.0

        # Actualizar buffers
        self._median_buffer.append(value)
        self._mad_buffer.append(value)
        self._prev_x = value

        # Vector Phi_t
        phi = np.array([
            x_raw, dx_norm, dev_trend, momentum,
            r_seasonal, r_A, s_collapse, d_t, delta_active, is_rest
        ], dtype=np.float64)

        # Normalización robusta
        phi = np.clip(phi, -100.0, 100.0)
        phi = phi / (np.abs(phi) + 1.0) * 10.0

        return phi

    def reset(self) -> None:
        self.seasonal = DailySeasonalRobustZ()
        self.gram = GramSubspaceTracker()
        self.ramp = CausalRampGradientTracker()
        self.active_drift = ActiveCentroidDriftTracker()
        self._median_buffer.clear()
        self._mad_buffer.clear()
        self._ema_fast = None
        self._ema_slow = None
        self._prev_x = None


class OnlineLDS:
    """Filtro de Kalman estable con factor de olvido."""

    def __init__(
        self,
        latent_dim: int = 4,
        obs_dim: int = 10,
        forget_factor: float = 0.98,
        stability_eps: float = 1e-4,
    ) -> None:
        self.k = latent_dim
        self.D = obs_dim
        self.lambda_forget = forget_factor
        self.eps = stability_eps

        self.A = np.eye(self.k, dtype=np.float64) * 0.99
        self.C = np.random.randn(self.D, self.k).astype(np.float64) * 0.1
        self._C_learned = False

        self.Q = np.eye(self.k, dtype=np.float64) * 0.1
        self.R = np.eye(self.D, dtype=np.float64) * 5.0
        self.R[0, 0] = 50.0
        self.R[3, 3] = 50.0
        self.R[7, 7] = 30.0

        self.z_hat = np.zeros(self.k, dtype=np.float64)
        self.P = np.eye(self.k, dtype=np.float64) * 1.0

        self._step = 0
        self._warmup_steps = 50

    def update(self, phi: np.ndarray, timestamp: str = "") -> float:
        if phi.shape[0] != self.D:
            raise ValueError(f"Expected observation dim {self.D}, got {phi.shape[0]}")

        self._step += 1

        if not self._C_learned:
            if np.linalg.norm(phi) > 1e-6:
                self.C = np.outer(phi, np.ones(self.k)) * 0.01
                self._C_learned = True
            return 0.0

        # Predicción con factor de olvido
        z_pred = self.A @ self.z_hat
        P_pred = self.A @ self.P @ self.A.T + self.Q
        P_pred = P_pred / self.lambda_forget

        # Innovación
        y = phi - self.C @ z_pred
        S = self.C @ P_pred @ self.C.T + self.R

        S_reg = S + np.eye(self.D) * self.eps

        try:
            S_inv = np.linalg.inv(S_reg)
            mahal = float(y.T @ S_inv @ y)
        except np.linalg.LinAlgError:
            mahal = float(y.T @ y) / max(np.trace(S_reg), 1e-8)

        # Actualización Kalman
        if True:
            try:
                K = P_pred @ self.C.T @ S_inv
                self.z_hat = z_pred + K @ y
                I_KC = np.eye(self.k) - K @ self.C
                self.P = I_KC @ P_pred @ I_KC.T + K @ self.R @ K.T
            except np.linalg.LinAlgError:
                self.z_hat = z_pred
                self.P = P_pred

        # Clipping anti-overflow
        max_state = 1e6
        self.z_hat = np.clip(self.z_hat, -max_state, max_state)
        self.P = np.clip(self.P, -max_state, max_state)

        return mahal

    def reset(self) -> None:
        self.C = np.random.randn(self.D, self.k).astype(np.float64) * 0.1
        self._C_learned = False
        self.z_hat = np.zeros(self.k)
        self.P = np.eye(self.k) * 1.0
        self._step = 0


class CausalMultiscaleLDS(Detector):
    def __init__(
        self,
        latent_dim: int = 4,
        forget_factor: float = 0.98,
        stability_eps: float = 1e-4,
        calibrator_min_history: int = 100,
    ) -> None:
        self.extractor = CausalFeatureExtractor()
        self.lds = OnlineLDS(
            latent_dim=latent_dim,
            obs_dim=10,
            forget_factor=forget_factor,
            stability_eps=stability_eps,
        )
        self._mahal_history: list[float] = []
        self._min_history = calibrator_min_history

    def _percentile_score(self, mahal: float, step: int) -> float:
        if mahal <= 0.0:
            return 0.0
        if step <= 30:
            import bisect
            bisect.insort(self._mahal_history, mahal)
            return 0.0
        if len(self._mahal_history) < self._min_history:
            import bisect
            bisect.insort(self._mahal_history, mahal)
            return 0.0
        import bisect
        menores = bisect.bisect_left(self._mahal_history, mahal)
        iguales = bisect.bisect_right(self._mahal_history, mahal) - menores
        n = len(self._mahal_history)
        score = (menores + 0.5 * iguales + 0.5) / (n + 1)
        bisect.insort(self._mahal_history, mahal)
        return max(0.0, min(1.0, float(score)))

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        if not timestamp:
            timestamp = "2026-10-12T00:00:00"
        phi = self.extractor.update(value, timestamp)
        mahal_score = self.lds.update(phi, timestamp)
        score = self._percentile_score(mahal_score, self.lds._step)
        return score

    def reset(self) -> None:
        self.extractor.reset()
        self.lds.reset()
        self._mahal_history.clear()
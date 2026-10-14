"""
Causal Multiscale Linear Dynamical System (CMSS) — versión de producción.

Pipeline causal estricto O(1) por paso:
    x_t -> Phi_t (R^12) -> OnlineUKF/OnlineEKF (k=4) -> NLL -> PercentileCalibrator -> [0,1]

Componentes:
- CausalFeatureExtractor: 12 dimensiones causales normalizadas por MAD global rodante.
- OnlineEKF / OnlineUKF: filtro de Kalman (extendido / unscented) en streaming con
  regularización diagonal S_t + eps*I, factor de olvido y adaptación lenta de C (y A)
  con tasa decayente, congelada ante anomalías (gating por Mahalanobis).
- Un único PercentileCalibrator acoplado a la salida del filtro vía passthrough,
  preservando la monotonía de la cola derecha para el optimizador NAB.
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


# ---------------------------------------------------------------------------
# Utilidades numéricas
# ---------------------------------------------------------------------------

def _make_spd(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Proyecta M a simétrica positiva definida (clipping espectral)."""
    M = np.asarray(M, dtype=np.float64)
    M = (M + M.T) * 0.5
    M = np.clip(M, -1e8, 1e8)
    try:
        w, V = np.linalg.eigh(M)
    except np.linalg.LinAlgError:
        return np.eye(M.shape[0], dtype=np.float64) * max(eps, 1e-6)
    w = np.maximum(w, eps)
    return (V * w) @ V.T


def _stabilize_transition(A: np.ndarray, rho_max: float = 0.99) -> np.ndarray:
    """Proyecta A para que su radio espectral sea < rho_max."""
    try:
        w, V = np.linalg.eig(A)
        w = np.clip(w, -rho_max, rho_max)
        A_new = (V @ np.diag(w) @ np.linalg.inv(V)).real
        if not np.all(np.isfinite(A_new)):
            return A * 0.99
        return np.asarray(A_new, dtype=np.float64)
    except np.linalg.LinAlgError:
        return np.asarray(A, dtype=np.float64) * 0.99


def _safe_cond(S: np.ndarray) -> float:
    """Condition number con guardia ante fallo de SVD."""
    try:
        c = float(np.linalg.cond(S))
        if not math.isfinite(c):
            return float("inf")
        return c
    except np.linalg.LinAlgError:
        return float("inf")


class _ScorePassthrough(Detector):
    """Detector identidad: re-emite el score crudo para calibrarlo con PercentileCalibrator."""

    def update(self, value: float, timestamp: str = "") -> float:
        if value is None:
            return 0.0
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v


# ---------------------------------------------------------------------------
# Extractor causal de características (R^12)
# ---------------------------------------------------------------------------

class CausalFeatureExtractor:
    """Extractor causal de características multiescala Phi_t en R^12.

    Las 12 dimensiones (todas causales, O(1)):
      0: x_norm_global   = (x - mediana_global) / (MAD_global + eps)
      1: dx_norm         = (x - x_{t-1}) / (MAD_global + eps)
      2: dev_trend_50    = (x - mediana_50) / (MAD_50 + eps)
      3: momentum        = (EMA_rapida - EMA_lenta) / (MAD_global + eps)
      4: r_seasonal      (residuo DailySeasonalRobustZ)
      5: gram_score      (salida fusionada GramSubspaceTracker, una sola llamada)
      6: s_collapse      (log-ratio de traza del Gram tracker)
      7: D_t             (acumulador de rampa con fuga gamma=0.8)
      8: delta_active    (shift de centroide activo)
      9: is_rest         (1.0 si |x| <= eps_base, else 0.0)
     10: vol_ratio       = MAD_20 / (MAD_global + eps)
     11: d2_norm         = (x - 2*x_{t-1} + x_{t-2}) / (MAD_global + eps)
    """

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
        global_window: int = 1000,
        global_recompute_every: int = 20,
    ) -> None:
        self.eps_base = eps_base
        self.ema_fast_alpha = ema_fast_alpha
        self.ema_slow_alpha = ema_slow_alpha
        self.median_window = median_window
        self.global_window = global_window
        self.global_recompute_every = max(1, global_recompute_every)

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

        self._median_buf: deque[float] = deque(maxlen=median_window)
        self._global_buf: deque[float] = deque(maxlen=global_window)
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._prev: Optional[float] = None
        self._prev2: Optional[float] = None

        self._global_median: float = 0.0
        self._global_mad: float = 1.0
        self._global_counter: int = 0
        self._recompute_global()

    def _recompute_global(self) -> None:
        if len(self._global_buf) >= 20:
            arr = np.asarray(self._global_buf, dtype=np.float64)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) * 1.4826
            self._global_median = med
            self._global_mad = max(mad, 1e-6)
        elif len(self._global_buf) >= 5:
            arr = np.asarray(self._global_buf, dtype=np.float64)
            self._global_median = float(np.median(arr))
            self._global_mad = max(float(np.mean(np.abs(arr - self._global_median))), 1e-6)
        else:
            self._global_median = 0.0
            self._global_mad = 1.0

    @staticmethod
    def _mad_of(data) -> float:
        arr = np.asarray(list(data), dtype=np.float64)
        if arr.size < 3:
            return 1e-6
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) * 1.4826
        return max(mad, 1e-6)

    def update(self, value: float, timestamp: str = "") -> np.ndarray:
        if math.isnan(value) or math.isinf(value):
            return np.zeros(12, dtype=np.float64)
        if not timestamp:
            timestamp = "2026-10-12T00:00:00"

        gm = self._global_median
        gs = self._global_mad

        # 0. x normalizado global (historia pasada, sin fuga)
        x_norm = (value - gm) / (gs + 1e-8)

        # 1. primera diferencia normalizada
        if self._prev is not None:
            dx_norm = (value - self._prev) / (gs + 1e-8)
        else:
            dx_norm = 0.0

        # 2. desviación de tendencia inmediata (mediana 50 pasada)
        if len(self._median_buf) >= 5:
            med50 = float(np.median(self._median_buf))
            mad50 = self._mad_of(self._median_buf)
            dev_trend = (value - med50) / (mad50 + 1e-8)
        else:
            dev_trend = 0.0

        # 3. momentum EMA rápida - lenta (recursivo causal)
        if self._ema_fast is None:
            self._ema_fast = value
            self._ema_slow = value
        else:
            self._ema_fast = (1.0 - self.ema_fast_alpha) * self._ema_fast + self.ema_fast_alpha * value
            self._ema_slow = (1.0 - self.ema_slow_alpha) * self._ema_slow + self.ema_slow_alpha * value
        momentum = (self._ema_fast - self._ema_slow) / (gs + 1e-8)

        # 4. residuo estacional (componente existente, una sola llamada)
        r_seasonal = float(self.seasonal.update(value, timestamp))

        # 5-6. Gram: UNA sola llamada causal; score fusionado + colapso de traza
        gram_score = float(self.gram.update(value, timestamp))
        s_collapse = 0.0
        try:
            if self.gram._mean_trace > 1e-8 and self.gram._S is not None:
                current_trace = float(np.trace(self.gram._S))
                if current_trace >= 0.0:
                    ratio = (self.gram._mean_trace + 1e-8) / (current_trace + 1e-8)
                    s_collapse = max(0.0, float(math.log(ratio)))
        except Exception:
            s_collapse = 0.0
        r_A = gram_score

        # 7. persistencia de rampa
        d_t = float(self.ramp.update(value, timestamp))

        # 8. shift de centroide activo
        delta_active = float(self.active_drift.update(value, timestamp))

        # 9. indicador de reposo
        is_rest = 1.0 if abs(value) <= self.eps_base else 0.0

        # 10. volatilidad local relativa
        if len(self._median_buf) >= 10:
            tail = list(self._median_buf)[-20:]
            mad20 = self._mad_of(tail)
            vol_ratio = mad20 / (gs + 1e-8)
        else:
            vol_ratio = 0.0

        # 11. segunda diferencia normalizada
        if self._prev is not None and self._prev2 is not None:
            d2_norm = (value - 2.0 * self._prev + self._prev2) / (gs + 1e-8)
        else:
            d2_norm = 0.0

        phi = np.array([
            x_norm, dx_norm, dev_trend, momentum,
            r_seasonal, r_A, s_collapse, d_t,
            delta_active, is_rest, vol_ratio, d2_norm,
        ], dtype=np.float64)
        phi = np.clip(phi, -25.0, 25.0)

        # Ingesta causal POSTERIOR al cómputo (sin fuga del punto actual)
        self._median_buf.append(value)
        self._global_buf.append(value)
        self._prev2 = self._prev
        self._prev = value
        self._global_counter += 1
        if self._global_counter % self.global_recompute_every == 0:
            self._recompute_global()

        return phi

    def reset(self) -> None:
        self.seasonal = DailySeasonalRobustZ()
        self.gram = GramSubspaceTracker()
        self.ramp = CausalRampGradientTracker()
        self.active_drift = ActiveCentroidDriftTracker()
        self._median_buf.clear()
        self._global_buf.clear()
        self._ema_fast = None
        self._ema_slow = None
        self._prev = None
        self._prev2 = None
        self._global_median = 0.0
        self._global_mad = 1.0
        self._global_counter = 0


# ---------------------------------------------------------------------------
# Filtros de estado en línea (EKF / UKF)
# ---------------------------------------------------------------------------

class OnlineEKF:
    """Filtro de Kalman extendido en streaming con transición no lineal estable.

    f(z) = rho*z + (1-rho)*tanh(A z),  h(z) = C z.
    Score: log-verosimilitud negativa de la innovación.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        obs_dim: int = 12,
        rho: float = 0.98,
        forget_factor: float = 0.98,
        stability_eps: float = 1e-4,
        adapt_lr: float = 1e-5,
        adapt_decay: float = 0.999,
        min_lr: float = 1e-7,
        adapt_mahal_gate: float = 40.0,
        seed: int = 42,
    ) -> None:
        self.k = latent_dim
        self.D = obs_dim
        self.rho = rho
        self.lambda_forget = forget_factor
        self.eps = stability_eps
        self._adapt_lr0 = adapt_lr
        self._adapt_lr = adapt_lr
        self._adapt_decay = adapt_decay
        self._min_lr = min_lr
        self._gate = adapt_mahal_gate

        rng = np.random.RandomState(seed)
        self.A = np.eye(self.k, dtype=np.float64) * 0.9
        self.C = (rng.randn(self.D, self.k).astype(np.float64)) * 0.1
        self._C_learned = False

        self.Q = np.eye(self.k, dtype=np.float64) * 0.05
        self.R = np.eye(self.D, dtype=np.float64) * 3.0
        # Más ruido en dims de alta varianza / binarias
        if self.D > 0:
            self.R[0, 0] = 12.0
        if self.D > 3:
            self.R[3, 3] = 12.0
        if self.D > 7:
            self.R[7, 7] = 8.0
        if self.D > 9:
            self.R[9, 9] = 8.0

        self.z_hat = np.zeros(self.k, dtype=np.float64)
        self.P = np.eye(self.k, dtype=np.float64)

        self._step = 0
        self.last_cond: float = 1.0
        self.last_mahal: float = 0.0
        self.last_nll: float = 0.0

    def _f(self, z: np.ndarray) -> np.ndarray:
        return self.rho * z + (1.0 - self.rho) * np.tanh(self.A @ z)

    def _F_jac(self, z: np.ndarray) -> np.ndarray:
        u = self.A @ z
        sech2 = 1.0 - np.tanh(u) ** 2
        return self.rho * np.eye(self.k) + (1.0 - self.rho) * (sech2[:, None] * self.A)

    def update(self, phi: np.ndarray, timestamp: str = "") -> float:
        phi = np.asarray(phi, dtype=np.float64)
        if phi.shape[0] != self.D or not np.all(np.isfinite(phi)):
            return 0.0
        self._step += 1

        if not self._C_learned:
            if float(np.linalg.norm(phi)) > 1e-6:
                self.C = np.outer(phi, np.ones(self.k)) * 0.01
                self._C_learned = True
            return 0.0
        if self._step <= 10:
            # warmup: solo estabilizar estado
            z_pred = self._f(self.z_hat)
            F = self._F_jac(self.z_hat)
            P_pred = F @ self.P @ F.T + self.Q
            y = phi - self.C @ z_pred
            S = self.C @ P_pred @ self.C.T + self.R
            S_reg = S + np.eye(self.D) * self.eps
            try:
                v = np.linalg.solve(S_reg, y)
                K = np.linalg.solve(S_reg.T, (P_pred @ self.C.T).T).T
                self.z_hat = z_pred + K @ y
                I_KC = np.eye(self.k) - K @ self.C
                self.P = I_KC @ P_pred @ I_KC.T + K @ self.R @ K.T
            except np.linalg.LinAlgError:
                self.z_hat = z_pred
                self.P = P_pred
            self.P = np.clip(self.P, -1e6, 1e6)
            return 0.0

        z_pred = self._f(self.z_hat)
        F = self._F_jac(self.z_hat)
        P_pred = F @ self.P @ F.T + self.Q
        P_pred = P_pred / self.lambda_forget

        y = phi - self.C @ z_pred
        S = self.C @ P_pred @ self.C.T + self.R
        S_reg = S + np.eye(self.D) * self.eps
        cond = _safe_cond(S_reg)
        if not math.isfinite(cond) or cond > 500.0:
            boost = self.eps * max(1.0, (cond / 500.0) if math.isfinite(cond) else 100.0)
            S_reg = S + np.eye(self.D) * min(boost, 1.0)
            cond = _safe_cond(S_reg)
        self.last_cond = cond

        try:
            v = np.linalg.solve(S_reg, y)
            mahal = float(y @ v)
            sign, logdet = np.linalg.slogdet(S_reg)
            if sign <= 0 or not math.isfinite(logdet):
                logdet = float(self.D * math.log(max(float(np.mean(np.diag(S_reg))), self.eps)))
        except np.linalg.LinAlgError:
            mahal = float(y @ y) / max(float(np.trace(S_reg)), 1e-8)
            logdet = float(self.D * math.log(max(float(np.mean(np.diag(S_reg))), self.eps)))
        nll = 0.5 * (mahal + logdet + self.D * math.log(2.0 * math.pi))
        nll = float(np.clip(nll, 0.0, 1e6))
        self.last_mahal = mahal
        self.last_nll = nll

        try:
            K = np.linalg.solve(S_reg.T, (P_pred @ self.C.T).T).T
            self.z_hat = z_pred + K @ y
            I_KC = np.eye(self.k) - K @ self.C
            self.P = I_KC @ P_pred @ I_KC.T + K @ self.R @ K.T
        except np.linalg.LinAlgError:
            self.z_hat = z_pred
            self.P = P_pred
        self.z_hat = np.clip(self.z_hat, -1e6, 1e6)
        self.P = np.clip(self.P, -1e6, 1e6)

        # Adaptación lenta de C (y A) solo ante innovaciones normales
        if self._step > 200 and mahal < self._gate and self._adapt_lr > 0:
            lr = self._adapt_lr
            self.C = self.C + lr * np.outer(y, self.z_hat)
            self.C = np.clip(self.C, -3.0, 3.0)
            dA = lr * 0.1 * np.outer(self.z_hat - self.A @ self.z_hat, self.z_hat)
            self.A = _stabilize_transition(self.A + dA)
            self._adapt_lr = max(self._adapt_lr * self._adapt_decay, self._min_lr)

        return nll

    def reset(self) -> None:
        rng = np.random.RandomState(42)
        self.A = np.eye(self.k, dtype=np.float64) * 0.9
        self.C = rng.randn(self.D, self.k).astype(np.float64) * 0.1
        self._C_learned = False
        self.z_hat = np.zeros(self.k)
        self.P = np.eye(self.k)
        self._step = 0
        self._adapt_lr = self._adapt_lr0
        self.last_cond = 1.0
        self.last_mahal = 0.0
        self.last_nll = 0.0


class OnlineUKF:
    """Filtro de Kalman unscented en streaming con transición no lineal estable."""

    def __init__(
        self,
        latent_dim: int = 4,
        obs_dim: int = 12,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        rho: float = 0.98,
        forget_factor: float = 0.98,
        stability_eps: float = 1e-4,
        adapt_lr: float = 1e-5,
        adapt_decay: float = 0.999,
        min_lr: float = 1e-7,
        adapt_mahal_gate: float = 40.0,
        seed: int = 42,
    ) -> None:
        self.k = latent_dim
        self.D = obs_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.rho = rho
        self.lambda_forget = forget_factor
        self.eps = stability_eps
        self._adapt_lr0 = adapt_lr
        self._adapt_lr = adapt_lr
        self._adapt_decay = adapt_decay
        self._min_lr = min_lr
        self._gate = adapt_mahal_gate

        rng = np.random.RandomState(seed)
        self.A = np.eye(self.k, dtype=np.float64) * 0.9
        self.C = (rng.randn(self.D, self.k).astype(np.float64)) * 0.1
        self._C_learned = False

        self.Q = np.eye(self.k, dtype=np.float64) * 0.05
        self.R = np.eye(self.D, dtype=np.float64) * 3.0
        if self.D > 0:
            self.R[0, 0] = 12.0
        if self.D > 3:
            self.R[3, 3] = 12.0
        if self.D > 7:
            self.R[7, 7] = 8.0
        if self.D > 9:
            self.R[9, 9] = 8.0

        self.z_hat = np.zeros(self.k, dtype=np.float64)
        self.P = np.eye(self.k, dtype=np.float64)

        n = self.k
        lam = self.alpha ** 2 * (n + self.kappa) - n
        self._lam = lam
        self._Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        self._Wc = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        self._Wm[0] = lam / (n + lam)
        self._Wc[0] = lam / (n + lam) + (1.0 - self.alpha ** 2 + self.beta)

        self._step = 0
        self.last_cond: float = 1.0
        self.last_mahal: float = 0.0
        self.last_nll: float = 0.0

    def _f(self, z: np.ndarray) -> np.ndarray:
        return self.rho * z + (1.0 - self.rho) * np.tanh(self.A @ z)

    def _sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        n = self.k
        cov_r = _make_spd(cov + np.eye(n) * self.eps, self.eps)
        try:
            L = np.linalg.cholesky((n + self._lam) * cov_r)
        except np.linalg.LinAlgError:
            d = np.sqrt(np.maximum(np.diag(cov_r) * (n + self._lam), self.eps))
            L = np.diag(d)
        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = mean
        for i in range(n):
            sigmas[1 + i] = mean + L[:, i]
            sigmas[1 + n + i] = mean - L[:, i]
        return sigmas

    def update(self, phi: np.ndarray, timestamp: str = "") -> float:
        phi = np.asarray(phi, dtype=np.float64)
        if phi.shape[0] != self.D or not np.all(np.isfinite(phi)):
            return 0.0
        self._step += 1

        if not self._C_learned:
            if float(np.linalg.norm(phi)) > 1e-6:
                self.C = np.outer(phi, np.ones(self.k)) * 0.01
                self._C_learned = True
            return 0.0
        if self._step <= 10:
            # warmup corto sin score
            z_pred = self._f(self.z_hat)
            self.z_hat = z_pred
            return 0.0

        # --- Predicción unscented ---
        sigmas = self._sigma_points(self.z_hat, self.P)
        propagated = np.array([self._f(s) for s in sigmas])
        z_pred = self._Wm @ propagated
        dx = propagated - z_pred
        P_pred = (dx * self._Wc[:, None]).T @ dx + self.Q
        P_pred = _make_spd(P_pred / self.lambda_forget, self.eps)

        # --- Observación unscented (h lineal) ---
        Y = propagated @ self.C.T
        obs_pred = self._Wm @ Y
        dy = Y - obs_pred
        S = (dy * self._Wc[:, None]).T @ dy + self.R
        S_reg = S + np.eye(self.D) * self.eps
        cond = _safe_cond(S_reg)
        if not math.isfinite(cond) or cond > 500.0:
            boost = self.eps * max(1.0, (cond / 500.0) if math.isfinite(cond) else 100.0)
            S_reg = S + np.eye(self.D) * min(boost, 1.0)
            cond = _safe_cond(S_reg)
        self.last_cond = cond

        y = phi - obs_pred
        try:
            v = np.linalg.solve(S_reg, y)
            mahal = float(y @ v)
            sign, logdet = np.linalg.slogdet(S_reg)
            if sign <= 0 or not math.isfinite(logdet):
                logdet = float(self.D * math.log(max(float(np.mean(np.diag(S_reg))), self.eps)))
            Tc = (dx * self._Wc[:, None]).T @ dy
            K = np.linalg.solve(S_reg.T, Tc.T).T
            self.z_hat = z_pred + K @ y
            self.P = _make_spd(P_pred - K @ S_reg @ K.T, self.eps)
        except np.linalg.LinAlgError:
            mahal = float(y @ y) / max(float(np.trace(S_reg)), 1e-8)
            logdet = float(self.D * math.log(max(float(np.mean(np.diag(S_reg))), self.eps)))
            self.z_hat = z_pred
            self.P = P_pred

        nll = 0.5 * (mahal + logdet + self.D * math.log(2.0 * math.pi))
        nll = float(np.clip(nll, 0.0, 1e6))
        self.last_mahal = mahal
        self.last_nll = nll
        self.z_hat = np.clip(self.z_hat, -1e6, 1e6)
        self.P = np.clip(self.P, -1e6, 1e6)

        if self._step > 200 and mahal < self._gate and self._adapt_lr > 0:
            lr = self._adapt_lr
            self.C = np.clip(self.C + lr * np.outer(y, self.z_hat), -3.0, 3.0)
            self.A = _stabilize_transition(self.A + lr * 0.1 * np.outer(
                self.z_hat - self.A @ self.z_hat, self.z_hat))
            self._adapt_lr = max(self._adapt_lr * self._adapt_decay, self._min_lr)

        return nll

    def reset(self) -> None:
        rng = np.random.RandomState(42)
        self.A = np.eye(self.k, dtype=np.float64) * 0.9
        self.C = rng.randn(self.D, self.k).astype(np.float64) * 0.1
        self._C_learned = False
        self.z_hat = np.zeros(self.k)
        self.P = np.eye(self.k)
        self._step = 0
        self._adapt_lr = self._adapt_lr0
        self.last_cond = 1.0
        self.last_mahal = 0.0
        self.last_nll = 0.0


# Alias de compatibilidad
OnlineLDS = OnlineUKF


# ---------------------------------------------------------------------------
# Detector unificado con un único PercentileCalibrator
# ---------------------------------------------------------------------------

class CausalMultiscaleLDS(Detector):
    """Detector CMSS de producción: extractor R^12 + UKF/EKF + un calibrador.

    Pipeline causal estricto O(1):
        x_t -> Phi_t (R^12) -> filtro (k=4) -> NLL -> PercentileCalibrator -> [0,1]
    """

    OBS_DIM = 12

    def __init__(
        self,
        latent_dim: int = 4,
        filter_type: str = "ukf",
        forget_factor: float = 0.98,
        stability_eps: float = 1e-4,
        adapt_lr: float = 1e-5,
        calibrator_min_history: int = 288,
    ) -> None:
        self.latent_dim = latent_dim
        self.filter_type = filter_type
        self._min_history = calibrator_min_history

        self.extractor = CausalFeatureExtractor()
        if filter_type == "ekf":
            self.filter = OnlineEKF(
                latent_dim=latent_dim, obs_dim=self.OBS_DIM,
                forget_factor=forget_factor, stability_eps=stability_eps,
                adapt_lr=adapt_lr,
            )
        else:
            self.filter = OnlineUKF(
                latent_dim=latent_dim, obs_dim=self.OBS_DIM,
                forget_factor=forget_factor, stability_eps=stability_eps,
                adapt_lr=adapt_lr,
            )
        self._passthrough = _ScorePassthrough()
        self.calibrator = PercentileCalibrator(self._passthrough)
        self.calibrator.MIN_HISTORIAL = calibrator_min_history

    @property
    def last_cond(self) -> float:
        return float(getattr(self.filter, "last_cond", 1.0))

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        if not timestamp:
            timestamp = "2026-10-12T00:00:00"
        phi = self.extractor.update(value, timestamp)
        nll = float(self.filter.update(phi, timestamp))
        return float(self.calibrator.update(nll, timestamp))

    def reset(self) -> None:
        self.extractor.reset()
        self.filter.reset()
        self._passthrough = _ScorePassthrough()
        self.calibrator = PercentileCalibrator(self._passthrough)
        self.calibrator.MIN_HISTORIAL = self._min_history

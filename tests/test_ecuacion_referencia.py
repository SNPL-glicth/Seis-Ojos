"""Test de la implementación de referencia mínima en NumPy para la ecuación canónica de Six Eyes.

Verifica que la ecuación matemática implementada en NumPy puro (sin importar módulos de sixeyes)
reproduce bit a bit (tolerancia < 1e-9) la salida del pipeline real sobre 3 series canónicas de NAB.
"""

from collections import deque
import math
from pathlib import Path
import bisect
import numpy as np
import pandas as pd
import pytest

from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.detectors.refractory_suppressor import RefractorySuppressor


class PipelineEcuacionReferencia:
    """Implementación de referencia mínima en NumPy puro y biblioteca estándar.
    
    Reproduce las ecuaciones exactas del pipeline:
      z_t = (x_t - mu_{6400}) / sigma_{6400}
      v_t = [z_t, z_{t-1}, ..., z_{t-15}]^T
      h_t = tanh(W_in v_t + W_res h_{t-1})
      pred_t = w_{t-1}^T h_t
      e_t = (z_t - pred_t)^2
      p_t = (#{i in W: e_i < e_t} + 0.5 * #{e_i == e_t} + 0.5) / (|W| + 1)
      q_t = p_t si p_t > max(p_{t-24..t-1}); si no, p_t * exp(-(t - t*)/2.0)
      r_t = 1.0 si t >= 100 y x_t fuera de [m_t - 0.05*R_t, M_t + 0.05*R_t]; si no, q_t
      S_t = 0.0 durante 80 pasos tras cualquier r_t >= 0.990; si no, r_t
    """

    def __init__(self, seed: int = 42) -> None:
        self.d = 16
        self.Nres = 64
        self.rho = 0.9
        self.lam = 0.999
        self.delta = 1.0
        self.W_norm = 6400

        rng = np.random.RandomState(seed)
        self.W_in = rng.uniform(-1.0, 1.0, size=(self.Nres, self.d)) * 1.0
        w = rng.uniform(-1.0, 1.0, size=(self.Nres, self.Nres))
        rad = max(abs(np.linalg.eigvals(w)).max(), 1e-9)
        self.W_res = w * (self.rho / rad)

        self.h = np.zeros(self.Nres)
        self.wout = np.zeros(self.Nres)
        self.P = np.eye(self.Nres) * self.delta
        self.norm_deque: deque[float] = deque(maxlen=self.W_norm)
        self.nsum = 0.0
        self.nsumsq = 0.0
        self.delay_buf: deque[float] = deque(maxlen=self.d)

        self.W_cal = 16000
        self.min_cal = 288
        self.cal_hist: list[float] = []
        self.cal_cola: deque[float] = deque()

        self.K = 24
        self.tau = 2.0
        self.peak_hist: deque[float] = deque(maxlen=self.K)
        self.t = 0
        self.t_peak = 0

        self.tol = 0.05
        self.warmup = 100
        self.min_val: float | None = None
        self.max_val: float | None = None
        self.spatial_count = 0

        self.w_rest = 80
        self.trigger = 0.990
        self.refr_t = 0
        self.silence_until = -1

    def step(self, value: float) -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0

        # 1. Estandarización causal rodante
        n = len(self.norm_deque)
        if n < 2:
            z = 0.0
        else:
            mean = self.nsum / n
            var = max(self.nsumsq / n - mean * mean, 0.0)
            std = math.sqrt(var)
            z = 0.0 if std <= 1e-9 else (value - mean) / std

        # 2. Embedding de retardos
        self.delay_buf.append(z)
        if len(self.delay_buf) < self.d:
            vec = None
            ready = False
        else:
            buf_list = list(self.delay_buf)
            vec = np.array([buf_list[-(1 + i)] for i in range(self.d)], dtype=np.float64)
            ready = True

        if not ready or vec is None:
            if len(self.norm_deque) == self.W_norm:
                old = self.norm_deque[0]
                self.nsum -= old
                self.nsumsq -= old * old
            self.norm_deque.append(value)
            self.nsum += value
            self.nsumsq += value * value
            raw_err = 0.0
        else:
            h_new = np.tanh(self.W_in @ vec + self.W_res @ self.h)
            pred_z = float(self.wout @ h_new)
            if not math.isfinite(pred_z) or not bool(np.all(np.isfinite(self.P))):
                self.wout = np.zeros(self.Nres)
                self.P = np.eye(self.Nres) * self.delta
                pred_z = 0.0
            resid = z - pred_z
            if math.isnan(resid):
                err = 0.0
            else:
                err = min(resid * resid, 1e12) if math.isfinite(resid) else 1e12

            # Actualización RLS después de puntuar
            h_col = h_new.reshape(-1, 1)
            Ph = self.P @ h_col
            denom = self.lam + float((h_col.T @ Ph)[0, 0])
            k_gain = Ph / max(denom, 1e-12)
            self.wout = (self.wout.reshape(-1, 1) + k_gain * (z - pred_z)).ravel()
            self.P = (self.P - (k_gain @ h_col.T) @ self.P) / self.lam
            self.h = h_new

            if len(self.norm_deque) == self.W_norm:
                old = self.norm_deque[0]
                self.nsum -= old
                self.nsumsq -= old * old
            self.norm_deque.append(value)
            self.nsum += value
            self.nsumsq += value * value
            raw_err = max(float(err), 0.0)

        # 3. Calibrador ECDF
        if math.isnan(raw_err) or math.isinf(raw_err) or raw_err == 0.0:
            p_t = 0.0
        elif len(self.cal_hist) < self.min_cal:
            bisect.insort(self.cal_hist, raw_err)
            self.cal_cola.append(raw_err)
            p_t = 0.0
        else:
            menores = bisect.bisect_left(self.cal_hist, raw_err)
            iguales = bisect.bisect_right(self.cal_hist, raw_err) - menores
            n_hist = len(self.cal_hist)
            p_t = (menores + 0.5 * iguales + 0.5) / (n_hist + 1)
            if len(self.cal_hist) >= self.W_cal and self.cal_cola:
                oldest = self.cal_cola.popleft()
                idx = bisect.bisect_left(self.cal_hist, oldest)
                for j in range(idx, min(idx + 4, len(self.cal_hist))):
                    if self.cal_hist[j] == oldest:
                        self.cal_hist.pop(j)
                        break
                else:
                    try:
                        self.cal_hist.remove(oldest)
                    except ValueError:
                        pass
            bisect.insort(self.cal_hist, raw_err)
            self.cal_cola.append(raw_err)

        # 4. PeakDecay
        p_raw = max(0.0, min(1.0, float(p_t)))
        cur_t = self.t
        self.t += 1
        if not self.peak_hist or p_raw > max(self.peak_hist):
            self.t_peak = cur_t
            q_t = p_raw
        else:
            decay = math.exp(-(cur_t - self.t_peak) / self.tau)
            q_t = p_raw * decay
        self.peak_hist.append(p_raw)
        q_t = max(0.0, min(1.0, float(q_t)))

        # 5. SpatialSpikeBypass
        spatial_anomaly = False
        if self.spatial_count >= self.warmup and self.min_val is not None and self.max_val is not None:
            span = self.max_val - self.min_val
            if span > 1e-12:
                margin = span * self.tol
                if value > self.max_val + margin or value < self.min_val - margin:
                    spatial_anomaly = True
        if self.min_val is None or value < self.min_val:
            self.min_val = value
        if self.max_val is None or value > self.max_val:
            self.max_val = value
        self.spatial_count += 1

        r_t = 1.0 if spatial_anomaly else q_t

        # 6. RefractorySuppressor
        r_val = max(0.0, min(1.0, float(r_t)))
        rf_t = self.refr_t
        self.refr_t += 1
        if rf_t < self.silence_until:
            s_t = 0.0
        else:
            if r_val >= self.trigger and self.w_rest > 0:
                self.silence_until = rf_t + 1 + self.w_rest
            s_t = r_val

        return s_t


def _build_real_pipeline(seed: int = 42):
    return RefractorySuppressor(
        SpatialSpikeBypass(
            StreamingPeakDecay(
                PercentileCalibrator(EchoStatePredictor(seed=seed), max_history=16000),
                k_window=24,
                tau=2.0,
            ),
            tolerance=0.05,
            warmup=100,
        ),
        w_rest=80,
        trigger_threshold=0.990,
    )


@pytest.mark.parametrize(
    "rel_path",
    [
        "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
        "realAdExchange/exchange-2_cpc_results.csv",
        "realKnownCause/ambient_temperature_system_failure.csv",
    ],
)
def test_ecuacion_referencia_coincide_con_pipeline(rel_path: str):
    nab_dir = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "nab" / "data"
    csv_path = nab_dir / rel_path
    if not csv_path.exists():
        pytest.skip(f"Archivo de datos NAB no encontrado en {csv_path}")

    df = pd.read_csv(csv_path)
    values = df["value"].tolist()

    real_detector = _build_real_pipeline(seed=42)
    ref_detector = PipelineEcuacionReferencia(seed=42)

    scores_real = np.array([real_detector.update(v) for v in values], dtype=np.float64)
    scores_ref = np.array([ref_detector.step(v) for v in values], dtype=np.float64)

    diff = np.abs(scores_real - scores_ref)
    max_diff = float(np.max(diff))

    assert max_diff < 1e-9, f"Diferencia máxima {max_diff:.3e} excede tolerancia 1e-9 en {rel_path}"

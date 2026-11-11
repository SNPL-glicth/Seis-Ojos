"""Exportador de Pesos y Vectores Dorados de Seis Ojos (Captura Intrínseca).

Genera:
1. Pesos binarios (W_in, W_res, P_0) para las semillas 42, 7 y 99.
2. Trazas completas de 10.000 pasos capturadas dentro de la ejecución oficial
   de cada componente (sin discrepancias de fase ni adivinación de estados).
"""

import gzip
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
NAB_DATA_DIR = ROOT_DIR.parent / "benchmarks" / "nab" / "data"
OUTPUT_DIR_RUST = ROOT_DIR.parent / "RELIC" / "crates" / "relic_core" / "src" / "anomaly" / "data"
OUTPUT_DIR_SEIS = ROOT_DIR / "results" / "golden"

OUTPUT_DIR_RUST.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_SEIS.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.detectors.refractory_suppressor import RefractorySuppressor


def export_weights(seed: int, output_dir: Path):
    esn = EchoStatePredictor(seed=seed)
    win = esn._win.astype(np.float64)
    wres = esn._wres.astype(np.float64)
    p_init = esn._P.astype(np.float64)

    radius = max(abs(np.linalg.eigvals(wres)).max(), 1e-9)
    print(f"Semilla {seed}: Radio espectral de W_res = {radius:.6f}")

    bin_path = output_dir / f"weights_seed{seed}.bin"
    with open(bin_path, "wb") as f:
        f.write(win.tobytes())
        f.write(wres.tobytes())
        f.write(p_init.tobytes())

    return bin_path


class CaptureESN(EchoStatePredictor):
    def __init__(self, seed=42):
        super().__init__(seed=seed)
        self.last_z = 0.0
        self.last_ready = False
        self.last_vec = None
        self.last_pred_z = 0.0
        self.last_err = 0.0
        self.last_h_cand = np.zeros(self._n)

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            self.last_z = 0.0
            self.last_ready = False
            self.last_vec = None
            self.last_pred_z = 0.0
            self.last_err = 0.0
            return 0.0

        z = self._standardize(value)
        self.last_z = float(z)
        vec, ready = self._embed.update(z, timestamp)
        self.last_ready = bool(ready)
        self.last_vec = vec.copy() if (ready and vec is not None) else None

        if not ready or vec is None:
            self._ingest_norm(value)
            self.last_pred_z = 0.0
            self.last_err = 0.0
            self.last_h_cand = self._h.copy()
            return 0.0

        h_new = np.tanh(self._win @ vec + self._wres @ self._h)
        self.last_h_cand = h_new.copy()
        pred_z = float(self._wout @ h_new)
        self.last_pred_z = float(pred_z)

        if not math.isfinite(pred_z) or not bool(np.all(np.isfinite(self._P))):
            self._wout = np.zeros(self._n)
            self._P = np.eye(self._n) * 1.0
            pred_z = 0.0
            self.last_pred_z = 0.0

        resid = z - pred_z
        err = min(resid * resid, 1e12) if math.isfinite(resid) else 1e12
        self.last_err = float(err)

        # RLS update
        h = h_new.reshape(-1, 1)
        Ph = self._P @ h
        denom = self._lam + float((h.T @ Ph)[0, 0])
        K = Ph / max(denom, 1e-12)
        self._wout = (self._wout.reshape(-1, 1) + K * (z - pred_z)).ravel()
        self._P = (self._P - (K @ h.T) @ self._P) / self._lam
        self._h = h_new
        self._ingest_norm(value)
        return float(err)


class CaptureCalibrator(PercentileCalibrator):
    def __init__(self, detector):
        super().__init__(detector, max_history=16000)
        self.last_score = 0.0

    def update(self, value: float, timestamp: Any = None) -> float:
        raw = self._detector.update(value, timestamp if timestamp is not None else "")
        if math.isnan(raw) or math.isinf(raw) or raw == 0.0:
            self.last_score = 0.0
            return 0.0

        if len(self._historial) < self.MIN_HISTORIAL:
            import bisect
            bisect.insort(self._historial, raw)
            self._cola.append(raw)
            self.last_score = 0.0
            return 0.0

        import bisect
        menores = bisect.bisect_left(self._historial, raw)
        iguales = bisect.bisect_right(self._historial, raw) - menores
        n = len(self._historial)
        score = (menores + 0.5 * iguales + 0.5) / (n + 1)
        self.last_score = float(score)

        self._evict_if_full()
        bisect.insort(self._historial, raw)
        self._cola.append(raw)
        return score


class CapturePeakDecay(StreamingPeakDecay):
    def __init__(self, detector):
        super().__init__(detector, k_window=24, tau=2.0)
        self.last_is_peak = False
        self.last_nms_score = 0.0

    def update(self, value: float, timestamp: str = "") -> float:
        raw_score = self._detector.update(value, timestamp)
        if raw_score is None or math.isnan(raw_score) or math.isinf(raw_score):
            self.last_is_peak = False
            self.last_nms_score = 0.0
            return 0.0

        raw_score = max(0.0, min(1.0, float(raw_score)))
        t = self._t
        self._t += 1

        if not self._history or raw_score > max(self._history):
            self._t_peak = t
            score_out = raw_score
            self.last_is_peak = True
        else:
            decay = math.exp(-(t - self._t_peak) / self.tau)
            score_out = raw_score * decay
            self.last_is_peak = False

        self.last_nms_score = float(score_out)
        self._history.append(raw_score)
        return max(0.0, min(1.0, float(score_out)))


class CaptureSpatial(SpatialSpikeBypass):
    def __init__(self, detector):
        super().__init__(detector, tolerance=0.05, warmup=100)
        self.last_is_spatial = False

    def update(self, value: float, timestamp: Any = "") -> float:
        if math.isnan(value) or math.isinf(value):
            self.last_is_spatial = false
            return self._detector.update(value, timestamp)

        spatial_anomaly = False
        if self._count >= self.warmup and self._min_val is not None and self._max_val is not None:
            span = self._max_val - self._min_val
            if span > 1e-12:
                margin = span * self.tolerance
                if value > self._max_val + margin or value < self._min_val - margin:
                    spatial_anomaly = True

        self.last_is_spatial = spatial_anomaly

        if self._min_val is None or value < self._min_val:
            self._min_val = value
        if self._max_val is None or value > self._max_val:
            self._max_val = value

        self._count += 1
        base_score = self._detector.update(value, timestamp)

        if spatial_anomaly:
            return 1.0
        return base_score


class CaptureRefractory(RefractorySuppressor):
    def __init__(self, detector):
        super().__init__(detector, w_rest=80, trigger_threshold=0.990)
        self.last_score_nab = 0.0

    def update(self, value: float, timestamp: str = "") -> float:
        score = super().update(value, timestamp)
        self.last_score_nab = float(score)
        return float(score)


def main():
    print("=== EXPORTADOR DE PESOS Y VECTORES DORADOS DE SEIS OJOS (V3) ===")

    for seed in [42, 7, 99]:
        export_weights(seed, OUTPUT_DIR_SEIS)
        export_weights(seed, OUTPUT_DIR_RUST)

    meta = {
        "seeds": [42, 7, 99],
        "embed_dim": 16,
        "reservoir_size": 64,
        "spectral_radius": 0.9,
        "input_scale": 1.0,
        "rls_delta": 1.0,
        "rls_forget": 0.999,
        "norm_window": 6400,
        "ecdf_window": 16000,
        "min_historial": 288,
        "peak_window": 24,
        "tau": 2.0,
        "spatial_tolerance": 0.05,
        "spatial_warmup": 100,
        "refractory_rest": 80,
        "refractory_threshold": 0.990,
    }
    with open(OUTPUT_DIR_RUST / "config.json", "w") as f:
        json.dump(meta, f, indent=2)

    dataset_file = NAB_DATA_DIR / "realKnownCause" / "cpu_utilization_asg_misconfiguration.csv"
    if not dataset_file.is_file():
        print(f"ERROR: No se encontró el dataset en {dataset_file}")
        sys.exit(1)

    df = pd.read_csv(dataset_file)
    values = df["value"].values[:10000]
    timestamps = df["timestamp"].values[:10000]

    esn = CaptureESN(seed=42)
    cal = CaptureCalibrator(esn)
    decay = CapturePeakDecay(cal)
    spatial = CaptureSpatial(decay)
    refr = CaptureRefractory(spatial)

    steps = []

    print(f"Ejecutando pipeline intrínseco para {len(values)} observaciones...")
    for t, (val, ts) in enumerate(zip(values, timestamps)):
        val_f = float(val)
        ts_s = str(ts)
        
        score_nab = refr.update(val_f, ts_s)
        score_relic = 1.0 if spatial.last_is_spatial else decay.last_nms_score

        record = {
            "t": t,
            "raw_val": val_f,
            "timestamp": ts_s,
            "z": esn.last_z,
            "ready": esn.last_ready,
            "pred_z": esn.last_pred_z,
            "err": esn.last_err,
            "cal_score": cal.last_score,
            "is_peak": decay.last_is_peak,
            "nms_score": decay.last_nms_score,
            "is_spatial": spatial.last_is_spatial,
            "score_nab": score_nab,
            "score_relic": score_relic,
            "h_first4": [float(x) for x in esn._h[:4]],
            "wout_first4": [float(x) for x in esn._wout[:4]],
            "p_diag_first4": [float(esn._P[i, i]) for i in range(4)],
        }

        if t < 50 or t % 100 == 0:
            record["h_full"] = [float(x) for x in esn._h]
            record["wout_full"] = [float(x) for x in esn._wout]
            record["p_diag_full"] = [float(esn._P[i, i]) for i in range(64)]
            if esn.last_vec is not None:
                record["vec_takens"] = [float(x) for x in esn.last_vec]

        steps.append(record)

    out_gz_rust = OUTPUT_DIR_RUST / "golden_stream_10k.json.gz"
    out_gz_seis = OUTPUT_DIR_SEIS / "golden_stream_10k.json.gz"
    
    data_bytes = json.dumps(steps).encode("utf-8")
    with gzip.open(out_gz_rust, "wb") as f:
        f.write(data_bytes)
    with gzip.open(out_gz_seis, "wb") as f:
        f.write(data_bytes)

    print(f"Guardado golden_stream_10k.json.gz ({out_gz_rust.stat().st_size} bytes)")

    out_json_sample = OUTPUT_DIR_RUST / "golden_stream_sample500.json"
    with open(out_json_sample, "w") as f:
        json.dump(steps[:500], f, indent=2)
    print(f"Guardado golden_stream_sample500.json ({out_json_sample.stat().st_size} bytes)")

    out_json_1000 = OUTPUT_DIR_RUST / "golden_stream_1000.json"
    with open(out_json_1000, "w") as f:
        json.dump(steps[:1000], f, indent=2)
    print(f"Guardado golden_stream_1000.json ({out_json_1000.stat().st_size} bytes)")

    print("\n¡EXPORTACIÓN V3 COMPLETADA CON ÉXITO!")


if __name__ == "__main__":
    main()

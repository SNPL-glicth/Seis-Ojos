"""Diagnóstico micro-analítico de plasticidad RLS y memoria recurrente en PredictiveReadoutESN.

Condiciones experimentales (todas con warmup suficiente para ECDF > 288 pasos):
- Condición A: Meseta anómala sostenida (salto de nivel de 200 pasos y retorno a normalidad).
- Condición B: Deriva lineal lenta legítima (concept drift sin anomalías).
- Condición C: Régimen estacionario puro (control negativo sin anomalías ni deriva).

NO modifica el detector oficial ni el experimental de producción.
Instrumenta una variante diagnóstica local que registra paso a paso:
- Etiquetas reales
- Error de reconstrucción (e_rec^2) y de pronóstico (e_fc^2)
- Magnitud de actualización RLS: ||Delta w_rec||_2 y ||Delta w_fc||_2
- Score final calibrado
- Métricas forenses: absorción de la anomalía, falso pico de recuperación y seguimiento de deriva.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

# Resolver imports del proyecto
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.detectors.delay_embedding import DelayCoordinateEmbedding

# Constantes idénticas al candidato P05 congelado
EMBED_DIM = 16
RESERVOIR_SIZE = 64
SPECTRAL_RADIUS = 0.9
INPUT_SCALE = 1.0
RLS_DELTA = 1.0
RLS_FORGET = 0.999
NORM_WINDOW = 6400
SEED = 42
ERR_CAP = 1e12


class _InstrumentedRLSHead:
    """Cabeza RLS idéntica a producción pero instrumentada para medir la actualización."""
    __slots__ = ("w", "P", "n", "lam", "last_delta_norm")

    def __init__(self, n: int, lam: float = RLS_FORGET) -> None:
        self.n = n
        self.lam = lam
        self.last_delta_norm = 0.0
        self.reset()

    def predict(self, h: np.ndarray) -> float:
        return float(self.w @ h)

    def update(self, h: np.ndarray, target: float, pred: float, freeze: bool = False) -> float:
        if freeze:
            self.last_delta_norm = 0.0
            return 0.0
        if not math.isfinite(pred) or not bool(np.all(np.isfinite(self.P))):
            self.reset()
            self.last_delta_norm = 0.0
            return 0.0
        h_col = h.reshape(-1, 1)
        Ph = self.P @ h_col
        denom = self.lam + float((h_col.T @ Ph)[0, 0])
        K = Ph / max(denom, 1e-12)
        delta_w = (K * (target - pred)).ravel()
        self.w = self.w + delta_w
        self.P = (self.P - (K @ h_col.T) @ self.P) / self.lam
        self.last_delta_norm = float(np.linalg.norm(delta_w))
        return self.last_delta_norm

    def reset(self) -> None:
        self.w = np.zeros(self.n)
        self.P = np.eye(self.n) * RLS_DELTA
        self.last_delta_norm = 0.0


class InstrumentedPredictiveESN:
    """PredictiveReadoutESN instrumentado para trazabilidad interna punto a punto."""

    def __init__(self, seed: int = SEED, alpha_recon: float = 0.5) -> None:
        self._n = RESERVOIR_SIZE
        self._alpha = float(alpha_recon)
        self._embed = DelayCoordinateEmbedding(dimension=EMBED_DIM, stride=1)
        rng = np.random.RandomState(seed)
        self._win = rng.uniform(-1.0, 1.0, size=(RESERVOIR_SIZE, EMBED_DIM)) * INPUT_SCALE
        w = rng.uniform(-1.0, 1.0, size=(RESERVOIR_SIZE, RESERVOIR_SIZE))
        radius = max(abs(np.linalg.eigvals(w)).max(), 1e-9)
        self._wres = w * (SPECTRAL_RADIUS / radius)
        self._h = np.zeros(RESERVOIR_SIZE)
        self._rec = _InstrumentedRLSHead(RESERVOIR_SIZE)
        self._fc = _InstrumentedRLSHead(RESERVOIR_SIZE)
        self._norm: deque[float] = deque(maxlen=NORM_WINDOW)
        self._nsum = 0.0
        self._nsumsq = 0.0

        # Registro del último paso
        self.last_e_rec_sq = 0.0
        self.last_e_fc_sq = 0.0
        self.last_w_rec_delta = 0.0
        self.last_w_fc_delta = 0.0

    def _standardize(self, value: float) -> float:
        n = len(self._norm)
        if n < 2:
            return 0.0
        mean = self._nsum / n
        var = max(self._nsumsq / n - mean * mean, 0.0)
        std = math.sqrt(var)
        if std <= 1e-9:
            return 0.0
        return (value - mean) / std

    def _ingest_norm(self, value: float) -> None:
        if len(self._norm) == self._norm.maxlen:
            old = self._norm[0]
            self._nsum -= old
            self._nsumsq -= old * old
        self._norm.append(value)
        self._nsum += value
        self._nsumsq += value * value

    @staticmethod
    def _sq(resid: float) -> float:
        if math.isnan(resid):
            return 0.0
        if not math.isfinite(resid):
            return ERR_CAP
        return min(resid * resid, ERR_CAP)

    def update(self, value: float, timestamp: str = "", freeze: bool = False) -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        z = self._standardize(value)
        vec, ready = self._embed.update(z, timestamp)
        if not ready or vec is None:
            self._ingest_norm(value)
            self.last_e_rec_sq = 0.0
            self.last_e_fc_sq = 0.0
            self.last_w_rec_delta = 0.0
            self.last_w_fc_delta = 0.0
            return 0.0

        h_prev = self._h
        h_new = np.tanh(self._win @ vec + self._wres @ h_prev)

        pred_rec = self._rec.predict(h_new)
        if not math.isfinite(pred_rec) or not bool(np.all(np.isfinite(self._rec.P))):
            self._rec.reset()
            pred_rec = 0.0

        pred_fc = self._fc.predict(h_prev)
        if not math.isfinite(pred_fc) or not bool(np.all(np.isfinite(self._fc.P))):
            self._fc.reset()
            pred_fc = 0.0

        e_rec = z - pred_rec
        e_fc = z - pred_fc
        self.last_e_rec_sq = self._sq(e_rec)
        self.last_e_fc_sq = self._sq(e_fc)

        score = self._alpha * self.last_e_rec_sq + (1.0 - self._alpha) * self.last_e_fc_sq

        # Actualización RLS (con opción de congelar)
        self.last_w_rec_delta = self._rec.update(h_new, z, pred_rec, freeze=freeze)
        self.last_w_fc_delta = self._fc.update(h_prev, z, pred_fc, freeze=freeze)

        self._h = h_new
        self._ingest_norm(value)
        return max(float(score), 0.0)


def generate_benchmark_signals(n: int = 2000) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Genera las tres señales sintéticas con longitud suficiente para superar el warmup (288)."""
    t = np.arange(n)
    # Señal base sinusoidal (período 50 pasos) con ruido blanco ligero
    rng = np.random.RandomState(12345)
    noise = rng.normal(0, 0.05, size=n)
    base_signal = np.sin(2 * np.pi * t / 50.0) + noise

    # Condición A: Meseta anómala entre t=600 y t=800 (duración 200 pasos)
    sig_a = base_signal.copy()
    labels_a = np.zeros(n, dtype=int)
    sig_a[600:800] += 3.0  # Salto de nivel en meseta
    labels_a[600:800] = 1

    # Condición B: Deriva lineal lenta legítima (+0.001 por paso tras t=500, sin saltos)
    sig_b = base_signal.copy()
    labels_b = np.zeros(n, dtype=int)
    drift = np.zeros(n)
    drift[500:] = 0.0015 * (t[500:] - 500)
    sig_b += drift

    # Condición C: Régimen estacionario puro sin anomalías
    sig_c = base_signal.copy()
    labels_c = np.zeros(n, dtype=int)

    return {
        "A_meseta": (sig_a, labels_a),
        "B_deriva": (sig_b, labels_b),
        "C_estacionaria": (sig_c, labels_c),
    }


def run_condition_trace(signal: np.ndarray, labels: np.ndarray, name: str) -> List[Dict[str, Any]]:
    """Ejecuta el pipeline completo instrumentado y captura todas las variables paso a paso."""
    esn = InstrumentedPredictiveESN(seed=42, alpha_recon=0.5)
    cal = PercentileCalibrator(esn, max_history=16000)
    det = SpatialSpikeBypass(cal, tolerance=0.05, warmup=100)

    rows = []
    for i, (val, lbl) in enumerate(zip(signal, labels)):
        score = det.update(float(val), str(i))
        rows.append({
            "step": i,
            "value": float(val),
            "label": int(lbl),
            "score": float(score),
            "e_rec_sq": float(esn.last_e_rec_sq),
            "e_fc_sq": float(esn.last_e_fc_sq),
            "delta_w_rec": float(esn.last_w_rec_delta),
            "delta_w_fc": float(esn.last_w_fc_delta),
        })
    return rows


def analyze_meseta_diagnostics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcula métricas analíticas sobre la meseta anómala y el período de recuperación."""
    # Meseta: t in [600, 800)
    # Recuperación: t in [800, 950)
    scores = np.array([r["score"] for r in rows])
    steps = np.array([r["step"] for r in rows])

    meseta_scores = scores[600:800]
    recup_scores = scores[800:950]

    initial_anomaly_score = float(np.mean(meseta_scores[:10]))  # Primeros 10 pasos del salto
    final_anomaly_score = float(np.mean(meseta_scores[-10:]))   # Últimos 10 pasos de la meseta

    # Proporción de anomalía absorbida por el modelo:
    # 0 = score no cayó; 1.0 = score cayó a 0 (absorción completa)
    if initial_anomaly_score > 0:
        absorption_ratio = max(0.0, (initial_anomaly_score - final_anomaly_score) / initial_anomaly_score)
    else:
        absorption_ratio = 0.0

    max_recovery_score = float(np.max(recup_scores))
    mean_recovery_score = float(np.mean(recup_scores))

    # Duración de la falsa alarma de recuperación (pasos con score > 0.5 tras el retorno normal)
    fa_recovery_steps = int(np.sum(recup_scores > 0.5))

    return {
        "initial_anomaly_score": initial_anomaly_score,
        "final_anomaly_score": final_anomaly_score,
        "absorption_ratio": absorption_ratio,
        "max_recovery_score": max_recovery_score,
        "mean_recovery_score": mean_recovery_score,
        "fa_recovery_steps": fa_recovery_steps,
    }


def analyze_drift_diagnostics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcula métricas sobre el seguimiento de deriva y falsas alarmas."""
    scores = np.array([r["score"] for r in rows])
    # Región de deriva: t >= 600 (para dar warmup holgado)
    drift_scores = scores[600:]
    return {
        "mean_drift_score": float(np.mean(drift_scores)),
        "max_drift_score": float(np.max(drift_scores)),
        "false_alarm_rate_drift": float(np.mean(drift_scores > 0.5)),
    }


def analyze_stationary_diagnostics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcula métricas de control en régimen estacionario puro."""
    scores = np.array([r["score"] for r in rows])
    steady_scores = scores[300:]  # Post-warmup
    return {
        "mean_steady_score": float(np.mean(steady_scores)),
        "max_steady_score": float(np.max(steady_scores)),
        "false_alarm_rate_steady": float(np.mean(steady_scores > 0.5)),
    }


def main():
    print("=" * 70)
    print("EJECUTANDO DIAGNÓSTICO MICRO DE PLASTICIDAD RLS (FASE A)")
    print("=" * 70)

    signals = generate_benchmark_signals(n=2000)
    out_dir = REPO_ROOT / "results" / "diagnostics_plasticity"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for name, (sig, lbl) in signals.items():
        print(f"\n[Traza] Evaluando Condición: {name}...")
        trace = run_condition_trace(sig, lbl, name)
        csv_file = out_dir / f"trace_{name}.csv"

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            writer.writeheader()
            writer.writerows(trace)

        if name == "A_meseta":
            summary["meseta_diagnostics"] = analyze_meseta_diagnostics(trace)
        elif name == "B_deriva":
            summary["drift_diagnostics"] = analyze_drift_diagnostics(trace)
        elif name == "C_estacionaria":
            summary["stationary_diagnostics"] = analyze_stationary_diagnostics(trace)

    json_file = out_dir / "diagnostics_summary.json"
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    print("\n" + "=" * 70)
    print("RESULTADOS FORENSES DEL DIAGNÓSTICO (ESTADO ORIGINAL)")
    print("=" * 70)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

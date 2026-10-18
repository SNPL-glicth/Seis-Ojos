"""Fase B: Ablación Controlada de Plasticidad (RLS Continuo vs RLS Congelado vs Compuerta Prefrontal Causal).

Condiciones y variantes evaluadas bajo estricta causalidad y sin sesgo:
- Variante 1: RLS continuo (referencia original sin cambios, adapta siempre).
- Variante 2: RLS congelado (tras warmup de 300 pasos, w permanece fijo: control negativo de plasticidad cero).
- Variante 3: Compuerta prefrontal causal (bloquea actualización RLS si la amenaza causal local supera el umbral pre-fijado).

Aislamiento de mecanismos:
- Separa el error crudo del ESN (pre-ECDF), la inercia del estado recurrente h_t, y el score final percentil.
- Registra paso a paso:
    step, value, label, variant, condition, score, e_rec_sq, e_fc_sq, raw_score,
    gate_closed, delta_w_rec, delta_w_fc, cum_w_rec_norm, cum_w_fc_norm, h_norm
- Salida guardada en results/diagnostics_plasticity/phase_b/
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.detectors.delay_embedding import DelayCoordinateEmbedding

# Constantes del modelo
EMBED_DIM = 16
RESERVOIR_SIZE = 64
SPECTRAL_RADIUS = 0.9
INPUT_SCALE = 1.0
RLS_DELTA = 1.0
RLS_FORGET = 0.999
NORM_WINDOW = 6400
SEED = 42
ERR_CAP = 1e12

# Parámetros del experimento de ablación
WARMUP_STEPS = 300  # Cubre el MIN_HISTORIAL=288 de la ECDF
# Umbral de compuerta prefrontal pre-fijado sin mirar resultados:
# La amenaza se mide como el residuo crudo estandarizado frente a la varianza previa de residuos.
# Un z-score de residuo > 3.0 (equivalente a 3 sigmas de sorpresa) cierra la compuerta.
GATE_SURPRISE_SIGMAS = 3.0


class _AblationRLSHead:
    __slots__ = ("w", "P", "n", "lam", "cum_delta_norm")

    def __init__(self, n: int, lam: float = RLS_FORGET) -> None:
        self.n = n
        self.lam = lam
        self.cum_delta_norm = 0.0
        self.reset()

    def predict(self, h: np.ndarray) -> float:
        return float(self.w @ h)

    def update(self, h: np.ndarray, target: float, pred: float, freeze: bool = False) -> float:
        if freeze:
            return 0.0
        if not math.isfinite(pred) or not bool(np.all(np.isfinite(self.P))):
            self.reset()
            return 0.0
        h_col = h.reshape(-1, 1)
        Ph = self.P @ h_col
        denom = self.lam + float((h_col.T @ Ph)[0, 0])
        K = Ph / max(denom, 1e-12)
        delta_w = (K * (target - pred)).ravel()
        self.w = self.w + delta_w
        self.P = (self.P - (K @ h_col.T) @ self.P) / self.lam
        norm_val = float(np.linalg.norm(delta_w))
        self.cum_delta_norm += norm_val
        return norm_val

    def reset(self) -> None:
        self.w = np.zeros(self.n)
        self.P = np.eye(self.n) * RLS_DELTA
        self.cum_delta_norm = 0.0


class AblationPredictiveESN:
    """Implementa las 3 variantes con idéntica inicialización y causalidad."""

    def __init__(self, mode: str, seed: int = SEED, alpha_recon: float = 0.5) -> None:
        assert mode in ("continuous", "frozen", "gated"), f"Modo inválido: {mode}"
        self.mode = mode
        self._n = RESERVOIR_SIZE
        self._alpha = float(alpha_recon)
        self._embed = DelayCoordinateEmbedding(dimension=EMBED_DIM, stride=1)
        rng = np.random.RandomState(seed)
        self._win = rng.uniform(-1.0, 1.0, size=(RESERVOIR_SIZE, EMBED_DIM)) * INPUT_SCALE
        w = rng.uniform(-1.0, 1.0, size=(RESERVOIR_SIZE, RESERVOIR_SIZE))
        radius = max(abs(np.linalg.eigvals(w)).max(), 1e-9)
        self._wres = w * (SPECTRAL_RADIUS / radius)
        self._h = np.zeros(RESERVOIR_SIZE)
        self._rec = _AblationRLSHead(RESERVOIR_SIZE)
        self._fc = _AblationRLSHead(RESERVOIR_SIZE)
        self._norm: deque[float] = deque(maxlen=NORM_WINDOW)
        self._nsum = 0.0
        self._nsumsq = 0.0

        # Historial de residuos causales para la compuerta prefrontal (estadísticas antes del punto actual)
        self._step = 0
        self._res_sum = 0.0
        self._res_sq_sum = 0.0
        self._res_count = 0

        # Registros del paso
        self.last_e_rec_sq = 0.0
        self.last_e_fc_sq = 0.0
        self.last_raw_score = 0.0
        self.last_gate_closed = False
        self.last_delta_w_rec = 0.0
        self.last_delta_w_fc = 0.0
        self.last_h_norm = 0.0

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

    def update(self, value: float, timestamp: str = "") -> float:
        self._step += 1
        if math.isnan(value) or math.isinf(value):
            return 0.0
        z = self._standardize(value)
        vec, ready = self._embed.update(z, timestamp)
        if not ready or vec is None:
            self._ingest_norm(value)
            return 0.0

        h_prev = self._h
        h_new = np.tanh(self._win @ vec + self._wres @ h_prev)
        self.last_h_norm = float(np.linalg.norm(h_new))

        # 1. PREDICCIÓN CAUSAL ANTES DE CUALQUIER ACTUALIZACIÓN
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
        raw_score = self._alpha * self.last_e_rec_sq + (1.0 - self._alpha) * self.last_e_fc_sq
        self.last_raw_score = float(raw_score)

        # 2. DECISIÓN DE LA COMPUERTA PREFRONTAL (CAUSAL Y PREVIA A LA ACTUALIZACIÓN)
        freeze = False
        if self.mode == "continuous":
            freeze = False
        elif self.mode == "frozen":
            freeze = (self._step > WARMUP_STEPS)
        elif self.mode == "gated":
            if self._step <= WARMUP_STEPS:
                freeze = False
            else:
                # Criterio causal: sorpresa con respecto a la media/std histórica previa
                if self._res_count > 10:
                    r_mean = self._res_sum / self._res_count
                    r_var = max(self._res_sq_sum / self._res_count - r_mean * r_mean, 1e-6)
                    r_std = math.sqrt(r_var)
                    # Si el residuo actual supera la media histórica por más de K sigmas, amenaza detectada
                    is_threat = (raw_score - r_mean) > (GATE_SURPRISE_SIGMAS * r_std)
                    freeze = is_threat
                else:
                    freeze = False

        self.last_gate_closed = bool(freeze)

        # 3. ACTUALIZACIÓN RLS CONDICIONADA
        self.last_delta_w_rec = self._rec.update(h_new, z, pred_rec, freeze=freeze)
        self.last_delta_w_fc = self._fc.update(h_prev, z, pred_fc, freeze=freeze)

        # Actualizar acumuladores históricos de la compuerta solo con régimen normal (no amenazante)
        if not freeze:
            self._res_sum += raw_score
            self._res_sq_sum += raw_score * raw_score
            self._res_count += 1

        self._h = h_new
        self._ingest_norm(value)
        return max(float(raw_score), 0.0)


def generate_benchmark_signals(n: int = 2000) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Mismas señales sintéticas para garantizar paridad exacta de entrada."""
    t = np.arange(n)
    rng = np.random.RandomState(12345)
    noise = rng.normal(0, 0.05, size=n)
    base_signal = np.sin(2 * np.pi * t / 50.0) + noise

    # Condición A: Meseta anómala entre t=600 y t=800 (salto +3.0)
    sig_a = base_signal.copy()
    labels_a = np.zeros(n, dtype=int)
    sig_a[600:800] += 3.0
    labels_a[600:800] = 1

    # Condición B: Deriva lineal lenta legítima tras t=500 (+0.0015/paso)
    sig_b = base_signal.copy()
    labels_b = np.zeros(n, dtype=int)
    drift = np.zeros(n)
    drift[500:] = 0.0015 * (t[500:] - 500)
    sig_b += drift

    # Condición C: Régimen estacionario puro
    sig_c = base_signal.copy()
    labels_c = np.zeros(n, dtype=int)

    return {
        "A_meseta": (sig_a, labels_a),
        "B_deriva": (sig_b, labels_b),
        "C_estacionaria": (sig_c, labels_c),
    }


def run_pipeline(mode: str, signal: np.ndarray, labels: np.ndarray, cond_name: str) -> List[Dict[str, Any]]:
    """Ejecuta el pipeline completo ESN + Calibrador ECDF + SpatialBypass con trazabilidad."""
    esn = AblationPredictiveESN(mode=mode, seed=42, alpha_recon=0.5)
    cal = PercentileCalibrator(esn, max_history=16000)
    det = SpatialSpikeBypass(cal, tolerance=0.05, warmup=100)

    rows = []
    for i, (val, lbl) in enumerate(zip(signal, labels)):
        score = det.update(float(val), str(i))
        rows.append({
            "step": i,
            "value": float(val),
            "label": int(lbl),
            "variant": mode,
            "condition": cond_name,
            "score": float(score),
            "raw_score": float(esn.last_raw_score),
            "e_rec_sq": float(esn.last_e_rec_sq),
            "e_fc_sq": float(esn.last_e_fc_sq),
            "gate_closed": int(esn.last_gate_closed),
            "delta_w_rec": float(esn.last_delta_w_rec),
            "delta_w_fc": float(esn.last_delta_w_fc),
            "cum_w_rec_norm": float(esn._rec.cum_delta_norm),
            "cum_w_fc_norm": float(esn._fc.cum_delta_norm),
            "h_norm": float(esn.last_h_norm),
        })
    return rows


def compute_metrics(traces_by_variant: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
    """Calcula las métricas comparativas formales para cada variante en las 3 condiciones."""
    res = {}
    for var, conds in traces_by_variant.items():
        res[var] = {}

        # 1. Análisis Meseta
        rows_a = conds["A_meseta"]
        scores_a = np.array([r["score"] for r in rows_a])
        raw_a = np.array([r["raw_score"] for r in rows_a])
        meseta_scores = scores_a[600:800]
        meseta_raw = raw_a[600:800]
        recup_scores = scores_a[800:950]
        recup_raw = raw_a[800:950]

        # Absorción dentro de la meseta (error inicial vs final)
        init_raw = float(np.mean(meseta_raw[:10]))
        fin_raw = float(np.mean(meseta_raw[-10:]))
        raw_absorption = max(0.0, (init_raw - fin_raw) / max(init_raw, 1e-6))

        init_score = float(np.mean(meseta_scores[:10]))
        fin_score = float(np.mean(meseta_scores[-10:]))
        score_absorption = max(0.0, (init_score - fin_score) / max(init_score, 1e-6))

        res[var]["A_meseta"] = {
            "init_raw_score": init_raw,
            "fin_raw_score": fin_raw,
            "raw_absorption_ratio": raw_absorption,
            "init_calibrated_score": init_score,
            "fin_calibrated_score": fin_score,
            "score_absorption_ratio": score_absorption,
            "max_recovery_score": float(np.max(recup_scores)),
            "mean_recovery_score": float(np.mean(recup_scores)),
            "fa_recovery_steps_gt_05": int(np.sum(recup_scores > 0.5)),
            "cum_w_rec_total": float(rows_a[-1]["cum_w_rec_norm"]),
            "cum_w_fc_total": float(rows_a[-1]["cum_w_fc_norm"]),
        }

        # 2. Análisis Deriva
        rows_b = conds["B_deriva"]
        scores_b = np.array([r["score"] for r in rows_b])[600:]
        raw_b = np.array([r["raw_score"] for r in rows_b])[600:]
        res[var]["B_deriva"] = {
            "mean_drift_score": float(np.mean(scores_b)),
            "max_drift_score": float(np.max(scores_b)),
            "false_alarm_rate_gt_05": float(np.mean(scores_b > 0.5)),
            "mean_raw_drift_error": float(np.mean(raw_b)),
            "cum_w_rec_total": float(rows_b[-1]["cum_w_rec_norm"]),
            "cum_w_fc_total": float(rows_b[-1]["cum_w_fc_norm"]),
        }

        # 3. Análisis Estacionario
        rows_c = conds["C_estacionaria"]
        scores_c = np.array([r["score"] for r in rows_c])[300:]
        raw_c = np.array([r["raw_score"] for r in rows_c])[300:]
        res[var]["C_estacionaria"] = {
            "mean_steady_score": float(np.mean(scores_c)),
            "max_steady_score": float(np.max(scores_c)),
            "false_alarm_rate_gt_05": float(np.mean(scores_c > 0.5)),
            "mean_raw_steady_error": float(np.mean(raw_c)),
            "cum_w_rec_total": float(rows_c[-1]["cum_w_rec_norm"]),
            "cum_w_fc_total": float(rows_c[-1]["cum_w_fc_norm"]),
        }

    return res


def main():
    print("=" * 75)
    print("FASE B: ABLACIÓN CONTROLADA DE PLASTICIDAD RLS")
    print("=" * 75)

    signals = generate_benchmark_signals(n=2000)
    out_dir = REPO_ROOT / "results" / "diagnostics_plasticity" / "phase_b"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ("continuous", "frozen", "gated")
    traces_by_variant = {v: {} for v in variants}

    for var in variants:
        print(f"\n[Variante] Evaluando: {var.upper()}...")
        for cond_name, (sig, lbl) in signals.items():
            trace = run_pipeline(var, sig, lbl, cond_name)
            traces_by_variant[var][cond_name] = trace

            csv_file = out_dir / f"trace_{var}_{cond_name}.csv"
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
                writer.writeheader()
                writer.writerows(trace)

    summary = compute_metrics(traces_by_variant)
    summary_file = out_dir / "diagnostics_summary.json"
    with open(summary_file, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    print("\n" + "=" * 75)
    print("RESUMEN COMPARATIVO FORMAL (FASE B)")
    print("=" * 75)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

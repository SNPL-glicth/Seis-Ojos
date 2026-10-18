"""Fase C: Aislamiento Factorial de Mecanismos de Memoria (C.1 Plasticidad RLS y C.2 Inercia de Reservorio).

Intervenciones de laboratorio controladas (aisladas del código de producción):
- C.1 (Plasticidad Pura): Mantiene normalización fija durante la meseta (evitando oráculo en producción;
  aquí es intervención de laboratorio) para medir cuánto de la absorción y del error de recuperación
  corresponde 100% a la adaptación de los pesos w frente a w congelado.
- C.2 (Inercia de Reservorio): En t=800 (retorno a normalidad), compara:
    (a) Sin intervención en h_t (inercia natural).
    (b) Reset h_800 = 0 (control de choque).
    (c) Warm-state replay: h_800 restaurado desde un estado previo normal en la misma fase periódica (t=550).

Mide periodos completos (múltiplos de T=50) para evitar distorsiones de fase armónica.
Guarda artefactos en results/diagnostics_plasticity/phase_c/
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

from sixeyes.detectors.delay_embedding import DelayCoordinateEmbedding

# Constantes del modelo (paridad con P05)
EMBED_DIM = 16
RESERVOIR_SIZE = 64
SPECTRAL_RADIUS = 0.9
INPUT_SCALE = 1.0
RLS_DELTA = 1.0
RLS_FORGET = 0.999
NORM_WINDOW = 6400
SEED = 42
ERR_CAP = 1e12
PERIOD = 50  # Período de la onda sintética


class _Head:
    __slots__ = ("w", "P", "n", "lam")

    def __init__(self, n: int, lam: float = RLS_FORGET) -> None:
        self.n = n
        self.lam = lam
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
        return float(np.linalg.norm(delta_w))

    def reset(self) -> None:
        self.w = np.zeros(self.n)
        self.P = np.eye(self.n) * RLS_DELTA


class FactorialESN:
    """Implementa intervenciones de laboratorio C.1 y C.2."""

    def __init__(
        self,
        freeze_w: bool = False,
        freeze_norm_in_anomaly: bool = False,
        h_intervention_at_800: str = "none",  # "none", "zero", "phase_ref"
        seed: int = SEED,
        alpha_recon: float = 0.5,
    ) -> None:
        self.freeze_w = freeze_w
        self.freeze_norm_in_anomaly = freeze_norm_in_anomaly
        self.h_intervention = h_intervention_at_800
        self._n = RESERVOIR_SIZE
        self._alpha = float(alpha_recon)
        self._embed = DelayCoordinateEmbedding(dimension=EMBED_DIM, stride=1)
        rng = np.random.RandomState(seed)
        self._win = rng.uniform(-1.0, 1.0, size=(RESERVOIR_SIZE, EMBED_DIM)) * INPUT_SCALE
        w = rng.uniform(-1.0, 1.0, size=(RESERVOIR_SIZE, RESERVOIR_SIZE))
        radius = max(abs(np.linalg.eigvals(w)).max(), 1e-9)
        self._wres = w * (SPECTRAL_RADIUS / radius)
        self._h = np.zeros(RESERVOIR_SIZE)
        self._rec = _Head(RESERVOIR_SIZE)
        self._fc = _Head(RESERVOIR_SIZE)
        self._norm: deque[float] = deque(maxlen=NORM_WINDOW)
        self._nsum = 0.0
        self._nsumsq = 0.0

        # Snapshot de h para intervención de fase (t=550 está exactamente a 5 períodos antes de t=800)
        self._h_snapshot_t550 = np.zeros(RESERVOIR_SIZE)

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

    def _ingest_norm(self, value: float, freeze_norm: bool) -> None:
        if freeze_norm:
            return
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

    def step(self, t: int, value: float, is_anomaly: bool) -> Dict[str, float]:
        if math.isnan(value) or math.isinf(value):
            return {"raw_score": 0.0, "e_rec_sq": 0.0, "e_fc_sq": 0.0, "d_w": 0.0, "h_norm": 0.0}

        # Intervención C.2 en t=800 exactamente al inicio del paso
        if t == 800:
            if self.h_intervention == "zero":
                self._h = np.zeros(self._n)
            elif self.h_intervention == "phase_ref":
                self._h = self._h_snapshot_t550.copy()

        # Guardar snapshot en t=550 (misma fase periódica que t=800: (800 - 550) = 250 = 5 * 50)
        if t == 550:
            self._h_snapshot_t550 = self._h.copy()

        # Intervención C.1: congelar ingesta de normalización durante anomalía
        freeze_norm = bool(self.freeze_norm_in_anomaly and is_anomaly)
        z = self._standardize(value)
        vec, ready = self._embed.update(z, str(t))
        if not ready or vec is None:
            self._ingest_norm(value, freeze_norm)
            return {"raw_score": 0.0, "e_rec_sq": 0.0, "e_fc_sq": 0.0, "d_w": 0.0, "h_norm": 0.0}

        h_prev = self._h
        h_new = np.tanh(self._win @ vec + self._wres @ h_prev)
        h_norm = float(np.linalg.norm(h_new))

        # Predicción antes de actualizar
        pred_rec = self._rec.predict(h_new)
        if not math.isfinite(pred_rec):
            self._rec.reset()
            pred_rec = 0.0
        pred_fc = self._fc.predict(h_prev)
        if not math.isfinite(pred_fc):
            self._fc.reset()
            pred_fc = 0.0

        e_rec_sq = self._sq(z - pred_rec)
        e_fc_sq = self._sq(z - pred_fc)
        raw_score = float(self._alpha * e_rec_sq + (1.0 - self._alpha) * e_fc_sq)

        # Actualización de pesos (intervención C.1: congelar w tras warmup o durante meseta)
        freeze_w = bool(self.freeze_w and t > 300)
        d_rec = self._rec.update(h_new, z, pred_rec, freeze=freeze_w)
        d_fc = self._fc.update(h_prev, z, pred_fc, freeze=freeze_w)

        self._h = h_new
        self._ingest_norm(value, freeze_norm)
        return {
            "raw_score": raw_score,
            "e_rec_sq": float(e_rec_sq),
            "e_fc_sq": float(e_fc_sq),
            "d_w": float(d_rec + d_fc),
            "h_norm": h_norm,
        }


def generate_signal(n: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(n)
    rng = np.random.RandomState(12345)
    noise = rng.normal(0, 0.05, size=n)
    signal = np.sin(2 * np.pi * t / float(PERIOD)) + noise
    labels = np.zeros(n, dtype=int)
    # Meseta anómala: 4 períodos completos exactamente (t in [600, 800), 200 pasos)
    signal[600:800] += 3.0
    labels[600:800] = 1
    return signal, labels


def run_experiment_trace(
    signal: np.ndarray,
    labels: np.ndarray,
    freeze_w: bool,
    freeze_norm: bool,
    h_intervention: str,
) -> List[Dict[str, Any]]:
    model = FactorialESN(
        freeze_w=freeze_w,
        freeze_norm_in_anomaly=freeze_norm,
        h_intervention_at_800=h_intervention,
        seed=SEED,
        alpha_recon=0.5,
    )
    rows = []
    for t, (val, lbl) in enumerate(zip(signal, labels)):
        m = model.step(t, float(val), bool(lbl == 1))
        rows.append({
            "step": t,
            "value": float(val),
            "label": int(lbl),
            "raw_score": m["raw_score"],
            "e_rec_sq": m["e_rec_sq"],
            "e_fc_sq": m["e_fc_sq"],
            "d_w": m["d_w"],
            "h_norm": m["h_norm"],
        })
    return rows


def summarize_cycle_metrics(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw = np.array([r["raw_score"] for r in trace])
    # Períodos exactos (T=50):
    # Base normal antes de anomalía: t in [500, 600) (2 ciclos)
    # Meseta inicial (primer ciclo anómalo): t in [600, 650) (1 ciclo)
    # Meseta final (último ciclo anómalo): t in [750, 800) (1 ciclo)
    # Meseta completa: t in [600, 800) (4 ciclos)
    # Retorno inmediato (primer ciclo normal post-anomalía): t in [800, 850) (1 ciclo)
    # Retorno extendido: t in [800, 950) (3 ciclos)
    pre_normal = float(np.mean(raw[500:600]))
    init_anomaly_cycle = float(np.mean(raw[600:650]))
    fin_anomaly_cycle = float(np.mean(raw[750:800]))
    meseta_all = float(np.mean(raw[600:800]))

    # Absorción en ciclos completos (inmune a fase periódica)
    absorption_ratio = max(0.0, (init_anomaly_cycle - fin_anomaly_cycle) / max(init_anomaly_cycle, 1e-6))

    recup_immediate_cycle = float(np.mean(raw[800:850]))
    recup_extended_3cycles = float(np.mean(raw[800:950]))
    max_recup_peak = float(np.max(raw[800:950]))

    # Ratio de error de recuperación frente a la línea base normal previa:
    # 1.0 = retornó idéntico a normal; >1.0 = falsa alarma de recuperación activa
    recup_elevation_ratio = recup_immediate_cycle / max(pre_normal, 1e-6)

    # Duración de la cola: pasos en [800, 1000) con error > 3 * pre_normal
    tail_threshold = 3.0 * pre_normal
    tail_steps = int(np.sum(raw[800:1000] > tail_threshold))

    return {
        "pre_normal_baseline_raw": pre_normal,
        "init_anomaly_cycle_raw": init_anomaly_cycle,
        "fin_anomaly_cycle_raw": fin_anomaly_cycle,
        "meseta_all_mean_raw": meseta_all,
        "true_absorption_ratio": absorption_ratio,
        "recup_immediate_cycle_raw": recup_immediate_cycle,
        "recup_extended_3cycles_raw": recup_extended_3cycles,
        "max_recup_peak_raw": max_recup_peak,
        "recup_elevation_ratio": recup_elevation_ratio,
        "tail_steps_gt_3x_normal": tail_steps,
    }


def main():
    print("=" * 75)
    print("FASE C: AISLAMIENTO FACTORIAL (C.1 PLASTICIDAD RLS + C.2 INERCIA RESERVORIO)")
    print("=" * 75)

    signal, labels = generate_signal(n=2000)
    out_dir = REPO_ROOT / "results" / "diagnostics_plasticity" / "phase_c"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = {
        # --- C.1: Aislamiento de Plasticidad RLS (Normalización fija en anomalía para ambas) ---
        "c1_baseline_continuous_pure": {
            "freeze_w": False, "freeze_norm": True, "h_intervention": "none",
            "desc": "C.1 Control: RLS continuo con normalización limpia (plasticidad activa sin sesgo norm)"
        },
        "c1_frozen_w_pure": {
            "freeze_w": True, "freeze_norm": True, "h_intervention": "none",
            "desc": "C.1 Aislamiento RLS: Pesos congelados con normalización limpia (plasticidad pura = 0)"
        },
        # --- C.2: Aislamiento de Inercia de Reservorio en t=800 (Sobre base RLS continuo limpio) ---
        "c2_natural_h_inertia": {
            "freeze_w": False, "freeze_norm": True, "h_intervention": "none",
            "desc": "C.2 Control: Inercia natural de h_t al regresar"
        },
        "c2_h_reset_zero": {
            "freeze_w": False, "freeze_norm": True, "h_intervention": "zero",
            "desc": "C.2 Intervención: Reset de choque h_800 = 0"
        },
        "c2_h_phase_reference_restore": {
            "freeze_w": False, "freeze_norm": True, "h_intervention": "phase_ref",
            "desc": "C.2 Intervención: Restauración de h_800 desde referencia normal en misma fase periódica"
        },
    }

    results = {}
    for exp_id, cfg in experiments.items():
        print(f"\n[Ejecutando] {exp_id} -> {cfg['desc']}")
        trace = run_experiment_trace(
            signal, labels,
            freeze_w=cfg["freeze_w"],
            freeze_norm=cfg["freeze_norm"],
            h_intervention=cfg["h_intervention"],
        )

        csv_file = out_dir / f"{exp_id}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            writer.writeheader()
            writer.writerows(trace)

        summary = summarize_cycle_metrics(trace)
        summary["config"] = cfg
        results[exp_id] = summary

    json_file = out_dir / "phase_c_summary.json"
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2)

    print("\n" + "=" * 75)
    print("RESUMEN FORMAL ENTREGABLES FASE C (C.1 y C.2)")
    print("=" * 75)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

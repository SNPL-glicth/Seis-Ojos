"""Fase IV — Diagnóstico Temporal Fino de la Trayectoria de Plasticidad.

Objetivo:
  Determinar empíricamente con resolución de paso a paso (t in [450, 800]):
  1. ¿En qué paso exacto t* la ganancia eta_t comienza a reactivarse durante la meseta [500, 700)?
  2. ¿Cómo evoluciona la norma de los pesos ||w_t|| y la matriz P_t frente a la línea base?
  3. ¿Cuánto de la deformación final de w proviene de la reactivación por suelo adaptativo q_t?
  4. Perfilado de latencia: comparar el coste del MAD contemporáneo vs estimadores de escala streaming.

Guarda artefactos en:
  - results/diagnostics_plasticity/phase_iv_temporal_trajectory.csv
  - results/diagnostics_plasticity/phase_iv_temporal_summary.json
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sixeyes.experimental.predictive_readout_esn import PredictiveReadoutESN
from sixeyes.experimental.continuous_plasticity_controller import ContinuousPlasticityController

OUT_DIR = REPO_ROOT / "results" / "diagnostics_plasticity"
PERIOD = 50
SEED = 42


def run_temporal_diagnosis(n: int = 1200):
    t = np.arange(n)
    rng = np.random.RandomState(SEED)
    noise = rng.normal(0, 0.05, size=n)
    signal = np.sin(2 * np.pi * t / float(PERIOD)) + noise
    labels = np.zeros(n, dtype=int)
    # Meseta anómala prolongada de 200 pasos en t in [500, 700)
    signal[500:700] += 3.0
    labels[500:700] = 1

    # 1. Ejecutar línea base de referencia
    base = PredictiveReadoutESN(seed=SEED)
    base_scores = []
    base_w_rec_norms = []
    base_w_fc_norms = []
    for val in signal:
        s = base.update(float(val))
        base_scores.append(s)
        base_w_rec_norms.append(float(np.linalg.norm(base._rec.w)))
        base_w_fc_norms.append(float(np.linalg.norm(base._fc.w)))

    # 2. Ejecutar controlador continuo con traza activa
    ctrl = ContinuousPlasticityController(
        enable_controller=True, window=50, shock_center=6.0, shock_slope=1.5,
        record_trace=True, seed=SEED
    )
    ctrl_scores = []
    ctrl_w_rec_norms = []
    ctrl_w_fc_norms = []
    for val in signal:
        s = ctrl.update(float(val))
        ctrl_scores.append(s)
        ctrl_w_rec_norms.append(float(np.linalg.norm(ctrl._rec.w)))
        ctrl_w_fc_norms.append(float(np.linalg.norm(ctrl._fc.w)))

    tr = ctrl.get_trace()

    # Consolidar filas detalladas paso a paso
    rows = []
    for step in range(n):
        rows.append({
            "step": step,
            "signal": float(signal[step]),
            "label": int(labels[step]),
            "base_score": float(base_scores[step]),
            "base_w_fc_norm": float(base_w_fc_norms[step]),
            "ctrl_score": float(ctrl_scores[step]),
            "ctrl_w_fc_norm": float(ctrl_w_fc_norms[step]),
            "shock_v": float(tr["shock"][step]),
            "persistence_q": float(tr["persistence"][step]),
            "shock_memory_g": float(tr["g"][step]),
            "gain_eta": float(tr["eta"][step]),
        })

    # Analizar ventana crítica de la meseta [500, 700) y retorno [700, 850)
    eta_meseta = tr["eta"][500:700]
    q_meseta = tr["persistence"][500:700]
    g_meseta = tr["g"][500:700]

    # Paso donde eta vuelve a superar 0.2, 0.5 y 0.8 durante la meseta
    t_react_20 = None
    t_react_50 = None
    t_react_80 = None
    for idx, eta_val in enumerate(eta_meseta):
        abs_step = 500 + idx
        if t_react_20 is None and eta_val >= 0.2:
            t_react_20 = abs_step
        if t_react_50 is None and eta_val >= 0.5:
            t_react_50 = abs_step
        if t_react_80 is None and eta_val >= 0.8:
            t_react_80 = abs_step

    # Deformación de pesos en el retorno (t=700)
    w_fc_base_t700 = base_w_fc_norms[700]
    w_fc_ctrl_t700 = ctrl_w_fc_norms[700]
    w_fc_init_t500 = ctrl_w_fc_norms[500]

    # Duración de la cola residual de retorno post-700
    pre_normal_ctrl = float(np.mean(ctrl_scores[350:500]))
    tail_steps_ctrl = int(np.sum(np.array(ctrl_scores[700:850]) > 3.0 * pre_normal_ctrl))
    pre_normal_base = float(np.mean(base_scores[350:500]))
    tail_steps_base = int(np.sum(np.array(base_scores[700:850]) > 3.0 * pre_normal_base))

    summary = {
        "anomaly_window": [500, 700],
        "step_shock_onset_t500": {
            "shock_v": float(tr["shock"][500]),
            "initial_gain_eta": float(tr["eta"][500]),
            "initial_persistence_q": float(tr["persistence"][500]),
        },
        "plasticity_reactivation_steps": {
            "step_eta_gt_0p2": t_react_20,
            "steps_protected_under_0p2": (t_react_20 - 500) if t_react_20 else 200,
            "step_eta_gt_0p5": t_react_50,
            "steps_protected_under_0p5": (t_react_50 - 500) if t_react_50 else 200,
            "step_eta_gt_0p8": t_react_80,
            "steps_protected_under_0p8": (t_react_80 - 500) if t_react_80 else 200,
        },
        "weight_norms_fc": {
            "norm_at_onset_t500": w_fc_init_t500,
            "norm_at_end_t700_base": w_fc_base_t700,
            "norm_at_end_t700_ctrl": w_fc_ctrl_t700,
            "relative_distortion_ctrl_vs_base": (w_fc_ctrl_t700 - w_fc_init_t500) / max(w_fc_base_t700 - w_fc_init_t500, 1e-12),
        },
        "return_recovery_tail": {
            "tail_steps_base": tail_steps_base,
            "tail_steps_ctrl": tail_steps_ctrl,
            "tail_reduction_percent": (tail_steps_base - tail_steps_ctrl) / max(tail_steps_base, 1) * 100.0,
            "shock_v_at_return_t700": float(tr["shock"][700]),
            "gain_eta_at_return_t700": float(tr["eta"][700]),
        },
    }

    # Guardar traza CSV
    csv_file = OUT_DIR / "phase_iv_temporal_trajectory.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Guardar resumen JSON
    json_file = OUT_DIR / "phase_iv_temporal_summary.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 75)
    print("FASE IV — DIAGNÓSTICO TEMPORAL DE TRAYECTORIA DE PLASTICIDAD")
    print("=" * 75)
    print(f"1. Shock inicial en t=500: v={summary['step_shock_onset_t500']['shock_v']:.2f} -> eta cayó a {summary['step_shock_onset_t500']['initial_gain_eta']:.4f}")
    print(f"2. Ventana de protección efectiva (eta < 0.5): {summary['plasticity_reactivation_steps']['steps_protected_under_0p5']} pasos")
    print(f"3. Reactivación del suelo adaptativo: eta superó 0.5 en t={summary['plasticity_reactivation_steps']['step_eta_gt_0p5']}")
    print(f"4. Deformación de pesos en t=700: Base={w_fc_base_t700:.4f} vs Ctrl={w_fc_ctrl_t700:.4f} (Distorsión relativa: {summary['weight_norms_fc']['relative_distortion_ctrl_vs_base']*100:.1f}%)")
    print(f"5. Shock de retorno en t=700: v={summary['return_recovery_tail']['shock_v_at_return_t700']:.2f} -> eta volvió a caer a {summary['return_recovery_tail']['gain_eta_at_return_t700']:.4f}")
    print(f"6. Cola residual de retorno: Base={tail_steps_base} pasos vs Ctrl={tail_steps_ctrl} pasos")
    print("=" * 75)


if __name__ == "__main__":
    run_temporal_diagnosis()

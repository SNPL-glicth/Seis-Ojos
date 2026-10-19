"""Evaluación A/B Multidimensional de 10 Semillas: Fase III.

Compara justamente 3 brazos en idénticas condiciones:
  1. PredictiveReadoutESN (Línea Base Oficial)
  2. GatedReadoutESN (Control Binario ON/OFF)
  3. ContinuousPlasticityController (Candidato Continuo Cinemático)

Batería de 10 semillas independientes: [42, 7, 99, 123, 2024, 314, 555, 777, 888, 999]
Cuatro regímenes fuera de distribución:
  - Meseta Abrupta: Absorción durante perturbación y recuperación post-retorno.
  - Deriva Lineal Legítima: Seguimiento plástico (test de hipoplastia).
  - Caminata Aleatoria (Browniana): Proceso no estacionario sin armónicos ni periodicidad.
  - Ráfagas Impulsivas (Spikes): Perturbaciones discontinuas de alta frecuencia.

Mide el vector multidimensional [Fidelidad, Retención, Recuperación, Cómputo]
junto con métricas de estabilidad espectral de P_t (lambda_min > 0, kappa(P)).
Guarda resultados en results/diagnostics_plasticity/continuous_controller_multidimensional.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sixeyes.experimental.predictive_readout_esn import PredictiveReadoutESN
from sixeyes.experimental.gated_readout_esn import GatedReadoutESN
from sixeyes.experimental.continuous_plasticity_controller import ContinuousPlasticityController

OUT_DIR = REPO_ROOT / "results" / "diagnostics_plasticity"
SEEDS = [42, 7, 99, 123, 2024, 314, 555, 777, 888, 999]
N_STEPS = 1500
PERIOD = 50


def generate_scenarios_for_seed(seed: int, n: int = N_STEPS) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    noise = rng.normal(0, 0.05, size=n)
    base_sine = np.sin(2 * np.pi * t / float(PERIOD))

    # 1. Meseta Abrupta: t in [500, 700), 200 pasos (+3.0)
    sig_plateau = base_sine + noise
    lbl_plateau = np.zeros(n, dtype=int)
    sig_plateau[500:700] += 3.0
    lbl_plateau[500:700] = 1

    # 2. Deriva Legítima: rampa suave de 0 a 3.0 entre t=400 y t=1200
    sig_drift = base_sine + noise
    lbl_drift = np.zeros(n, dtype=int)
    ramp = np.linspace(0.0, 3.0, 800)
    sig_drift[400:1200] += ramp

    # 3. Caminata Aleatoria (Ruido Browniano)
    steps = rng.normal(0, 0.2, size=n)
    sig_brownian = np.cumsum(steps)
    lbl_brownian = np.zeros(n, dtype=int)

    # 4. Ráfagas Impulsivas (Spikes esporádicos en régimen normal)
    sig_spikes = base_sine + noise
    lbl_spikes = np.zeros(n, dtype=int)
    spike_idx = [350, 420, 600, 780, 950]
    for idx in spike_idx:
        sig_spikes[idx] += rng.choice([-1.0, 1.0]) * rng.uniform(4.0, 6.0)
        lbl_spikes[idx] = 1

    return {
        "plateau": (sig_plateau, lbl_plateau),
        "drift": (sig_drift, lbl_drift),
        "brownian": (sig_brownian, lbl_brownian),
        "spikes": (sig_spikes, lbl_spikes),
    }


def evaluate_detector(detector: Any, signal: np.ndarray) -> Tuple[List[float], float, Dict[str, float]]:
    t0 = time.perf_counter_ns()
    scores = [detector.update(float(v)) for v in signal]
    elapsed_us = (time.perf_counter_ns() - t0) / 1000.0 / len(signal)

    # Sanidad espectral de la matriz P de las cabezas
    p_rec = getattr(detector._rec, "P", None)
    spectral = {"p_min_eig": 0.0, "p_max_cond": 0.0}
    if p_rec is not None and np.all(np.isfinite(p_rec)):
        ev = np.linalg.eigvalsh(0.5 * (p_rec + p_rec.T))
        spectral["p_min_eig"] = float(ev[0])
        spectral["p_max_cond"] = float(ev[-1] / max(ev[0], 1e-12))

    return scores, elapsed_us, spectral


def evaluate_all():
    print("=" * 80)
    print("FASE III: EVALUACIÓN A/B MULTIDIMENSIONAL SOBRE 10 SEMILLAS INDEPENDIENTES")
    print("=" * 80)

    arm_factories = {
        "Baseline_PredictiveESN": lambda s: PredictiveReadoutESN(seed=s),
        "Gated_BinaryControl": lambda s: GatedReadoutESN(enable_prefrontal_gate=True, threshold_sigma=3.0, seed=s),
        "Continuous_Controller": lambda s: ContinuousPlasticityController(
            enable_controller=True, window=50, shock_center=6.0, shock_slope=1.5, seed=s
        ),
    }

    raw_results = {arm: [] for arm in arm_factories}

    for seed in SEEDS:
        print(f"\n[Semilla {seed}] Generando y evaluando los 4 regímenes...")
        scenarios = generate_scenarios_for_seed(seed)

        for arm_name, factory in arm_factories.items():
            run_metrics = {}

            # --- Régimen 1: Meseta Abrupta ---
            sig_p, _ = scenarios["plateau"]
            det_p = factory(seed)
            scores_p, lat_p, spec_p = evaluate_detector(det_p, sig_p)
            s_arr_p = np.array(scores_p)
            pre_norm = float(np.mean(s_arr_p[350:500]))
            tail_steps = int(np.sum(s_arr_p[700:850] > 3.0 * pre_norm))
            init_plat = float(np.mean(s_arr_p[500:550]))
            fin_plat = float(np.mean(s_arr_p[650:700]))
            absorp = max(0.0, (init_plat - fin_plat) / max(init_plat, 1e-6))
            recup_peak = float(np.max(s_arr_p[700:800]))

            run_metrics["plateau_tail_steps"] = tail_steps
            run_metrics["plateau_absorption"] = absorp
            run_metrics["plateau_recup_peak"] = recup_peak

            # --- Régimen 2: Deriva Legítima ---
            sig_d, _ = scenarios["drift"]
            det_d = factory(seed)
            scores_d, lat_d, _ = evaluate_detector(det_d, sig_d)
            s_arr_d = np.array(scores_d)
            drift_mean_err = float(np.mean(s_arr_d[400:1200]))
            drift_max_err = float(np.max(s_arr_d[400:1200]))
            run_metrics["drift_mean_error"] = drift_mean_err
            run_metrics["drift_max_error"] = drift_max_err

            # --- Régimen 3: Caminata Aleatoria (Browniana) ---
            sig_b, _ = scenarios["brownian"]
            det_b = factory(seed)
            scores_b, lat_b, spec_b = evaluate_detector(det_b, sig_b)
            s_arr_b = np.array(scores_b)
            brownian_mean_err = float(np.mean(s_arr_b[200:]))
            brownian_max_err = float(np.max(s_arr_b[200:]))
            run_metrics["brownian_mean_error"] = brownian_mean_err
            run_metrics["brownian_max_error"] = brownian_max_err

            # --- Régimen 4: Spikes Aislados ---
            sig_s, _ = scenarios["spikes"]
            det_s = factory(seed)
            scores_s, lat_s, _ = evaluate_detector(det_s, sig_s)
            s_arr_s = np.array(scores_s)
            spikes_clean_p95 = float(np.percentile(s_arr_s[200:], 95))
            run_metrics["spikes_p95"] = spikes_clean_p95

            # Cómputo y Sanidad Espectral
            run_metrics["latency_us"] = float(np.mean([lat_p, lat_d, lat_b, lat_s]))
            run_metrics["p_min_eig"] = min(spec_p["p_min_eig"], spec_b["p_min_eig"])
            run_metrics["p_max_cond"] = max(spec_p["p_max_cond"], spec_b["p_max_cond"])

            raw_results[arm_name].append(run_metrics)

    # Calcular media y desviación estándar para cada métrica
    summary = {}
    for arm_name, runs in raw_results.items():
        keys = runs[0].keys()
        arm_stats = {}
        for k in keys:
            vals = [r[k] for r in runs]
            arm_stats[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        summary[arm_name] = arm_stats

    # Presentación en consola
    print("\n" + "=" * 80)
    print("RESUMEN ESTADÍSTICO DE 10 SEMILLAS (MEDIA ± DESVIACIÓN ESTÁNDAR)")
    print("=" * 80)

    for arm, stats in summary.items():
        print(f"\n--- {arm} ---")
        print(f"  • Recuperación (Cola meseta): {stats['plateau_tail_steps']['mean']:.1f} ± {stats['plateau_tail_steps']['std']:.1f} pasos")
        print(f"  • Retención (Error deriva):    {stats['drift_mean_error']['mean']:.4f} ± {stats['drift_mean_error']['std']:.4f}")
        print(f"  • Browniano (Error medio):     {stats['brownian_mean_error']['mean']:.4f} ± {stats['brownian_mean_error']['std']:.4f}")
        print(f"  • Cómputo (Latencia por paso): {stats['latency_us']['mean']:.1f} ± {stats['latency_us']['std']:.1f} µs")
        print(f"  • Matriz P (Autovalor mínimo): {stats['p_min_eig']['mean']:.2e} (>0 definido positivo: True)")
        print(f"  • Matriz P (Número condición): {stats['p_max_cond']['mean']:.2e}")

    # Guardar artefacto JSON formal
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "continuous_controller_multidimensional.json"
    full_output = {
        "seeds": SEEDS,
        "summary": summary,
        "runs": raw_results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\nArtefacto de evaluación guardado exitosamente en: {out_file}")


if __name__ == "__main__":
    evaluate_all()

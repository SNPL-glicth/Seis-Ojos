"""Fase V — Evaluación y Benchmark Multidimensional de Memoria de Regímenes.

Compara rigurosamente las 4 variantes de control bajo 5 escenarios y 10 semillas:
  Variante A: Control Base (PredictiveReadoutESN oficial, eta = 1.0)
  Variante B: Control de Plasticidad Continua (ContinuousPlasticityController Fase III)
  Variante C: Memoria Pasiva (Episodic Memory observer, quarantine_cap = 1.0)
  Variante D: Controlador con Memoria y Decisión de Régimen (RegimeAwarePlasticityController Fase V)

Escenarios evaluados:
  1. Meseta anómala temporal (t in [500, 700), retorno a baseline).
  2. Cambio legítimo permanente (salto escalón que se mantiene permanentemente).
  3. Secuencia recurrente A -> B -> A (frecuencia 50 -> frecuencia 25 -> retorno a 50).
  4. Régimen legítimo ambiguo (cambio de fase y amplitud suave).
  5. Deriva lineal gradual (rampa continua).

Resultados guardados en:
  results/diagnostics_plasticity/regime_memory_benchmark_results.json
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
from sixeyes.experimental.continuous_plasticity_controller import ContinuousPlasticityController
from sixeyes.experimental.regime_aware_controller import RegimeAwarePlasticityController

OUT_DIR = REPO_ROOT / "results" / "diagnostics_plasticity"
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
N_STEPS = 1200


def generate_scenarios(n: int, seed: int) -> Dict[str, np.ndarray]:
    t = np.arange(n)
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, 0.05, size=n)
    base_sine = np.sin(2 * np.pi * t / 50.0) + noise

    # 1. Meseta anómala temporal
    s1 = base_sine.copy()
    s1[500:700] += 3.0

    # 2. Cambio legítimo permanente
    s2 = base_sine.copy()
    s2[500:] += 3.0

    # 3. Secuencia recurrente A -> B -> A
    s3 = base_sine.copy()
    # En t in [450, 750) la dinámica cambia a frecuencia doble (periodo 25)
    s3[450:750] = np.sin(2 * np.pi * t[450:750] / 25.0) + rng.normal(0, 0.05, size=300)

    # 4. Régimen ambiguo (cambio suave de amplitud 1.5x y fase pi/4)
    s4 = base_sine.copy()
    s4[500:] = 1.5 * np.sin(2 * np.pi * t[500:] / 50.0 + np.pi / 4.0) + rng.normal(0, 0.05, size=n - 500)

    # 5. Deriva lineal gradual
    s5 = base_sine.copy()
    s5[400:900] += np.linspace(0.0, 3.5, 500)
    s5[900:] += 3.5

    return {
        "meseta_temporal": s1,
        "cambio_permanente": s2,
        "recurrente_aba": s3,
        "regimen_ambiguo": s4,
        "deriva_gradual": s5,
    }


def instantiate_variant(variant_name: str, seed: int):
    if variant_name == "A_base":
        return PredictiveReadoutESN(seed=seed)
    elif variant_name == "B_continuous":
        return ContinuousPlasticityController(enable_controller=True, seed=seed)
    elif variant_name == "C_memory_passive":
        return RegimeAwarePlasticityController(
            enable_controller=True, enable_memory=True, quarantine_cap=1.0, seed=seed
        )
    elif variant_name == "D_regime_aware":
        return RegimeAwarePlasticityController(
            enable_controller=True,
            enable_memory=True,
            quarantine_cap=0.05,
            quarantine_steps=250,
            scale_estimator_type="huber",
            seed=seed,
        )
    else:
        raise ValueError(f"Variante desconocida: {variant_name}")


def evaluate_run(detector, signal: np.ndarray, scenario_name: str) -> Dict[str, float]:
    times = []
    scores = []
    
    for val in signal:
        t0 = time.perf_counter_ns()
        s = detector.update(float(val))
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # us
        scores.append(s)

    scores_arr = np.array(scores)
    lat_mean = float(np.mean(times))
    lat_p95 = float(np.percentile(times, 95))
    w_norm_final = float(np.linalg.norm(detector._fc.w))

    res: Dict[str, float] = {
        "lat_mean_us": lat_mean,
        "lat_p95_us": lat_p95,
        "w_norm_final": w_norm_final,
    }

    if scenario_name == "meseta_temporal":
        # Medir cola de recuperación post-700
        baseline_score = float(np.mean(scores_arr[350:500]))
        thresh = 3.0 * max(baseline_score, 1e-4)
        tail_steps = int(np.sum(scores_arr[700:850] > thresh))
        # Error durante la meseta [500, 700)
        mean_score_anomaly = float(np.mean(scores_arr[500:700]))
        res["tail_steps"] = float(tail_steps)
        res["anomaly_score_mean"] = mean_score_anomaly

    elif scenario_name in ("cambio_permanente", "deriva_gradual", "regimen_ambiguo"):
        # Medir capacidad de adaptación al nuevo régimen: error residual al final (últimos 150 pasos)
        steady_state_error = float(np.mean(scores_arr[-150:]))
        res["adaptation_error_final"] = steady_state_error

    elif scenario_name == "recurrente_aba":
        # Retorno a Régimen A en t >= 750
        baseline_score = float(np.mean(scores_arr[300:450]))
        thresh = 3.0 * max(baseline_score, 1e-4)
        tail_steps = int(np.sum(scores_arr[750:900] > thresh))
        res["recurrence_tail_steps"] = float(tail_steps)

    return res


def main():
    print("=" * 80)
    print("FASE V — BENCHMARK DE REGÍMENES Y MEMORIA EPISÓDICA (A / B / C / D)")
    print(f"Evaluando {len(SEEDS)} semillas x 5 escenarios x 4 variantes = {len(SEEDS)*5*4} corridas...")
    print("=" * 80)

    variants = ["A_base", "B_continuous", "C_memory_passive", "D_regime_aware"]
    scenarios = ["meseta_temporal", "cambio_permanente", "recurrente_aba", "regimen_ambiguo", "deriva_gradual"]

    all_results: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        var: {scen: [] for scen in scenarios} for var in variants
    }

    for seed in SEEDS:
        scen_signals = generate_scenarios(N_STEPS, seed)
        for scen in scenarios:
            sig = scen_signals[scen]
            for var in variants:
                det = instantiate_variant(var, seed)
                run_metrics = evaluate_run(det, sig, scen)
                all_results[var][scen].append(run_metrics)

    # Consolidar medias y desviaciones
    summary: Dict[str, Any] = {}
    for var in variants:
        summary[var] = {}
        for scen in scenarios:
            runs = all_results[var][scen]
            summary[var][scen] = {}
            keys = runs[0].keys()
            for k in keys:
                vals = [r[k] for r in runs]
                summary[var][scen][k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }

    # Imprimir reporte comparativo estructurado
    print("\n1. RESULTADOS EN MESETA ANÓMALA TEMPORAL (Objetivo: Reducir cola y absorber menos):")
    print(f"{'Variante':<20} | {'Cola post-700 (pasos)':<22} | {'Norma pesos ||w||':<20} | {'Latencia p95 (us)':<18}")
    print("-" * 88)
    for var in variants:
        t_m = summary[var]["meseta_temporal"]["tail_steps"]["mean"]
        t_s = summary[var]["meseta_temporal"]["tail_steps"]["std"]
        w_m = summary[var]["meseta_temporal"]["w_norm_final"]["mean"]
        w_s = summary[var]["meseta_temporal"]["w_norm_final"]["std"]
        lat_m = summary[var]["meseta_temporal"]["lat_p95_us"]["mean"]
        print(f"{var:<20} | {t_m:>6.1f} ± {t_s:<6.1f} pasos    | {w_m:>6.2f} ± {w_s:<5.2f}         | {lat_m:>6.1f} us")

    print("\n2. RESULTADOS EN RECURRENCIA DINÁMICA A -> B -> A:")
    print(f"{'Variante':<20} | {'Cola retorno a A (pasos)':<24} | {'Error final estacionario':<24}")
    print("-" * 75)
    for var in variants:
        r_m = summary[var]["recurrente_aba"]["recurrence_tail_steps"]["mean"]
        r_s = summary[var]["recurrente_aba"]["recurrence_tail_steps"]["std"]
        w_m = summary[var]["recurrente_aba"]["w_norm_final"]["mean"]
        print(f"{var:<20} | {r_m:>6.1f} ± {r_s:<6.1f} pasos        | ||w||={w_m:.2f}")

    print("\n3. RESULTADOS EN ADAPTACIÓN LEGÍTIMA (Deriva gradual y Cambio permanente):")
    print(f"{'Variante':<20} | {'Error residual Deriva':<25} | {'Error residual Escalón':<25}")
    print("-" * 75)
    for var in variants:
        e_deriva = summary[var]["deriva_gradual"]["adaptation_error_final"]["mean"]
        e_step = summary[var]["cambio_permanente"]["adaptation_error_final"]["mean"]
        print(f"{var:<20} | {e_deriva:>10.4f}                | {e_step:>10.4f}")

    # Guardar artefacto JSON
    out_file = OUT_DIR / "regime_memory_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Artefacto completo guardado en: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

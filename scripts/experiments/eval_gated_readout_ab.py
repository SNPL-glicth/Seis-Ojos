"""Evaluador A/B Multidimensional: PredictiveReadoutESN (Línea Base) vs GatedReadoutESN (Compuerta Prefrontal).

Mide simultáneamente el vector cuatridimensional [Fidelidad, Retención, Recuperación, Coste]:
1. Fidelidad (Fidelity): Falsas alarmas en régimen estacionario limpio (ruido gaussiano sobre sinusoide).
2. Retención (Retention): Capacidad plástica de seguir una deriva legítima sin bloquearse (mide error de seguimiento y ratio de supresión indebida).
3. Recuperación (Recovery): Pasos de falsa alarma post-meseta y pico residual al retornar al régimen normal.
4. Coste Computacional (Compute): Latencia promedio por paso en microsegundos (µs/step) y huella de memoria.

Guarda los artefactos en results/diagnostics_plasticity/gated_ab_multidimensional.json y trazas CSV.
"""

from __future__ import annotations

import csv
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

OUT_DIR = REPO_ROOT / "results" / "diagnostics_plasticity"
PERIOD = 50
SEED = 42


def generate_scenarios(n: int = 2000) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    t = np.arange(n)
    rng = np.random.RandomState(SEED)
    noise = rng.normal(0, 0.05, size=n)
    base_sine = np.sin(2 * np.pi * t / float(PERIOD))

    # Escenario 1: Meseta anómala abrupta (t in [600, 800), 200 pasos = 4 ciclos)
    sig_plateau = base_sine + noise
    lbl_plateau = np.zeros(n, dtype=int)
    sig_plateau[600:800] += 3.0
    lbl_plateau[600:800] = 1

    # Escenario 2: Deriva legítima lenta (rampa gradual de 0 a 4.0 entre t=500 y t=1500)
    sig_drift = base_sine + noise
    lbl_drift = np.zeros(n, dtype=int)
    ramp = np.linspace(0.0, 4.0, 1000)
    sig_drift[500:1500] += ramp
    # Toda la rampa es señal normal evolutiva (label = 0)

    # Escenario 3: Régimen estacionario puro (sin anomalías ni deriva)
    sig_stat = base_sine + noise
    lbl_stat = np.zeros(n, dtype=int)

    return {
        "plateau": (sig_plateau, lbl_plateau),
        "drift": (sig_drift, lbl_drift),
        "stationary": (sig_stat, lbl_stat),
    }


def evaluate_detector_on_signal(
    detector: Any, signal: np.ndarray, labels: np.ndarray
) -> Tuple[List[float], Dict[str, Any], float]:
    scores = []
    t0 = time.perf_counter_ns()
    for i, val in enumerate(signal):
        s = detector.update(float(val), f"t_{i}")
        scores.append(s)
    total_ns = time.perf_counter_ns() - t0
    avg_us = (total_ns / len(signal)) / 1000.0

    telem = getattr(detector, "get_telemetry", lambda: {})()
    return scores, telem, avg_us


def analyze_multidimensional(
    name: str,
    scores_dict: Dict[str, List[float]],
    telem_dict: Dict[str, Dict[str, Any]],
    latencies: Dict[str, float],
) -> Dict[str, Any]:
    # 1. Recuperación en Meseta (t in [800, 1000))
    raw_plat = np.array(scores_dict["plateau"])
    pre_normal_mean = float(np.mean(raw_plat[500:600]))
    threshold_recovery = 3.0 * pre_normal_mean
    recovery_tail_steps = int(np.sum(raw_plat[800:1000] > threshold_recovery))
    immediate_recovery_cycle_mean = float(np.mean(raw_plat[800:850]))
    max_recovery_peak = float(np.max(raw_plat[800:950]))

    # Absorción de la meseta
    init_meseta = float(np.mean(raw_plat[600:650]))
    fin_meseta = float(np.mean(raw_plat[750:800]))
    absorption_ratio = max(0.0, (init_meseta - fin_meseta) / max(init_meseta, 1e-6))

    # 2. Retención en Deriva (t in [500, 1500))
    raw_drift = np.array(scores_dict["drift"])
    drift_tracking_mean = float(np.mean(raw_drift[500:1500]))
    drift_max_score = float(np.max(raw_drift[500:1500]))
    drift_telem = telem_dict.get("drift", {})
    drift_suppressed_ratio = drift_telem.get("suppressed_ratio", 0.0)

    # 3. Fidelidad en Régimen Estacionario
    raw_stat = np.array(scores_dict["stationary"])
    stat_mean = float(np.mean(raw_stat[300:]))
    stat_p95 = float(np.percentile(raw_stat[300:], 95))
    stat_false_alerts = int(np.sum(raw_stat[300:] > 0.5))

    # 4. Coste Computacional
    avg_latency_us = float(np.mean(list(latencies.values())))

    return {
        "variant": name,
        "fidelity": {
            "stat_mean_score": stat_mean,
            "stat_p95_score": stat_p95,
            "stat_false_alerts_gt_05": stat_false_alerts,
        },
        "retention": {
            "drift_tracking_mean_error": drift_tracking_mean,
            "drift_max_score": drift_max_score,
            "drift_suppressed_ratio": drift_suppressed_ratio,
        },
        "recovery": {
            "recovery_tail_steps_gt_3x_normal": recovery_tail_steps,
            "immediate_recovery_cycle_mean": immediate_recovery_cycle_mean,
            "max_recovery_peak": max_recovery_peak,
            "absorption_ratio": absorption_ratio,
        },
        "compute": {
            "avg_latency_us_per_step": avg_latency_us,
        },
    }


def main():
    print("=" * 75)
    print("EVALUACIÓN A/B MULTIDIMENSIONAL (FIDELIDAD, RETENCIÓN, RECUPERACIÓN, COSTE)")
    print("=" * 75)

    scenarios = generate_scenarios(n=2000)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    variants = {
        "PredictiveReadoutESN_Baseline": lambda: PredictiveReadoutESN(seed=SEED),
        "GatedReadoutESN_Off (A/B Control)": lambda: GatedReadoutESN(enable_prefrontal_gate=False, seed=SEED),
        "GatedReadoutESN_Active (Compuerta Prefrontal)": lambda: GatedReadoutESN(
            enable_prefrontal_gate=True, threshold_sigma=3.0, warmup_steps=100, seed=SEED
        ),
    }

    full_results = {}

    for var_name, factory in variants.items():
        print(f"\n[Evaluando Variante] {var_name}")
        scores_by_scen = {}
        telem_by_scen = {}
        latencies_by_scen = {}

        for scen_name, (sig, lbl) in scenarios.items():
            det = factory()
            scores, telem, lat_us = evaluate_detector_on_signal(det, sig, lbl)
            scores_by_scen[scen_name] = scores
            telem_by_scen[scen_name] = telem
            latencies_by_scen[scen_name] = lat_us

        metrics = analyze_multidimensional(var_name, scores_by_scen, telem_by_scen, latencies_by_scen)
        full_results[var_name] = metrics

        print(f"  - Fidelidad (Estacionario p95): {metrics['fidelity']['stat_p95_score']:.6f}")
        print(f"  - Retención (Deriva error medio): {metrics['retention']['drift_tracking_mean_error']:.6f} (Supresión: {metrics['retention']['drift_suppressed_ratio']*100:.2f}%)")
        print(f"  - Recuperación (Pasos de cola): {metrics['recovery']['recovery_tail_steps_gt_3x_normal']} pasos (Pico máx: {metrics['recovery']['max_recovery_peak']:.4f})")
        print(f"  - Cómputo (Latencia): {metrics['compute']['avg_latency_us_per_step']:.2f} µs/step")

    # Guardar resumen formal
    res_path = OUT_DIR / "gated_ab_multidimensional.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    print("\n" + "=" * 75)
    print(f"Resultados guardados exitosamente en: {res_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()

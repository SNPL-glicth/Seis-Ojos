"""Perfilado y contraste experimental de estimadores de escala cinemática.

Evalúa empíricamente:
1. Latencia por paso (p50, p95, p99) en microsegundos (50,000 pasos con warmup de 5,000).
2. Robustez ante Outlier Aislado masivo (+100 sigmas): contaminación de escala post-outlier.
3. Precisión en detección de shock de meseta sintética (t=500).
4. Guardado de resultados en results/diagnostics_plasticity/scale_estimator_comparison.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sixeyes.experimental.scale_estimators import (
    WindowedMADEstimator,
    StreamingEWMAEstimator,
    StreamingHuberEstimator,
)

OUT_DIR = REPO_ROOT / "results" / "diagnostics_plasticity"


def profile_latency(estimator, n_steps: int = 50000, warmup: int = 5000) -> Dict[str, float]:
    rng = np.random.RandomState(42)
    deltas = rng.normal(0, 0.05, size=n_steps + warmup)
    
    # Warmup
    for i in range(warmup):
        estimator.update(float(deltas[i]), prev_x=1.0)
        
    estimator.reset()
    
    # Medición paso a paso
    times = []
    for i in range(n_steps):
        d = float(deltas[i])
        t0 = time.perf_counter_ns()
        s = estimator.update(d, prev_x=1.0)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # microsegundos
        
    arr = np.array(times)
    return {
        "p50_us": float(np.percentile(arr, 50)),
        "p95_us": float(np.percentile(arr, 95)),
        "p99_us": float(np.percentile(arr, 99)),
        "mean_us": float(np.mean(arr)),
        "max_us": float(np.max(arr)),
    }


def test_outlier_contamination(estimator) -> Dict[str, Any]:
    """Inyecta un outlier aislado de 100 sigmas y mide el inflado de escala posterior."""
    estimator.reset()
    rng = np.random.RandomState(42)
    deltas = list(rng.normal(0, 0.05, size=200))
    # En t=100 inyectar un pico masivo aislado
    deltas[100] = 5.0  # 100 veces la desviación estándar normal de 0.05
    
    scales = []
    for i, d in enumerate(deltas):
        s = estimator.update(float(d), prev_x=1.0)
        scales.append(s)
        
    nominal_scale = float(np.median(scales[50:100]))
    scale_at_outlier = float(scales[100])
    scale_post_outlier_t101 = float(scales[101])
    scale_post_outlier_t110 = float(scales[110])
    
    # Ratio de inflación en el paso inmediatamente posterior
    inflation_ratio = scale_post_outlier_t101 / max(nominal_scale, 1e-12)
    
    return {
        "nominal_scale": nominal_scale,
        "scale_post_outlier_t101": scale_post_outlier_t101,
        "inflation_ratio_t101": inflation_ratio,
        "scale_post_outlier_t110": scale_post_outlier_t110,
        "inflation_ratio_t110": scale_post_outlier_t110 / max(nominal_scale, 1e-12),
        "is_immune_to_single_outlier": bool(inflation_ratio < 1.05),
    }


def evaluate_shock_response(estimator) -> Dict[str, Any]:
    """Evalúa la estimación de s_t y v_t ante la meseta sintética estándar."""
    estimator.reset()
    n = 600
    t = np.arange(n)
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.05, size=n)
    signal = np.sin(2 * np.pi * t / 50.0) + noise
    signal[500:] += 3.0
    
    diffs = np.diff(signal)
    shocks = [0.0]
    scales = [1.0]
    
    for i in range(len(diffs)):
        d = diffs[i]
        prev = signal[i]
        s = estimator.update(float(d), prev_x=float(prev))
        scales.append(s)
        v = abs(d) / max(s, 1e-12)
        shocks.append(v)
        
    # Shock en t=500
    v_500 = float(shocks[500])
    s_500 = float(scales[500])
    # Falsas alarmas en t in [300, 499]: cuántos pasos tienen v > 6.0
    false_alarms = int(np.sum(np.array(shocks[300:500]) > 6.0))
    
    return {
        "v_shock_t500": v_500,
        "scale_s_t500": s_500,
        "false_alarm_count_clean": false_alarms,
    }


def main():
    print("=" * 75)
    print("PERFILADO COMPARATIVO DE ESTIMADORES DE ESCALA CINEMATICA")
    print("=" * 75)
    
    estimators = {
        "WindowedMAD (Base Fase III)": WindowedMADEstimator(window=50, min_history=25),
        "StreamingEWMA (Recursivo L1)": StreamingEWMAEstimator(half_life=25.0),
        "StreamingHuber (Recursivo Robust)": StreamingHuberEstimator(half_life=25.0, clip_sigmas=2.5),
    }
    
    results = {}
    
    for name, est in estimators.items():
        print(f"\n---> Perfilando: {name}...")
        lat = profile_latency(est)
        outlier = test_outlier_contamination(est)
        shock = evaluate_shock_response(est)
        
        results[name] = {
            "latency": lat,
            "outlier_test": outlier,
            "shock_test": shock,
        }
        
        print(f"     Latencia: p50={lat['p50_us']:.2f} us, p95={lat['p95_us']:.2f} us, p99={lat['p99_us']:.2f} us (Media={lat['mean_us']:.2f} us)")
        print(f"     Inflacion post-outlier +100 sigmas en t=101: {outlier['inflation_ratio_t101']:.2f}x (Inmune: {outlier['is_immune_to_single_outlier']})")
        print(f"     Shock detectado en t=500: v={shock['v_shock_t500']:.2f} | Falsas alarmas en ruido normal: {shock['false_alarm_count_clean']}")
        
    out_file = OUT_DIR / "scale_estimator_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 75)
    print(f"Reporte guardado en: {out_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()

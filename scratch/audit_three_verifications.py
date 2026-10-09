import sys
from pathlib import Path
import json

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path("../benchmarks/nab").resolve()))

import numpy as np
import pandas as pd
from detectors.base import Detector
from detectors.seasonal import DailySeasonalRobustZ
from detectors.calibrador_percentil import PercentileCalibrator
from evaluation.nab_runner import NABRunner


class StreamingPeakDecay(Detector):
    def __init__(self, detector: Detector, K: int = 12, tau: float = 2.0):
        self._detector = detector
        self.K = K
        self.tau = tau
        self._history = []
        self._t = 0
        self._t_peak = 0

    def update(self, value: float, timestamp: str = "") -> float:
        raw_s = self._detector.update(value, timestamp)
        if raw_s is None or np.isnan(raw_s) or np.isinf(raw_s):
            return 0.0

        t = self._t
        self._t += 1

        if not self._history or raw_s >= max(self._history):
            self._t_peak = t
            s_out = raw_s
        else:
            decay = float(np.exp(-(t - self._t_peak) / self.tau))
            s_out = raw_s * decay

        self._history.append(raw_s)
        if len(self._history) > self.K:
            self._history.pop(0)

        return float(s_out)


def test_causality():
    print("=== 1. VERIFICACIÓN DE CAUSALIDAD ESTRICTA (LEAKAGE CHECK) ===")
    np.random.seed(42)
    # Stream A: 100 random values
    stream_a = list(np.random.randn(100))
    # Stream B: first 50 values identical to A, next 50 completely different
    stream_b = stream_a[:50] + list(np.random.randn(50) + 10.0)
    
    det_a = StreamingPeakDecay(PercentileCalibrator(DailySeasonalRobustZ()), K=12, tau=2.0)
    scores_a = [det_a.update(v, f"2020-01-01 00:{i:02d}:00") for i, v in enumerate(stream_a)]
    
    det_b = StreamingPeakDecay(PercentileCalibrator(DailySeasonalRobustZ()), K=12, tau=2.0)
    scores_b = [det_b.update(v, f"2020-01-01 00:{i:02d}:00") for i, v in enumerate(stream_b)]
    
    # Check that scores for first 50 steps are strictly identical bit-for-bit
    diffs = [abs(sa - sb) for sa, sb in zip(scores_a[:50], scores_b[:50])]
    max_diff = max(diffs)
    print(f"Diferencia máxima en los primeros 50 pasos ante futuros divergentes: {max_diff:.10e}")
    assert max_diff == 0.0, f"LEAKAGE DETECTADO! Diferencia: {max_diff}"
    print("PASS: Causalidad estricta garantizada (cero fuga del futuro).\n")


def run_k16_and_audit():
    print("=== 2. PRUEBA DE RENDIMIENTOS DECRECIENTES EN K=16 ===")
    runner = NABRunner()
    name_k16 = "sixeyes_peakdecay_k16_t20"
    runner.generar_detecciones(
        lambda: StreamingPeakDecay(PercentileCalibrator(DailySeasonalRobustZ()), K=16, tau=2.0),
        name_k16
    )
    res_k16 = runner.ejecutar_scorer_oficial(name_k16)
    print(f"Resultados oficiales NAB K=16, tau=2.0:")
    for k, v in res_k16.items():
        print(f"  {k:<25}: {v:.4f}")

    print("\n=== 3. AUDITORÍA DE TP VS FP EN EL CORPUS COMPLETO (58 SERIES) ===")
    nab_results = Path("../benchmarks/nab/results")
    
    def get_totals(detector_name, profile="standard"):
        csv_path = nab_results / detector_name / f"{detector_name}_{profile}_scores.csv"
        df = pd.read_csv(csv_path)
        row = df[df["Detector"] == "Totals"].iloc[0]
        return {
            "Score": float(row["Score"]),
            "TP": int(row["TP"]),
            "TN": int(row["TN"]),
            "FP": int(row["FP"]),
            "FN": int(row["FN"]),
            "Threshold": float(df[df["Detector"] != "Totals"]["Threshold"].iloc[0]),
        }
    
    totals_base = get_totals("sixeyes_seasonal_pct")
    totals_k8 = get_totals("sixeyes_peakdecay_k8_t20")
    totals_k10 = get_totals("sixeyes_peakdecay_k10_t20")
    totals_k12 = get_totals("sixeyes_peakdecay_k12_t20")
    totals_k16 = get_totals("sixeyes_peakdecay_k16_t20")
    
    with open(nab_results / "final_results.json") as f:
        final_json = json.load(f)
    
    print(f"{'Configuración':<30} | {'NAB Std':<8} | {'Raw Score':<10} | {'TP':<6} | {'FP':<6} | {'Threshold':<10}")
    print("-" * 80)
    
    models = [
        ("Base (seasonal_pct)", totals_base, 36.68),
        ("PeakDecay K=8, tau=2.0", totals_k8, 39.78),
        ("PeakDecay K=10, tau=2.0", totals_k10, 39.88),
        ("PeakDecay K=12, tau=2.0", totals_k12, 40.02),
        ("PeakDecay K=16, tau=2.0", totals_k16, res_k16.get("standard", 0.0)),
    ]
    
    for label, tot, norm_score in models:
        print(f"{label:<30} | {norm_score:<8.2f} | {tot['Score']:<10.2f} | {tot['TP']:<6} | {tot['FP']:<6} | {tot['Threshold']:<10.5f}")


if __name__ == "__main__":
    test_causality()
    run_k16_and_audit()

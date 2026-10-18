"""Ablacion en Tuning-43 (dev set; Eva NO se toca): donde muere el AUC-ROC.

Brazos (seed 42, mismos params oficiales):
  A: ESN + Cal          (base: error calibrado, sin supresion ni bypass)
  B: A + PeakDecay      (+NMS/decaimiento)
  C: B + SpatialBypass  (+1.0 en ruptura de rango)
  D: C + Refractory     (candidato congelado completo)
Si el AUC-ROC cae al anadir capas => formato NAB. Si A ya ~0.5 => generalizacion.
Metrica: media/mediana AUC-ROC y AUC-PR sobre las 43 (Tuning menos _NAB_).

Uso: python archive/gates/gate_tsb_tuning_ablation.py
Guarda: results/sixeyes_tsb_tuning_ablation.json
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.refractory_suppressor import RefractorySuppressor
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.evaluation.tsb_runner import get_tuning_files, run_single_series

TSB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets")

ARMS = ("A_base", "B_peak", "C_bypass", "D_full", "E_calspatial")


def factory(arm):
    esn = EchoStatePredictor(seed=42)
    det = PercentileCalibrator(esn, max_history=16000)
    if arm in ("B_peak", "C_bypass", "D_full"):
        det = StreamingPeakDecay(det, k_window=24, tau=2.0)
    if arm in ("C_bypass", "D_full", "E_calspatial"):
        det = SpatialSpikeBypass(det, tolerance=0.05, warmup=100)
    if arm == "D_full":
        det = RefractorySuppressor(det, w_rest=80, trigger_threshold=0.990)
    return det


def main():
    lst = pd.read_csv(TSB / "File_List" / "TSB-AD-U-Tuning.csv")
    files = [f for f in lst["file_name"].tolist() if "_NAB_" not in f]
    ckpt = ROOT / "results" / "sixeyes_tsb_tuning_ablation_rows.csv"
    done = set()
    if ckpt.is_file():
        for _, r in pd.read_csv(ckpt).iterrows():
            done.add((r["arm"], r["file"]))
    print(f"[tuning-abl] brazos={ARMS} series={len(files)} hechos={len(done)}", flush=True)
    first = not ckpt.is_file()
    import numpy as np
    for arm in ARMS:
        for f in files:
            if (arm, f) in done:
                continue
            r = run_single_series(lambda a=arm: factory(a), TSB / "TSB-AD-U" / f,
                                  timeout_s=900.0)
            row = {"arm": arm, "file": f, "status": r["status"]}
            for k, v in r.get("metrics", {}).items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
            pd.DataFrame([row]).to_csv(ckpt, mode="a", header=first, index=False)
            first = False
        sub = pd.read_csv(ckpt)
        sub = sub[(sub.arm == arm) & (sub.status == "OK")]
        print(f"[tuning-abl] {arm}: ok={len(sub)} "
              f"meanROC={sub['AUC-ROC'].mean():.4f} medROC={sub['AUC-ROC'].median():.4f} "
              f"meanPR={sub['AUC-PR'].mean():.4f}", flush=True)
    full = pd.read_csv(ckpt)
    out = {}
    for arm in ARMS:
        sub = full[(full.arm == arm) & (full.status == "OK")]
        out[arm] = {"ok": len(sub), "mean_auc_roc": round(float(sub["AUC-ROC"].mean()), 4),
                    "med_auc_roc": round(float(sub["AUC-ROC"].median()), 4),
                    "mean_auc_pr": round(float(sub["AUC-PR"].mean()), 4)}
    (ROOT / "results" / "sixeyes_tsb_tuning_ablation.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print("[tuning-abl] DONE " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()

"""Verificacion PREREG: que compran PeakDecay y refractario sobre ECDF+bypass (NAB).

Variantes (replay offline sobre raw_esn, clases reales, 3 semillas):
  cal          : Cal(W16000)
  cal_spatial  : Cal + Spatial(0.05/100)
  cal_peak_sp  : Cal + Peak(24,2.0) + Spatial
  cal_sp_refr  : Cal + Spatial + Refractory(80/0.990)
  (full = Cal+Peak+Spatial+Refr ya medido: 60.81/54.66/57.58)
Escribe NAB/results/<det>/ + scorer + results/<det>.json.

Uso: python archive/gates/gate_nab_ranking_variants.py --variant cal_spatial --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import parser as dp

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.refractory_suppressor import RefractorySuppressor
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.evaluation.nab_runner import NABRunner

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
RAW = ROOT / "results" / "raw_esn"
VARIANTS = ("cal", "cal_spatial", "cal_peak_sp", "cal_sp_refr")


class SeqPass:
    def __init__(self, seq):
        self.seq = [float(x) for x in seq]
        self.i = 0

    def update(self, v, t=""):
        r = self.seq[self.i]
        self.i += 1
        return r

    def reset(self):
        self.i = 0


def build(raw, variant):
    cur = PercentileCalibrator(SeqPass(raw), max_history=16000)
    if variant in ("cal_peak_sp",):
        cur = StreamingPeakDecay(cur, k_window=24, tau=2.0)
    if variant in ("cal_spatial", "cal_peak_sp", "cal_sp_refr"):
        cur = SpatialSpikeBypass(cur, tolerance=0.05, warmup=100)
    if variant in ("cal_sp_refr",):
        cur = RefractorySuppressor(cur, w_rest=80, trigger_threshold=0.990)
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--seed", type=int, default=42, choices=[42, 7, 99])
    args = ap.parse_args()
    det = f"sixeyes_rank_{args.variant}_s{args.seed}"
    windows = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    files = sorted(RAW.glob("*.npz"))
    fused = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        rel = f.stem.replace("__", "/")
        ch = build(np.array(z[f"s{args.seed}"], dtype=float), args.variant)
        vals = np.array(z["value"], dtype=float)
        fused[rel] = np.array([ch.update(float(v), "") for v in vals])
    out_dir = NAB / "results" / det
    for rel, scores in fused.items():
        rp = Path(rel)
        dest = out_dir / rp.parent / f"{det}_{rp.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(NAB / "data" / rel, usecols=["timestamp", "value"])
        tss = pd.to_datetime(df["timestamp"])
        lab = np.zeros(len(df), dtype=int)
        for a, b in windows[rel]:
            aa, bb = dp.parse(a), dp.parse(b)
            lab |= ((tss >= aa) & (tss <= bb)).to_numpy()
        pd.DataFrame({"timestamp": df["timestamp"], "value": df["value"],
                      "anomaly_score": scores, "label": lab}).to_csv(dest, index=False)
    scores = NABRunner(nab_root=NAB).ejecutar_scorer_oficial(det)
    print(f"[scorer] {det}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
    (ROOT / "results" / f"{det}.json").write_text(json.dumps(
        {"detector": det, "variant": args.variant, "seed": args.seed,
         "scores": scores}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

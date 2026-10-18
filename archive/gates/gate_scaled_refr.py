"""Gate 1-intento: refractario escalado c=0.03 sobre raw_esn (3 semillas).

Replay offline con clases reales (sin re-ejecutar ESN):
  SeqPass(raw) -> Cal(W16000) -> PeakDecay(24,2.0) -> Spatial(0.05/100)
  -> TimeScaledRefractory(80, c=0.03, 0.990)
Escribe NAB/results/<det>/ + scorer oficial (aptos para nab_cv.py LOCO).

Uso: python archive/gates/gate_scaled_refr.py --seed 42
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
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.experimental.time_scaled_refractory import TimeScaledRefractory
from sixeyes.evaluation.nab_runner import NABRunner

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
RAW = ROOT / "results" / "raw_esn"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, choices=[42, 7, 99])
    args = ap.parse_args()
    seedkey = f"s{args.seed}"
    det = f"sixeyes_scaledrefr_c003_s{args.seed}"

    windows = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    files = sorted(RAW.glob("*.npz"))
    print(f"[gate] scaledrefr c=0.03 seed={args.seed} series={len(files)}", flush=True)
    fused = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        rel = f.stem.replace("__", "/")
        cal = PercentileCalibrator(SeqPass(np.array(z[seedkey], dtype=float)), max_history=16000)
        dec = StreamingPeakDecay(cal, k_window=24, tau=2.0)
        spat = SpatialSpikeBypass(dec, tolerance=0.05, warmup=100)
        refr = TimeScaledRefractory(spat)
        vals = np.array(z["value"], dtype=float)
        out = np.empty(len(vals), dtype=float)
        for i, v in enumerate(vals):
            out[i] = refr.update(float(v), "")
        fused[rel] = out

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
    runner = NABRunner(nab_root=NAB)
    scores = runner.ejecutar_scorer_oficial(det)
    print(f"[scorer] {det}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
    (ROOT / "results" / f"{det}.json").write_text(
        json.dumps({"detector": det, "seed": seedkey, "c": 0.03,
                    "scores": scores}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

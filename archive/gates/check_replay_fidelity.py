"""Chequeo de fidelidad del replay offline vs CSVs oficiales base (seed 42)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay


class SeqPass:
    def __init__(self, seq):
        self.seq = list(seq)
        self.i = 0

    def update(self, v, t=""):
        r = float(self.seq[self.i])
        self.i += 1
        return r

    def reset(self):
        self.i = 0


NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
TESTS = [
    "realAdExchange__exchange-2_cpc_results.csv.npz",
    "artificialWithAnomaly__art_daily_jumpsup.csv.npz",
]

for fn in TESTS:
    z = np.load(ROOT / "results" / "raw_esn" / fn, allow_pickle=True)
    raws = np.array(z["s42"], dtype=float)
    ts = [str(x) for x in z["timestamp"].tolist()]
    cal = PercentileCalibrator(SeqPass(raws), max_history=16000)
    dec = StreamingPeakDecay(cal, k_window=24, tau=2.0)
    rep = np.array([dec.update(float(r), t) for r, t in zip(raws, ts)])
    stem = fn[:-4]
    # stem: cat__file.csv ; NAB rel = cat/file.csv
    cat, fname = stem.split("__")
    csv = NAB / "results" / "sixeyes_esn_k24_tau2p0_w16000" / cat / (
        "sixeyes_esn_k24_tau2p0_w16000_" + fname
    )
    df = pd.read_csv(csv)
    off = df["anomaly_score"].to_numpy(dtype=float)
    print(fn, "max|diff|=", float(np.max(np.abs(rep - off))), "len", len(rep), len(off))

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, "src")
NAB = Path("../NAB").resolve()
sys.path.insert(0, str(NAB))
from nab.sweeper import Sweeper

profiles = json.load(open(NAB / "config" / "profiles.json", encoding="utf-8"))
labels = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
cm = profiles["standard"]["CostMatrix"]
sw = Sweeper(probationPercent=0.15, costMatrix=cm)

det_dir = NAB / "results" / "sixeyes_esn_k24_tau2p0_w16000"

# Cargar y pre-parsear todo a memoria de una sola vez
print("Cargando datos...", flush=True)
cached_data = []
for rel, wins in labels.items():
    rp = Path(rel)
    csv_file = det_dir / rp.parent / f"sixeyes_esn_k24_tau2p0_w16000_{rp.name}"
    df = pd.read_csv(csv_file)
    ts = list(pd.to_datetime(df["timestamp"]))
    sc = df["anomaly_score"].to_numpy(dtype=float)
    wl = [(pd.to_datetime(a), pd.to_datetime(b)) for a, b in wins]
    cached_data.append((rel, ts, sc, wl))

null_total = 0.0
for rel, wins in labels.items():
    rp = Path(rel)
    ndf = pd.read_csv(NAB / "results" / "null" / rp.parent / f"null_{rp.name}")
    ts = tuple(pd.to_datetime(ndf["timestamp"]))
    sc = tuple(float(x) for x in ndf["anomaly_score"])
    wl = tuple((pd.to_datetime(a), pd.to_datetime(b)) for a, b in wins)
    _, mrow = sw.scoreDataSet(ts, sc, wl, rel, 0.5)
    null_total += mrow.score if mrow is not None else 0.0

norm = lambda raw: (raw + 116.0) / 2.32


def eval_w(w_rest: int, trigger_thr: float):
    all_pts = []
    for rel, ts, sc, wl in cached_data:
        sc_mod = np.copy(sc)
        if w_rest > 0:
            silence_until = -1
            for i in range(len(sc_mod)):
                if i < silence_until:
                    sc_mod[i] = 0.0
                elif sc_mod[i] >= trigger_thr:
                    silence_until = i + w_rest
        pts = sw.calcSweepScore(ts, list(sc_mod), wl, rel)
        all_pts.extend(pts)
    thresholds = sw.calcScoreByThreshold(all_pts)
    best = max(thresholds, key=lambda x: x.score)
    return norm(best.score), best.threshold, best.tp, best.fp


print(f"Baseline (W_rest=0):", flush=True)
b_score, b_thr, b_tp, b_fp = eval_w(0, 0.99)
print(f"  Score: {b_score:.2f} (opt_thr={b_thr:.5f}, TP={b_tp}, FP={b_fp})", flush=True)

for w in [10, 24, 48, 80, 100, 120, 150, 200]:
    for trig in [0.99, 0.995, 0.996]:
        sc, thr, tp, fp = eval_w(w, trig)
        print(f"W_rest={w:3d}, Trigger={trig:.3f} -> Score: {sc:6.2f} (opt_thr={thr:.5f}, TP={tp:3d}, FP={fp:3d})", flush=True)

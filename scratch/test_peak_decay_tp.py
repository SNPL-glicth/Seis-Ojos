import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path("../benchmarks/nab").resolve()))

from nab.sweeper import Sweeper
from detectors.seasonal import DailySeasonalRobustZ
from detectors.calibrador_percentil import PercentileCalibrator

with open("../benchmarks/nab/labels/combined_windows.json") as f:
    combined_windows = json.load(f)
with open("../benchmarks/nab/config/profiles.json") as f:
    costMatrix = json.load(f)["standard"]["CostMatrix"]

sweeper = Sweeper(probationPercent=0.15, costMatrix=costMatrix)

# Pick 10 files that have >= 1 ground truth anomaly window
anomaly_files = [k for k, v in combined_windows.items() if len(v) > 0][:12]

def peak_decay_builder(K=6, tau=3.0):
    def fn(scores):
        out = []
        recent = []
        t_peak = 0
        for t, s in enumerate(scores):
            if len(recent) == 0 or s >= max(recent):
                out_s = s
                t_peak = t
            else:
                decay = np.exp(-(t - t_peak) / tau)
                out_s = s * decay
            recent.append(s)
            if len(recent) > K:
                recent.pop(0)
            out.append(out_s)
        return out
    return fn

print("Comparing Baseline vs Peak Decay on 12 Anomaly Files:")
print(f"{'File':<40} | {'Base Score':<10} | {'Base TP':<7} | {'Base FP':<7} | {'Decay Score':<11} | {'Decay TP':<8} | {'Decay FP':<8}")
print("-" * 105)

decay_fn = peak_decay_builder(K=6, tau=3.0)

tot_b_score, tot_b_tp, tot_b_fp = 0.0, 0, 0
tot_d_score, tot_d_tp, tot_d_fp = 0.0, 0, 0

for rel in anomaly_files:
    data_file = Path("../benchmarks/nab/data") / rel
    if not data_file.exists():
        continue
    df = pd.read_csv(data_file)
    windows = combined_windows.get(rel, [])
    
    det = PercentileCalibrator(DailySeasonalRobustZ())
    raw_scores = [det.update(v, t) for v, t in zip(df["value"], df["timestamp"])]
    decay_scores = decay_fn(raw_scores)
    
    _, b = sweeper.scoreDataSet(df["timestamp"], raw_scores, windows, rel, 0.9984)
    _, d = sweeper.scoreDataSet(df["timestamp"], decay_scores, windows, rel, 0.9984)
    
    tot_b_score += b.score
    tot_b_tp += b.tp
    tot_b_fp += b.fp
    
    tot_d_score += d.score
    tot_d_tp += d.tp
    tot_d_fp += d.fp
    
    fname = Path(rel).name
    print(f"{fname:<40} | {b.score:<10.2f} | {b.tp:<7} | {b.fp:<7} | {d.score:<11.2f} | {d.tp:<8} | {d.fp:<8}")

print("-" * 105)
print(f"{'TOTAL':<40} | {tot_b_score:<10.2f} | {tot_b_tp:<7} | {tot_b_fp:<7} | {tot_d_score:<11.2f} | {tot_d_tp:<8} | {tot_d_fp:<8}")

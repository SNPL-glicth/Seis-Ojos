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

rel = "realTweets/Twitter_volume_IBM.csv"
data_path = Path("../benchmarks/nab/data") / rel
df = pd.read_csv(data_path)

with open("../benchmarks/nab/labels/combined_windows.json") as f:
    windows = json.load(f)[rel]
with open("../benchmarks/nab/config/profiles.json") as f:
    costMatrix = json.load(f)["standard"]["CostMatrix"]

sweeper = Sweeper(probationPercent=0.15, costMatrix=costMatrix)

# Base scores
det = PercentileCalibrator(DailySeasonalRobustZ())
base_scores = np.array([det.update(v, t) for v, t in zip(df["value"], df["timestamp"])])

print(f"=== NMS / Cooldown Audit on {rel} (Threshold=0.99775) ===")
_, b0 = sweeper.scoreDataSet(df["timestamp"], list(base_scores), windows, rel, 0.99775)
print(f"Base (no suppression) -> Score: {b0.score:.4f} | TP: {b0.tp} | FP: {b0.fp}")

for k in [1, 2, 3, 5, 8, 12]:
    nms_scores = base_scores.copy()
    cooldown = 0
    for i in range(len(nms_scores)):
        if cooldown > 0:
            nms_scores[i] = 0.0
            cooldown -= 1
        elif nms_scores[i] > 0.99775:
            cooldown = k
            
    _, b_nms = sweeper.scoreDataSet(df["timestamp"], list(nms_scores), windows, rel, 0.99775)
    print(f"Cooldown K={k:<2}          -> Score: {b_nms.score:.4f} | TP: {b_nms.tp} | FP: {b_nms.fp}")

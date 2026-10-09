import sys
from pathlib import Path
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

from detectors.seasonal import DailySeasonalRobustZ
from detectors.calibrador_percentil import PercentileCalibrator

df = pd.read_csv("../benchmarks/nab/data/realTweets/Twitter_volume_IBM.csv")
det_orig = PercentileCalibrator(DailySeasonalRobustZ())
scores_orig = pd.Series([det_orig.update(v, t) for v, t in zip(df["value"], df["timestamp"])])

det_raw = DailySeasonalRobustZ()
raws = pd.Series([det_raw.update(v, t) for v, t in zip(df["value"], df["timestamp"])])

print("IBM raw scores count:", len(raws))
print("Max raw:", raws.max())
print("Raw > 0.5 (Z > 3.5):", (raws > 0.5).sum())
print("Raw > 0.6 (Z > 5.25):", (raws > 0.6).sum())
print("Original scores > 0.9984:", (scores_orig > 0.9984).sum())

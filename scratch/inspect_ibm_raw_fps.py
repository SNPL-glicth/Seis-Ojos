import sys
from pathlib import Path
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

from detectors.seasonal import DailySeasonalRobustZ
from detectors.calibrador_percentil import PercentileCalibrator

df = pd.read_csv("../benchmarks/nab/data/realTweets/Twitter_volume_IBM.csv")
det = DailySeasonalRobustZ()
raws = [det.update(v, t) for v, t in zip(df["value"], df["timestamp"])]
df["raw"] = raws

det_p = PercentileCalibrator(DailySeasonalRobustZ())
df["pct"] = [det_p.update(v, t) for v, t in zip(df["value"], df["timestamp"])]

fps = df[df["pct"] > 0.9984]
print("Count of points with pct > 0.9984:", len(fps))
print("Raw stats:")
print(fps["raw"].describe())
print("Value stats:")
print(fps["value"].describe())
print("\nTop 15 rows with pct > 0.9984:")
print(fps[["timestamp", "value", "raw", "pct"]].sort_values("value", ascending=False).head(15))

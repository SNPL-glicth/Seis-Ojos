import json
import sys
from pathlib import Path
import bisect
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path("../benchmarks/nab").resolve()))

from nab.sweeper import Sweeper
from detectors.seasonal import DailySeasonalRobustZ
from detectors.calibrador_percentil import PercentileCalibrator


class OutputGatedCalibrator(PercentileCalibrator):
    def __init__(self, detector, min_raw: float = 0.0):
        super().__init__(detector)
        self.min_raw = min_raw

    def update(self, value: float, timestamp: str = "") -> float:
        # Obtain raw score from detector
        raw = self._detector.update(value, timestamp)
        if raw is None or np.isnan(raw) or np.isinf(raw) or raw == 0.0:
            return 0.0

        # Always maintain global ECDF
        if len(self._historial) < self.MIN_HISTORIAL:
            bisect.insort(self._historial, raw)
            return 0.0

        menores = bisect.bisect_left(self._historial, raw)
        iguales = bisect.bisect_right(self._historial, raw) - menores
        n = len(self._historial)
        score = (menores + 0.5 * iguales + 0.5) / (n + 1)
        bisect.insort(self._historial, raw)

        # Output Gate: squelch if raw does not reach statistical significance
        if raw < self.min_raw:
            return 0.0

        return score


rel = "realTweets/Twitter_volume_IBM.csv"
data_path = Path("../benchmarks/nab/data") / rel
df = pd.read_csv(data_path)

with open("../benchmarks/nab/labels/combined_windows.json") as f:
    windows = json.load(f)[rel]
with open("../benchmarks/nab/config/profiles.json") as f:
    costMatrix = json.load(f)["standard"]["CostMatrix"]

sweeper = Sweeper(probationPercent=0.15, costMatrix=costMatrix)

print(f"=== Auditoría Output Gate en {rel} (Threshold=0.9984) ===")
print(f"{'min_raw':<10} | {'Z aprox':<10} | {'NAB Score':<12} | {'TP':<6} | {'FP':<6} | {'FN':<6}")
print("-" * 65)

for min_raw in [0.0, 0.30, 0.40, 0.4615, 0.50, 0.55, 0.60]:
    # z approx: raw = z / (z + 3.5) => z = 3.5 * raw / (1 - raw)
    z_approx = (3.5 * min_raw / (1.0 - min_raw)) if min_raw < 1.0 else 999.0
    det = OutputGatedCalibrator(DailySeasonalRobustZ(), min_raw=min_raw)
    scores = [det.update(v, t) for v, t in zip(df["value"], df["timestamp"])]
    _, best = sweeper.scoreDataSet(df["timestamp"], scores, windows, rel, 0.9984)
    print(f"{min_raw:<10.4f} | {z_approx:<10.2f} | {best.score:<12.4f} | {best.tp:<6} | {best.fp:<6} | {best.fn:<6}")

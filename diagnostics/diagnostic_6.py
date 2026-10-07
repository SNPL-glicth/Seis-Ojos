import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from detectors.seasonal import DailySeasonalRobustZ

df = pd.read_csv("../benchmarks/nab/data/realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv", parse_dates=["timestamp"])
timestamps = df["timestamp"].tolist()
values = df["value"].tolist()

k = len(values) // 2

det_normal = DailySeasonalRobustZ()
scores_normal = []
for val, ts in zip(values, timestamps):
    scores_normal.append(det_normal.update(val, str(ts)))

det_alt = DailySeasonalRobustZ()
scores_alt = []
for i, (val, ts) in enumerate(zip(values, timestamps)):
    if i >= k:
        val = 9999.0
    scores_alt.append(det_alt.update(val, str(ts)))

# Check that there are scores > 0.0 before k
non_zero = sum(1 for s in scores_normal[:k] if s > 0.0)

diff = False
for i in range(k):
    if scores_normal[i] != scores_alt[i]:
        diff = True

print(f"Total points: {len(values)}, k = {k}")
print(f"Scores > 0.0 before k: {non_zero}")
print(f"¿Los scores hasta k son idénticos a pesar de modificar el futuro?: {not diff}")

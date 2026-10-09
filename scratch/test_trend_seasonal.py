import json
import sys
from pathlib import Path
from collections import deque
import statistics
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

from detectors.seasonal import DailySeasonalRobustZ
from detectors.calibrador_percentil import PercentileCalibrator


class DailySeasonalWithTrend(DailySeasonalRobustZ):
    def __init__(self, bucket_size_minutes: int = 15, window_size: int = 20, min_observations: int = 5, trend_window: int = 288):
        super().__init__(bucket_size_minutes=bucket_size_minutes, window_size=window_size, min_observations=min_observations)
        self._trend_window = trend_window
        self._trend_buffer = deque(maxlen=trend_window)

    def update(self, value: float, timestamp: str) -> float:
        if len(self._trend_buffer) >= 10:
            trend = statistics.median(self._trend_buffer)
            detrended_val = value - trend
        else:
            detrended_val = value
        
        self._trend_buffer.append(value)
        return super().update(detrended_val, timestamp)


windows_path = Path("../benchmarks/nab/labels/combined_windows.json")
with open(windows_path, "r", encoding="utf-8") as f:
    all_windows = json.load(f)

test_files = [
    "realKnownCause/machine_temperature_system_failure.csv",
    "realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv",
    "realKnownCause/cpu_utilization_asg_misconfiguration.csv",
    "realAWSCloudwatch/ec2_network_in_5abac7.csv",
    "realAWSCloudwatch/grok_asg_anomaly.csv"
]

print(f"{'Archivo':<48} | {'Ventana':<7} | {'Max Base':<9} | {'Max Trend':<9} | {'FP Base (>0.998)':<16} | {'FP Trend (>0.998)':<16}")
print("-" * 115)

for rel in test_files:
    data_path = Path("../benchmarks/nab/data") / rel
    if not data_path.is_file():
        continue
    df = pd.read_csv(data_path)
    file_windows = all_windows.get(rel.replace("\\", "/"), [])

    # Base seasonal_pct
    det_base = PercentileCalibrator(DailySeasonalRobustZ())
    scores_base = [det_base.update(v, t) for v, t in zip(df["value"], df["timestamp"])]

    # Trend seasonal_pct
    det_trend = PercentileCalibrator(DailySeasonalWithTrend())
    scores_trend = [det_trend.update(v, t) for v, t in zip(df["value"], df["timestamp"])]

    df["base"] = scores_base
    df["trend"] = scores_trend

    # Out of all windows:
    in_any_window = np.zeros(len(df), dtype=bool)
    for w in file_windows:
        mask = (df["timestamp"] >= w[0]) & (df["timestamp"] <= w[1])
        in_any_window |= mask

    fp_base = ((df["base"] > 0.998) & (~in_any_window)).sum()
    fp_trend = ((df["trend"] > 0.998) & (~in_any_window)).sum()

    for idx, w in enumerate(file_windows):
        w_df = df[(df["timestamp"] >= w[0]) & (df["timestamp"] <= w[1])]
        max_base = w_df["base"].max()
        max_trend = w_df["trend"].max()
        print(f"{rel:<48} | W{idx:<6} | {max_base:<9.5f} | {max_trend:<9.5f} | {fp_base:<16} | {fp_trend:<16}")

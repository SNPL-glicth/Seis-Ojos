import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))
from detectors.subespacio import SubspaceResidualDetector
from detectors.calibrador_percentil import PercentileCalibrator
from detectors.seasonal import DailySeasonalRobustZ

windows_path = Path("../benchmarks/nab/labels/combined_windows.json")
with open(windows_path, "r", encoding="utf-8") as f:
    windows = json.load(f)

for rel in ["realKnownCause/rogue_agent_key_hold.csv", "realAWSCloudwatch/grok_asg_anomaly.csv", "realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv"]:
    data_path = Path("../benchmarks/nab/data") / rel
    if not data_path.is_file():
        continue
    df = pd.read_csv(data_path)
    file_windows = windows.get(rel.replace("\\", "/"), [])

    # Subspace + Calibrator
    sub_det = PercentileCalibrator(SubspaceResidualDetector())
    sub_scores = [sub_det.update(v, t) for v, t in zip(df["value"], df["timestamp"])]

    # Seasonal + Calibrator
    sea_det = PercentileCalibrator(DailySeasonalRobustZ())
    sea_scores = [sea_det.update(v, t) for v, t in zip(df["value"], df["timestamp"])]

    # Raw Subspace
    sub_raw = SubspaceResidualDetector()
    raw_scores = [sub_raw.update(v, t) for v, t in zip(df["value"], df["timestamp"])]

    df["sub_score"] = sub_scores
    df["sea_score"] = sea_scores
    df["raw_sub"] = raw_scores

    # Check window scores
    print(f"\n==================== {rel} ====================")
    print(f"Total points: {len(df)}, Windows count: {len(file_windows)}")
    for w in file_windows:
        w_df = df[(df["timestamp"] >= w[0]) & (df["timestamp"] <= w[1])]
        out_df = df[(df["timestamp"] < w[0]) | (df["timestamp"] > w[1])]
        print(f"Window: {w[0]} to {w[1]} (length {len(w_df)})")
        print(f"  In window  -> Sub max: {w_df['sub_score'].max():.5f}, Sea max: {w_df['sea_score'].max():.5f}")
        print(f"  In window  -> Raw sub: max={w_df['raw_sub'].max():.5f}, mean={w_df['raw_sub'].mean():.5f}")
        print(f"  Out window -> Sub > 0.998 count: {(out_df['sub_score'] > 0.998).sum()}, Sea > 0.998 count: {(out_df['sea_score'] > 0.998).sum()}")

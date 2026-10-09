import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path("../benchmarks/nab").resolve()))

from detectors.seasonal import DailySeasonalRobustZ
from detectors.calibrador_percentil import PercentileCalibrator
from dateutil import parser as date_parser

with open("../benchmarks/nab/labels/combined_windows.json") as f:
    raw_windows = json.load(f)

# Total anomaly windows in the entire NAB corpus
total_windows = sum(len(v) for v in raw_windows.values())
print(f"Total anomaly windows in NAB corpus: {total_windows}")

class StreamingPeakDecay:
    def __init__(self, detector, K: int = 12, tau: float = 2.0):
        self._detector = detector
        self.K = K
        self.tau = tau
        self._history = []
        self._t = 0
        self._t_peak = 0

    def update(self, value: float, timestamp: str = "") -> float:
        raw_s = self._detector.update(value, timestamp)
        if raw_s is None or np.isnan(raw_s) or np.isinf(raw_s):
            return 0.0

        t = self._t
        self._t += 1

        if not self._history or raw_s >= max(self._history):
            self._t_peak = t
            s_out = raw_s
        else:
            decay = float(np.exp(-(t - self._t_peak) / self.tau))
            s_out = raw_s * decay

        self._history.append(raw_s)
        if len(self._history) > self.K:
            self._history.pop(0)

        return float(s_out)


def audit_windows_hit(name, make_detector, threshold):
    detected_windows = 0
    total_active_windows = 0
    
    for rel_path, windows in raw_windows.items():
        data_path = Path("../benchmarks/nab/data") / rel_path
        if not data_path.exists() or len(windows) == 0:
            continue
            
        df = pd.read_csv(data_path)
        det = make_detector()
        scores = [det.update(v, t) for v, t in zip(df["value"], df["timestamp"])]
        df["score"] = scores
        df["dt"] = [date_parser.parse(t) for t in df["timestamp"]]
        
        # Probationary cutoff: first 15% (max 750 points)
        probation_len = min(int(0.15 * len(df)), 750)
        df_valid = df.iloc[probation_len:]
        
        for w in windows:
            t1 = date_parser.parse(w[0])
            t2 = date_parser.parse(w[1])
            total_active_windows += 1
            
            # Points inside window
            pts_in_window = df_valid[(df_valid["dt"] >= t1) & (df_valid["dt"] <= t2)]
            if len(pts_in_window) > 0 and (pts_in_window["score"] > threshold).any():
                detected_windows += 1

    print(f"{name:<30} (th={threshold:.5f}): {detected_windows}/{total_active_windows} ventanas detectadas ({100.0*detected_windows/total_active_windows:.2f}%)")

print("\n--- Auditoría de Detección de Ventanas de Anomalía (Ground Truth Events) ---")
audit_windows_hit(
    "Base (seasonal_pct)",
    lambda: PercentileCalibrator(DailySeasonalRobustZ()),
    0.99775
)
audit_windows_hit(
    "PeakDecay K=12, tau=2.0",
    lambda: StreamingPeakDecay(PercentileCalibrator(DailySeasonalRobustZ()), K=12, tau=2.0),
    0.99665
)
audit_windows_hit(
    "PeakDecay K=16, tau=2.0",
    lambda: StreamingPeakDecay(PercentileCalibrator(DailySeasonalRobustZ()), K=16, tau=2.0),
    0.99665
)

"""Six Eyes: Detección de anomalías en streaming para series temporales y telemetría."""

from sixeyes.detectors.base import Detector
from sixeyes.detectors.aleatorio import RandomDetector
from sixeyes.detectors.zscore_robusto import RollingRobustZ
from sixeyes.detectors.seasonal import DailySeasonalRobustZ
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.subespacio import SubspaceResidualDetector
from sixeyes.detectors.peak_decay import StreamingPeakDecay, DEFAULT_K_WINDOW, DEFAULT_TAU
from sixeyes.evaluation.nab_runner import NABRunner
from sixeyes.evaluation.tsb_runner import run_single_series, get_tuning_files

__version__ = "0.2.0"

__all__ = [
    "Detector",
    "RandomDetector",
    "RollingRobustZ",
    "DailySeasonalRobustZ",
    "PercentileCalibrator",
    "SubspaceResidualDetector",
    "StreamingPeakDecay",
    "DEFAULT_K_WINDOW",
    "DEFAULT_TAU",
    "NABRunner",
    "run_single_series",
    "get_tuning_files",
]

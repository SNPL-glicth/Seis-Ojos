"""Módulo de detectores de anomalías."""

from sixeyes.detectors.aleatorio import RandomDetector
from sixeyes.detectors.base import Detector
from sixeyes.detectors.zscore_robusto import RollingRobustZ
from sixeyes.detectors.seasonal import DailySeasonalRobustZ
from sixeyes.detectors.biseasonal import BiSeasonalRobustZ
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.subespacio import SubspaceResidualDetector
from sixeyes.detectors.combinador import MaxOfDetectors, Combinado
from sixeyes.detectors.peak_decay import StreamingPeakDecay

__all__ = [
    "Detector",
    "RandomDetector",
    "RollingRobustZ",
    "DailySeasonalRobustZ",
    "BiSeasonalRobustZ",
    "PercentileCalibrator",
    "SubspaceResidualDetector",
    "MaxOfDetectors",
    "Combinado",
    "StreamingPeakDecay",
]

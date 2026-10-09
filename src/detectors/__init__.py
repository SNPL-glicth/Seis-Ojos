"""Módulo de detectores de anomalías."""

try:
    from detectors.aleatorio import RandomDetector
    from detectors.base import Detector
    from detectors.zscore_robusto import RollingRobustZ
    from detectors.seasonal import DailySeasonalRobustZ
    from detectors.calibrador_percentil import PercentileCalibrator
    from detectors.subespacio import SubspaceResidualDetector
    from detectors.combinador import MaxOfDetectors, Combinado
except ImportError:
    from src.detectors.aleatorio import RandomDetector
    from src.detectors.base import Detector
    from src.detectors.zscore_robusto import RollingRobustZ
    from src.detectors.seasonal import DailySeasonalRobustZ
    from src.detectors.calibrador_percentil import PercentileCalibrator
    from src.detectors.subespacio import SubspaceResidualDetector
    from src.detectors.combinador import MaxOfDetectors, Combinado

__all__ = [
    "Detector",
    "RandomDetector",
    "RollingRobustZ",
    "DailySeasonalRobustZ",
    "PercentileCalibrator",
    "SubspaceResidualDetector",
    "MaxOfDetectors",
    "Combinado",
]

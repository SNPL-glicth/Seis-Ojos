"""Módulo de detectores oficiales de anomalías.

Pipeline oficial congelado:
    SeasonalDualScaleRobustZ + PercentileCalibrator + StreamingPeakDecay(K=12, tau=2.0)

Solo se exportan componentes oficiales + baselines causales mínimos.
BiSeasonal, Combinado/MaxOfDetectors, DelayEmbedding y RampGradient viven en
``sixeyes.experimental`` (no oficiales) y no forman parte del namespace oficial.
"""

from sixeyes.detectors.aleatorio import RandomDetector
from sixeyes.detectors.base import Detector
from sixeyes.detectors.zscore_robusto import RollingRobustZ
from sixeyes.detectors.seasonal import DailySeasonalRobustZ
from sixeyes.detectors.dual_scale import DualScaleRobustZ, SeasonalDualScaleRobustZ
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.subespacio import SubspaceResidualDetector
from sixeyes.detectors.peak_decay import StreamingPeakDecay, DEFAULT_K_WINDOW, DEFAULT_TAU

__all__ = [
    "Detector",
    "RandomDetector",
    "RollingRobustZ",
    "DailySeasonalRobustZ",
    "DualScaleRobustZ",
    "SeasonalDualScaleRobustZ",
    "PercentileCalibrator",
    "SubspaceResidualDetector",
    "StreamingPeakDecay",
    "DEFAULT_K_WINDOW",
    "DEFAULT_TAU",
]

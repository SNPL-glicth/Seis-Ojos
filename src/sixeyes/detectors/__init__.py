"""Módulo de detectores oficiales de anomalías.

Pipeline oficial vigente:
    RefractorySuppressor + SpatialSpikeBypass + StreamingPeakDecay + PercentileCalibrator + EchoStatePredictor

Solo se exportan componentes oficiales + baselines causales mínimos.
"""

from sixeyes.detectors.aleatorio import RandomDetector
from sixeyes.detectors.base import Detector
from sixeyes.detectors.zscore_robusto import RollingRobustZ
from sixeyes.detectors.seasonal import DailySeasonalRobustZ
from sixeyes.detectors.dual_scale import DualScaleRobustZ, SeasonalDualScaleRobustZ
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.subespacio import SubspaceResidualDetector
from sixeyes.detectors.peak_decay import StreamingPeakDecay, DEFAULT_K_WINDOW, DEFAULT_TAU
from sixeyes.detectors.refractory_suppressor import (
    RefractorySuppressor,
    DEFAULT_W_REST,
    DEFAULT_TRIGGER_THRESHOLD,
)
from sixeyes.detectors.spatial_spike_bypass import (
    SpatialSpikeBypass,
    DEFAULT_SPATIAL_TOLERANCE,
    DEFAULT_WARMUP,
)

__all__ = [
    "Detector",
    "RandomDetector",
    "RollingRobustZ",
    "DailySeasonalRobustZ",
    "DualScaleRobustZ",
    "SeasonalDualScaleRobustZ",
    "EchoStatePredictor",
    "PercentileCalibrator",
    "SubspaceResidualDetector",
    "StreamingPeakDecay",
    "RefractorySuppressor",
    "SpatialSpikeBypass",
    "DEFAULT_K_WINDOW",
    "DEFAULT_TAU",
    "DEFAULT_W_REST",
    "DEFAULT_TRIGGER_THRESHOLD",
    "DEFAULT_SPATIAL_TOLERANCE",
    "DEFAULT_WARMUP",
]

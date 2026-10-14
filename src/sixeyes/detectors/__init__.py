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
from sixeyes.detectors.gram_tracker import GramSubspaceTracker
from sixeyes.detectors.active_drift import ActiveCentroidDriftTracker
from sixeyes.detectors.causal_multiscale_lds import (
    CausalFeatureExtractor,
    OnlineEKF,
    OnlineUKF,
    OnlineLDS,
    CausalMultiscaleLDS,
)
from sixeyes.detectors.causal_spectral import CausalSTFTDetector
from sixeyes.detectors.master_spectral_peak import MasterSpectralPeakDecay
from sixeyes.detectors.peak_decay_ramp_boost import PeakDecayRampBoost

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
    "GramSubspaceTracker",
    "ActiveCentroidDriftTracker",
    "CausalFeatureExtractor",
    "OnlineEKF",
    "OnlineUKF",
    "OnlineLDS",
    "CausalMultiscaleLDS",
    "CausalSTFTDetector",
    "MasterSpectralPeakDecay",
    "PeakDecayRampBoost",
]

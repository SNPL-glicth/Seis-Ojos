"""Módulo de detectores de anomalías."""

from sixeyes.detectors.aleatorio import RandomDetector
from sixeyes.detectors.base import Detector
from sixeyes.detectors.zscore_robusto import RollingRobustZ

__all__ = ["Detector", "RandomDetector", "RollingRobustZ"]

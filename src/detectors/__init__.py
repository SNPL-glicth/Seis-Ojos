"""Módulo de detectores de anomalías."""

from detectors.aleatorio import RandomDetector
from detectors.base import Detector
from detectors.zscore_robusto import RollingRobustZ

__all__ = ["Detector", "RandomDetector", "RollingRobustZ"]

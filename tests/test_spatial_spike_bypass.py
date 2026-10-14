"""Tests para SpatialSpikeBypass."""

import pytest
import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass


class ConstantZeroDetector(Detector):
    def update(self, value: float, timestamp: str = "") -> float:
        return 0.1

    def reset(self) -> None:
        pass


def test_spatial_bypass_normal_values():
    """Valores dentro del rango no activan el bypass."""
    dummy = ConstantZeroDetector()
    bypass = SpatialSpikeBypass(dummy, tolerance=0.1, warmup=5)

    # 10 valores entre 10 y 20
    for v in [10.0, 15.0, 12.0, 18.0, 20.0, 11.0, 19.0]:
        s = bypass.update(v)
        assert s == 0.1


def test_spatial_bypass_trigger_outlier():
    """Un valor que rompe los extremos históricos por más del margen activa 1.0."""
    dummy = ConstantZeroDetector()
    bypass = SpatialSpikeBypass(dummy, tolerance=0.1, warmup=5)

    # Warmup con rango [10, 20], span = 10, margen 10% = 1.0 (cotas: 9.0 a 21.0)
    for v in [10.0, 15.0, 12.0, 18.0, 20.0]:
        s = bypass.update(v)
        assert s == 0.1

    # Valor 22.0 > 21.0 -> debe emitir 1.0
    assert bypass.update(22.0) == 1.0

    # Valor 8.0 < 9.0 -> debe emitir 1.0
    assert bypass.update(8.0) == 1.0


def test_spatial_bypass_reset():
    dummy = ConstantZeroDetector()
    bypass = SpatialSpikeBypass(dummy, tolerance=0.1, warmup=3)

    for v in [10.0, 15.0, 20.0]:
        bypass.update(v)

    assert bypass.update(30.0) == 1.0
    bypass.reset()

    # Tras reset, vuelve a estar en warmup
    assert bypass.update(30.0) == 0.1

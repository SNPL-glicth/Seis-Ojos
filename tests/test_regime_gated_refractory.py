"""Tests para RegimeGatedRefractory (1 intento pre-registrado)."""

import pytest

from sixeyes.detectors.base import Detector
from sixeyes.experimental.regime_gated_refractory import RegimeGatedRefractory


class Dummy(Detector):
    def __init__(self, scores):
        self.scores = list(scores)
        self.idx = 0

    def update(self, value=0.0, timestamp=""):
        if self.idx < len(self.scores):
            s = self.scores[self.idx]
            self.idx += 1
            return s
        return 0.0

    def reset(self):
        self.idx = 0


def test_calm_chooses_short_silence():
    # Sin excedencias previas: vol~0 -> W=40, silencio exacto de 40.
    ref = RegimeGatedRefractory(Dummy([0.995] + [0.5] * 50))
    assert ref.update(0.0) == pytest.approx(0.995)
    assert ref.last_w == 40
    for _ in range(40):
        assert ref.update(0.0) == 0.0
    assert ref.update(0.0) == pytest.approx(0.5)


def test_turbulent_chooses_long_silence():
    # Pre-llenar con cola alta para inflar vol por encima de 0.35.
    ref = RegimeGatedRefractory(Dummy([0.95] * 60 + [0.995] + [0.5] * 200))
    for _ in range(60):
        ref.update(0.0)
    assert ref._vol > 0.35
    assert ref.update(0.0) == pytest.approx(0.995)
    assert ref.last_w == 160
    for _ in range(160):
        assert ref.update(0.0) == 0.0


def test_never_wakes_early_rigidity():
    # Durante el silencio ni un pico posterior lo interrumpe.
    ref = RegimeGatedRefractory(Dummy([0.995, 0.999, 0.999, 0.5]))
    assert ref.update(0.0) == pytest.approx(0.995)
    assert ref.update(0.0) == 0.0  # 0.999 suprimido, no re-dispara
    assert ref.update(0.0) == 0.0
    # vol se sigue actualizando con el raw subyacente (no colapsa a 0):
    assert ref._vol > 0.0


def test_reset_invariance():
    seq = [0.95] * 10 + [0.995, 0.5]
    r = RegimeGatedRefractory(Dummy(seq))
    run1 = [r.update(0.0) for _ in seq]
    r.reset()
    assert r._vol == 0.0 and r.last_w == 0
    run2 = [r.update(0.0) for _ in seq]
    assert run1 == run2


def test_invalid_args():
    with pytest.raises(ValueError):
        RegimeGatedRefractory(Dummy([0.5]), w_calm=-1)
    with pytest.raises(ValueError):
        RegimeGatedRefractory(Dummy([0.5]), trigger_threshold=1.5)
    with pytest.raises(ValueError):
        RegimeGatedRefractory(Dummy([0.5]), alpha=0.0)
    with pytest.raises(ValueError):
        RegimeGatedRefractory(Dummy([0.5]), lo_vol=0.5, hi_vol=0.2)

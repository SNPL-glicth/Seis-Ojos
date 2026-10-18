"""Tests para TimeScaledRefractory."""

import pytest

from sixeyes.detectors.base import Detector
from sixeyes.experimental.time_scaled_refractory import TimeScaledRefractory


class Dummy(Detector):
    def __init__(self, scores):
        self.scores = list(scores)
        self.idx = 0

    def update(self, value=0.0, timestamp=""):
        s = self.scores[self.idx] if self.idx < len(self.scores) else 0.0
        self.idx += 1
        return s

    def reset(self):
        self.idx = 0


def test_base_silence_early_series():
    m = TimeScaledRefractory(Dummy([0.995] + [0.5] * 100))
    assert m.update(0.0) == pytest.approx(0.995)
    assert m.last_w == 80  # t=0 -> max(80, 0)
    for _ in range(80):
        assert m.update(0.0) == 0.0
    assert m.update(0.0) == pytest.approx(0.5)


def test_scaled_silence_late_series():
    m = TimeScaledRefractory(Dummy([0.1] * 5000 + [0.995] + [0.5] * 300))
    for _ in range(5000):
        m.update(0.0)
    assert m.update(0.0) == pytest.approx(0.995)
    assert m.last_w == 150  # max(80, 0.03*5000)
    for _ in range(150):
        assert m.update(0.0) == 0.0


def test_never_wakes_early():
    m = TimeScaledRefractory(Dummy([0.995, 0.999, 0.999, 0.5]))
    assert m.update(0.0) == pytest.approx(0.995)
    assert m.update(0.0) == 0.0
    assert m.update(0.0) == 0.0


def test_reset_invariance():
    seq = [0.995, 0.5, 0.5]
    m = TimeScaledRefractory(Dummy(seq))
    r1 = [m.update(0.0) for _ in seq]
    m.reset()
    assert (m._t, m._silence_until, m.last_w) == (0, -1, 0)
    assert [m.update(0.0) for _ in seq] == r1


def test_invalid_args():
    with pytest.raises(ValueError):
        TimeScaledRefractory(Dummy([0.5]), w_base=-1)
    with pytest.raises(ValueError):
        TimeScaledRefractory(Dummy([0.5]), c=-0.01)
    with pytest.raises(ValueError):
        TimeScaledRefractory(Dummy([0.5]), trigger_threshold=2.0)

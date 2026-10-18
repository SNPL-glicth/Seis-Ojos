"""Tests para CUSUMDrift."""

import pytest

from sixeyes.detectors.base import Detector
from sixeyes.experimental.cusum_drift import CUSUMDrift


class Dummy(Detector):
    def __init__(self, seq):
        self.seq = list(seq)
        self.idx = 0

    def update(self, value=0.0, timestamp=""):
        s = self.seq[self.idx] if self.idx < len(self.seq) else 0.0
        self.idx += 1
        return s

    def reset(self):
        self.idx = 0


def test_sustained_drift_ramps():
    c = CUSUMDrift(Dummy([1.0] * 20), drift=0.5, threshold=5.0)
    outs = [c.update(0.0) for _ in range(20)]
    assert outs[-1] > outs[0]  # acumula: S+ crece 0.5/paso
    assert outs[-1] == pytest.approx(20 * 0.5 / 5.0)


def test_symmetric_noise_stays_zero():
    c = CUSUMDrift(Dummy([1.0, -1.0] * 20), drift=0.5, threshold=5.0)
    outs = [c.update(0.0) for _ in range(40)]
    assert max(outs) <= 0.5 / 5.0 + 1e-9  # se auto-cancela


def test_direction_separation():
    up = CUSUMDrift(Dummy([1.0] * 10), drift=0.5, threshold=5.0)
    for _ in range(10):
        up.update(0.0)
    assert up._sp > 0 and up._sn == 0.0
    dn = CUSUMDrift(Dummy([-1.0] * 10), drift=0.5, threshold=5.0)
    for _ in range(10):
        dn.update(0.0)
    assert dn._sn > 0 and dn._sp == 0.0


def test_calm_zero_drift_no_accumulation():
    c = CUSUMDrift(Dummy([0.1] * 50), drift=0.5, threshold=5.0)
    assert max(c.update(0.0) for _ in range(50)) == 0.0


def test_determinism_and_prefix_invariance():
    seq = [0.8, -0.2, 0.9, 0.7, -0.9, 0.6]
    a = CUSUMDrift(Dummy(seq))
    b = CUSUMDrift(Dummy(seq[:3]))
    oa = [a.update(0.0) for _ in seq]
    ob = [b.update(0.0) for _ in seq[:3]]
    assert oa[:3] == ob


def test_state_o1_and_reset():
    c = CUSUMDrift(Dummy([0.7] * 5000))
    for _ in range(5000):
        c.update(0.0)
    assert set(c.__dict__) <= {"_det", "_k", "_h", "_sp", "_sn"}
    c.reset()
    assert (c._sp, c._sn) == (0.0, 0.0)


def test_invalid_args():
    with pytest.raises(ValueError):
        CUSUMDrift(Dummy([0.0]), drift=-0.1)
    with pytest.raises(ValueError):
        CUSUMDrift(Dummy([0.0]), threshold=0.0)

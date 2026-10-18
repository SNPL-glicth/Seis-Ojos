"""Tests para CausalAdaptiveThreshold."""

import pytest

from sixeyes.detectors.base import Detector
from sixeyes.experimental.causal_adaptive_threshold import CausalAdaptiveThreshold


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


def test_warmup_emits_normalized_exceedance_not_threshold():
    c = CausalAdaptiveThreshold(Dummy([0.99] * 100), warmup_steps=100)
    outs = [c.update(0.0) for _ in range(100)]
    # tr~=0.99+(0.99-0.5)*0.01~=0.9949; out=(0.99-tr)<0 -> rama baja: raw*0.5
    assert all(0.0 <= o <= 1.0 for o in outs)
    assert outs[-1] == pytest.approx(0.99 * 0.5)


def test_freeze_on_high_no_habituation():
    c = CausalAdaptiveThreshold(Dummy([0.5] * 100 + [0.999] * 5), warmup_steps=100)
    for _ in range(100):
        c.update(0.0)
    tr_before = c._tr
    c.update(0.0)  # 0.999 >= tr -> congelar
    assert c._tr == pytest.approx(tr_before)


def test_learns_only_in_calm():
    c = CausalAdaptiveThreshold(Dummy([0.5] * 200), warmup_steps=100)
    for _ in range(100):
        c.update(0.0)
    tr0 = c._tr
    for _ in range(50):
        c.update(0.0)  # calma 0.5 < tr -> deriva lenta hacia abajo
    assert c._tr < tr0


def test_reset_invariance():
    seq = [0.5] * 100 + [0.999, 0.3]
    c = CausalAdaptiveThreshold(Dummy(seq), warmup_steps=100)
    r1 = [c.update(0.0) for _ in seq]
    c.reset()
    assert c._tr == pytest.approx(0.990)
    r2 = [c.update(0.0) for _ in seq]
    assert r1 == r2


def test_invalid_args():
    with pytest.raises(ValueError):
        CausalAdaptiveThreshold(Dummy([0.5]), base_threshold=2.0)
    with pytest.raises(ValueError):
        CausalAdaptiveThreshold(Dummy([0.5]), alpha=0.0)
    with pytest.raises(ValueError):
        CausalAdaptiveThreshold(Dummy([0.5]), warmup_steps=5)

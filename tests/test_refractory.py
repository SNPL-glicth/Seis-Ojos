"""Tests para RefractorySuppressor."""

import pytest
import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.detectors.refractory_suppressor import RefractorySuppressor


class DummyDetector(Detector):
    def __init__(self, scores: list[float] | None = None):
        self.scores = scores or []
        self.idx = 0

    def update(self, value: float, timestamp: str = "") -> float:
        if self.idx < len(self.scores):
            s = self.scores[self.idx]
            self.idx += 1
            return s
        return 0.0

    def reset(self) -> None:
        self.idx = 0


def test_refractory_basic_suppression():
    """Verifica que un disparo >= trigger silencia exactamente W_rest pasos."""
    # Score en t=0 es 0.995 (disparo), seguido de scores altos que deben ser silenciados
    raw_scores = [0.995, 0.999, 0.998, 0.992, 0.5, 0.1, 0.995]
    dummy = DummyDetector(raw_scores)
    # W_rest = 3 pasos de silencio
    ref = RefractorySuppressor(dummy, w_rest=3, trigger_threshold=0.99)

    out = [ref.update(1.0) for _ in raw_scores]
    # t=0: emite 0.995 y programa silencio para t=1, 2, 3
    assert out[0] == 0.995
    # t=1: silenciado
    assert out[1] == 0.0
    # t=2: silenciado
    assert out[2] == 0.0
    # t=3: silenciado
    assert out[3] == 0.0
    # t=4: restaurado (raw=0.5)
    assert out[4] == 0.5
    # t=5: restaurado (raw=0.1)
    assert out[5] == 0.1
    # t=6: nuevo disparo (raw=0.995)
    assert out[6] == 0.995


def test_refractory_reset_invariance():
    """Verifica que reset() restaura el estado idéntico a una nueva instancia."""
    raw_scores = [0.995, 0.999, 0.5]
    dummy = DummyDetector(raw_scores)
    ref = RefractorySuppressor(dummy, w_rest=2, trigger_threshold=0.99)

    run1 = [ref.update(1.0) for _ in raw_scores]
    ref.reset()
    run2 = [ref.update(1.0) for _ in raw_scores]
    assert run1 == run2


def test_refractory_invalid_args():
    dummy = DummyDetector([0.5])
    with pytest.raises(ValueError):
        RefractorySuppressor(dummy, w_rest=-1)
    with pytest.raises(ValueError):
        RefractorySuppressor(dummy, trigger_threshold=1.5)

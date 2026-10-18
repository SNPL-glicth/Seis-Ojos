"""Tests para FrozenMetaClassifier: determinismo, causalidad, O(1), congelamiento."""

import pytest

from sixeyes.detectors.base import Detector
from sixeyes.experimental.frozen_meta_classifier import (
    FrozenMetaClassifier,
    N_FEATURES,
    sigmoid,
)


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


def test_output_range_and_suppression_only():
    seq = [0.5, 0.9, 0.995, 0.3, 0.99]
    m = FrozenMetaClassifier(Dummy(seq))
    outs = [m.update(0.0) for _ in seq]
    assert all(0.0 <= o <= 1.0 for o in outs)
    assert all(o <= b + 1e-12 for o, b in zip(outs, seq))  # jamas amplifica


def test_determinism_two_instances():
    seq = [0.1, 0.95, 0.99, 0.995, 0.2, 0.999]
    a = FrozenMetaClassifier(Dummy(seq))
    b = FrozenMetaClassifier(Dummy(seq))
    assert [a.update(0.0) for _ in seq] == [b.update(0.0) for _ in seq]


def test_causality_prefix_invariance():
    # Reproducir solo el prefijo da identicos outputs en el prefijo.
    full = [0.4, 0.93, 0.2, 0.99, 0.995, 0.1]
    m1 = FrozenMetaClassifier(Dummy(full))
    o1 = [m1.update(0.0) for _ in full]
    m2 = FrozenMetaClassifier(Dummy(full[:3]))
    o2 = [m2.update(0.0) for _ in full[:3]]
    assert o1[:3] == o2  # el futuro no contamina el pasado


def test_weights_frozen_during_streaming():
    m = FrozenMetaClassifier(Dummy([0.99] * 50))
    w0, c0 = m.weights, m.bias
    for _ in range(50):
        m.update(0.0)
    assert m.weights == w0 and m.bias == c0


def test_state_is_o1_bounded():
    m = FrozenMetaClassifier(Dummy([0.5] * 5000))
    for _ in range(5000):
        m.update(0.0)
    # Sin contenedores crecientes: solo escalares + detector interno.
    for k, v in m.__dict__.items():
        if k == "_det":
            continue
        assert not isinstance(v, (list, dict, set)), f"estado no acotado en {k}"
    assert m._t == 5000


def test_reset_invariance():
    seq = [0.9, 0.995, 0.2, 0.99]
    m = FrozenMetaClassifier(Dummy(seq))
    r1 = [m.update(0.0) for _ in seq]
    m.reset()
    assert (m._vol, m._lvl, m._prev, m._t) == (0.0, 0.0, 0.0, 0)
    r2 = [m.update(0.0) for _ in seq]
    assert r1 == r2


def test_set_weights_applies_and_validates():
    m = FrozenMetaClassifier(Dummy([0.99]))
    m.set_weights((1, 1, 1, 1, 1), 0.0)
    assert m.weights == (1, 1, 1, 1, 1) and m.bias == 0.0
    with pytest.raises(ValueError):
        m.set_weights((1, 2, 3), 0.0)


def test_invalid_args():
    with pytest.raises(ValueError):
        FrozenMetaClassifier(Dummy([0.5]), weights=(1, 2, 3))
    with pytest.raises(ValueError):
        FrozenMetaClassifier(Dummy([0.5]), alpha=0.0)
    with pytest.raises(ValueError):
        FrozenMetaClassifier(Dummy([0.5]), hi=1.5)


def test_sigmoid_extremes():
    assert sigmoid(100) == pytest.approx(1.0)
    assert sigmoid(-100) == pytest.approx(0.0)
    assert sigmoid(0) == pytest.approx(0.5)

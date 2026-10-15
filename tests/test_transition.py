import math

import numpy as np
import pytest

from sixeyes.transition.reservoir_state import ReservoirState
from sixeyes.transition.state_quantizer import OnlinePrototypeQuantizer
from sixeyes.transition.transition_model import TransitionModel
from sixeyes.transition.transition_scorer import TransitionSurpriseDetector


def _sine(n=300, seed=0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 12, n)
    return (np.sin(t) + 0.1 * rng.normal(size=n)).tolist()


def test_state_determinista():
    a, b = ReservoirState(seed=7), ReservoirState(seed=7)
    seq = _sine()
    ha = [a.update(v)[0] for v in seq]
    hb = [b.update(v)[0] for v in seq]
    for x, y in zip(ha, hb):
        if x is None or y is None:
            assert x is None and y is None
        else:
            assert np.allclose(x, y)


def test_state_prefijo_invariante():
    full = ReservoirState(seed=42)
    pre = ReservoirState(seed=42)
    seq = _sine(400)
    out_full = [full.update(v)[0] for v in seq]
    out_pre = [pre.update(v)[0] for v in seq[:200]]
    for x, y in zip(out_full[:200], out_pre):
        if x is None or y is None:
            assert x is None and y is None
        else:
            assert np.allclose(x, y)


def test_quantizer_asigna_mas_cercano_y_adapta():
    q = OnlinePrototypeQuantizer(n_states=2, dim=2, lr=0.5)
    q.update(np.array([0.0, 0.0]))
    q.update(np.array([10.0, 10.0]))
    assert q.assign(np.array([0.1, 0.0])) == 0
    assert q.assign(np.array([9.9, 10.0])) == 1
    antes = q._protos[0].copy()
    q.update(np.array([1.0, 1.0]))
    assert np.linalg.norm(q._protos[0] - np.array([1.0, 1.0])) < np.linalg.norm(
        antes - np.array([1.0, 1.0]))


def test_model_laplace_y_conteos():
    m = TransitionModel(n_states=3, epsilon=1.0)
    assert m.prob(0, 0) == pytest.approx(1 / 3)
    m.observe(0, 1)
    m.observe(0, 1)
    assert m.prob(0, 1) == pytest.approx(3 / 5)
    assert m.prob(0, 2) == pytest.approx(1 / 5)


def test_sorpresa_transicion_no_vista_mayor():
    d = TransitionSurpriseDetector(n_states=4, seed=42)
    alt = ([1.0] * 30 + [-1.0] * 30) * 6
    for v in alt:
        d.update(v)
    s_visto = d.update(1.0)
    d2 = TransitionSurpriseDetector(n_states=4, seed=42)
    for v in alt:
        d2.update(v)
    s_nuevo = d2.update(50.0)
    assert math.isfinite(s_visto) and math.isfinite(s_nuevo)
    assert s_nuevo > s_visto


def test_reset_replay_identico():
    seq = _sine(250)
    d = TransitionSurpriseDetector(seed=99)
    o1 = [d.update(v) for v in seq]
    d.reset()
    o2 = [d.update(v) for v in seq]
    assert o1 == o2


def test_scores_finitos_no_negativos():
    d = TransitionSurpriseDetector(seed=42)
    for v in _sine(600):
        s = d.update(v)
        assert math.isfinite(s) and s >= 0.0


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        OnlinePrototypeQuantizer(n_states=1)
    with pytest.raises(ValueError):
        TransitionModel(n_states=1)
    with pytest.raises(ValueError):
        TransitionModel(epsilon=0)

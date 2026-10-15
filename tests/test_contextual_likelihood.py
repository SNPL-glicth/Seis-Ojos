"""Pruebas unitarias para ContextualLikelihood."""

import math
import numpy as np
import pytest

from sixeyes.experimental.contextual_likelihood import ContextualLikelihood
from sixeyes.detectors.base import Detector


class DummyDetector(Detector):
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.idx = 0

    def update(self, value: float, timestamp: str = "") -> float:
        if self.idx < len(self.outputs):
            val = self.outputs[self.idx]
            self.idx += 1
            return val
        return 0.0

    def reset(self) -> None:
        self.idx = 0


def test_params_validation():
    with pytest.raises(ValueError, match="short_window"):
        ContextualLikelihood(short_window=0)
    with pytest.raises(ValueError, match="long_window"):
        ContextualLikelihood(short_window=10, long_window=10)
    with pytest.raises(ValueError, match="eps"):
        ContextualLikelihood(eps=0)


def test_stationary_noise_remains_moderate():
    """Bajo ruido estacionario, el score debe oscilar alrededor de 0.5 y nunca saturar a 1.0."""
    rng = np.random.RandomState(42)
    noise = rng.exponential(scale=1.0, size=2000)
    cl = ContextualLikelihood(short_window=10, long_window=500)

    scores = [cl.update(float(x)) for x in noise]
    # Tras periodo de calentamiento (ej. después de 200 pasos)
    stable_scores = scores[200:]
    assert max(stable_scores) < 0.99, f"Max score en ruido estacionario fue {max(stable_scores)}, no debió saturar"
    assert min(stable_scores) > 0.01, f"Min score fue {min(stable_scores)}"
    # La media debe rondar 0.5
    assert 0.40 <= np.mean(stable_scores) <= 0.60


def test_regime_change_bursts_to_near_one():
    """Un salto abrupto de error tras estabilidad debe elevar el score a > 0.99."""
    cl = ContextualLikelihood(short_window=5, long_window=500)
    
    # 500 pasos de error baseline muy pequeño ~0.01
    for _ in range(500):
        cl.update(0.01)

    # Ahora un régimen de error 1.0 (100 veces mayor)
    jump_scores = [cl.update(1.0) for _ in range(10)]
    assert any(s > 0.99 for s in jump_scores), f"Scores tras salto de régimen: {jump_scores}"


def test_reset_invariance():
    """reset() debe restaurar exactamente el comportamiento determinista."""
    cl = ContextualLikelihood(short_window=5, long_window=100)
    data = [0.1, 0.2, 0.15, 0.8, 0.9, 0.1, 0.05]
    
    res1 = [cl.update(x) for x in data]
    cl.reset()
    res2 = [cl.update(x) for x in data]
    
    np.testing.assert_allclose(res1, res2, atol=1e-12)


def test_base_detector_delegation():
    """Verifica que envuelve correctamente un detector base."""
    dummy = DummyDetector([0.1, 0.1, 0.1, 10.0])
    cl = ContextualLikelihood(base_detector=dummy, short_window=2, long_window=5)
    
    s = [cl.update(999.0) for _ in range(4)]
    assert len(s) == 4
    assert s[-1] > s[0]

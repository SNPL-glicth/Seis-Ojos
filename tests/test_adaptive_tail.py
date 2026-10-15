import math

import pytest

from sixeyes.experimental.adaptive_tail import AdaptiveTail


def test_warmup_da_cero():
    d = AdaptiveTail(window=100, warmup=10)
    for _ in range(10):
        assert d.update(0.99) == 0.0


def test_evento_en_serie_quieta_da_cola_alta():
    d = AdaptiveTail(window=500, warmup=50)
    for _ in range(500):
        d.update(0.1)
    s = d.update(0.95)
    assert s > 0.99


def test_mismo_score_en_serie_ruidosa_da_cola_baja():
    import random

    rng = random.Random(0)
    d = AdaptiveTail(window=500, warmup=50)
    for _ in range(500):
        d.update(rng.uniform(0.0, 1.0))
    s = d.update(0.95)
    assert s < 0.5


def test_reset_limpia():
    d = AdaptiveTail(window=100, warmup=10)
    for _ in range(200):
        d.update(0.5)
    d.reset()
    assert d.update(0.99) == 0.0


def test_nan_inf_dan_cero():
    d = AdaptiveTail()
    assert d.update(float("nan")) == 0.0
    assert d.update(float("inf")) == 0.0

import math

import numpy as np
import pytest

from sixeyes.experimental.level_rarity import LevelRarity


def test_warmup_devuelve_y_construye_bins():
    d = LevelRarity(n_bins=8, half_life=100, warmup=10, margin=0.1)
    for v in [1.0] * 10:
        assert d.update(v) == 0.0
    assert d._edges is not None
    assert len(d._edges) == 9  # n_bins + 1
    assert len(d._mass) == 10  # n_bins + 2 (2 overflow)


def test_pvalor_no_se_satura_cerca_de_1():
    """Con estacionariedad, el p-valor debe ser casi uniforme, no ~0.99."""
    rng = np.random.RandomState(0)
    d = LevelRarity(n_bins=20, half_life=100000, warmup=50, margin=0.1)
    vals = list(rng.normal(0, 1, 50))
    for v in vals:
        d.update(v)
    scores = [d.update(v) for v in rng.normal(0, 1, 2000)]
    scores = np.array(scores)
    assert scores.mean() == pytest.approx(0.5, abs=0.1)
    assert np.median(scores) == pytest.approx(0.5, abs=0.15)
    assert (scores > 0.99).mean() < 0.02


def test_nivel_raro_dentro_de_rango_da_score_alto():
    rng = np.random.RandomState(42)
    d = LevelRarity(n_bins=20, half_life=100000, warmup=200, margin=0.1)
    for v in rng.normal(0, 1, 200):
        d.update(v)
    s_normal = d.update(0.0)
    s_raro = d.update(1.8)
    assert s_raro > s_normal
    assert s_raro > 0.9


def test_overflow_extremo_da_score_1():
    d = LevelRarity(n_bins=20, half_life=100000, warmup=50, margin=0.1)
    for v in np.linspace(-1, 1, 50):
        d.update(v)
    s = d.update(100.0)
    assert s == pytest.approx(1.0)


def test_causalidad_y_reset():
    d = LevelRarity(n_bins=10, half_life=100, warmup=20, margin=0.1)
    seq = list(np.sin(np.linspace(0, 6, 200)))
    out1 = [d.update(v) for v in seq]
    d.reset()
    out2 = [d.update(v) for v in seq]
    assert out1 == out2


def test_decaimiento_reduce_masa():
    d = LevelRarity(n_bins=10, half_life=50, warmup=20, margin=0.1)
    for v in [0.0] * 20:
        d.update(v)
    d.update(5.0)
    b_rare = d._bin_index(5.0)
    m_inicial = float(d._mass[b_rare])
    for _ in range(200):
        d.update(0.0)
    dt = d._t - d._last_t[b_rare]
    m_efectiva = float(d._mass[b_rare] * math.pow(0.5, dt / d.half_life))
    assert m_efectiva < m_inicial * 0.9


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        LevelRarity(n_bins=2)
    with pytest.raises(ValueError):
        LevelRarity(half_life=0)
    with pytest.raises(ValueError):
        LevelRarity(warmup=1)
    with pytest.raises(ValueError):
        LevelRarity(margin=-0.1)

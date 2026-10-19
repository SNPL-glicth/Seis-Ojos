"""Pruebas del ContinuousPlasticityController: paridad, causalidad, cotas y casos límite."""

import math

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.experimental.continuous_plasticity_controller import (
    MAD_TO_SIGMA,
    ContinuousPlasticityController,
)
from sixeyes.experimental.predictive_readout_esn import PredictiveReadoutESN


def _sine(n, seed=0, noise=0.05):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / 50.0) + rng.normal(0, noise, n)


def test_protocolo_detector():
    det = ContinuousPlasticityController()
    assert isinstance(det, Detector)
    assert det.update(1.0) == 0.0


def test_desactivado_es_bit_a_bit_la_linea_base():
    x = _sine(1200, seed=1)
    x[600:800] += 3.0
    base = PredictiveReadoutESN(seed=42)
    ctrl = ContinuousPlasticityController(enable_controller=False, seed=42)
    sb = [base.update(float(v)) for v in x]
    sc = [ctrl.update(float(v)) for v in x]
    assert sb == sc
    for hb, hc in ((base._rec, ctrl._rec), (base._fc, ctrl._fc)):
        assert np.array_equal(hb.w, hc.w)
        assert np.array_equal(hb.P, hc.P)
    assert np.array_equal(base._h, ctrl._h)


def test_causalidad_scores_y_ganancia():
    rng = np.random.RandomState(3)
    common = list(_sine(400, seed=3))
    a = common + list(_sine(200, seed=4))
    b = common + list(rng.normal(50, 20, 200))
    da = ContinuousPlasticityController(record_trace=True, seed=9)
    db = ContinuousPlasticityController(record_trace=True, seed=9)
    sa = [da.update(float(v)) for v in a]
    sb = [db.update(float(v)) for v in b]
    assert sa[:400] == sb[:400]
    ta, tb = da.get_trace(), db.get_trace()
    for k in ("eta", "shock", "persistence", "g"):
        assert np.array_equal(ta[k][:400], tb[k][:400]), k


def test_shock_usa_solo_incrementos_anteriores():
    """Recalcula v_t offline con d_{t-W..t-1} y compara con la traza."""
    x = _sine(600, seed=5)
    x[400] += 2.0
    det = ContinuousPlasticityController(record_trace=True, seed=1)
    for v in x:
        det.update(float(v))
    tr = det.get_trace()
    W = det.window
    d = np.diff(x)  # d[i] = x[i+1] - x[i]
    checked = 0
    for t in range(200, 600):
        if not tr["active"][t]:
            continue
        hist = d[t - 1 - W:t - 1]  # incrementos d_{t-W}..d_{t-1}
        med = np.median(hist)
        s = max(MAD_TO_SIGMA * np.median(np.abs(hist - med)),
                np.mean(np.abs(hist)), 1e-6 * max(1.0, abs(x[t - 1])))
        expected = abs(x[t] - x[t - 1]) / s
        assert math.isclose(tr["shock"][t], expected, rel_tol=1e-12, abs_tol=1e-15), t
        checked += 1
    assert checked > 300


def test_cota_q_menor_igual_eta_menor_igual_1():
    x = _sine(1500, seed=6)
    x[600:800] += 3.0
    det = ContinuousPlasticityController(record_trace=True, seed=2)
    for v in x:
        det.update(float(v))
    tr = det.get_trace()
    act = tr["active"] > 0
    assert np.all(tr["eta"][act] <= 1.0)
    assert np.all(tr["eta"][act] >= tr["persistence"][act] - 1e-15)
    assert np.all(tr["eta"] >= 0.0)


def test_salto_abrupto_reduce_ganancia():
    x = _sine(900, seed=7)
    x[600:] += 3.0
    det = ContinuousPlasticityController(record_trace=True, seed=3)
    for v in x:
        det.update(float(v))
    tr = det.get_trace()
    assert tr["eta"][600] < 0.2
    assert np.median(tr["eta"][300:590]) > 0.99


def test_rampa_sin_ruido_no_dispara_shock():
    """MAD = 0 en una rampa perfecta; el suelo mean|d| debe evitar shocks."""
    x = np.linspace(0.0, 10.0, 800) + np.sin(np.arange(800) / 8.0)
    x[300:] += np.linspace(0.0, 4.0, 500)
    det = ContinuousPlasticityController(record_trace=True, seed=4)
    for v in x:
        det.update(float(v))
    tr = det.get_trace()
    assert np.min(tr["eta"][200:]) > 0.99


def test_senal_constante_y_cuantizada_estables():
    for x in (np.full(800, 5.0),
              np.cumsum(np.random.RandomState(8).rand(800) < 0.05).astype(float)):
        det = ContinuousPlasticityController(record_trace=True, seed=5)
        s = [det.update(float(v)) for v in x]
        tr = det.get_trace()
        assert all(math.isfinite(v) and v >= 0.0 for v in s)
        assert np.all(np.isfinite(tr["eta"])) and np.all(np.isfinite(tr["shock"]))
        assert np.all((tr["eta"] >= 0.0) & (tr["eta"] <= 1.0))
        for head in (det._rec, det._fc):
            assert np.all(np.isfinite(head.P))


def test_nan_inf_no_alteran_estado():
    det = ContinuousPlasticityController(record_trace=True, seed=6)
    for v in _sine(300, seed=9):
        det.update(float(v))
    w = det._fc.w.copy()
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert det.update(bad) == 0.0
    assert np.array_equal(det._fc.w, w)
    assert len(det.get_trace()["eta"]) == 303


def test_reset_replay_invariance():
    x = _sine(500, seed=10)
    x[300:350] += 2.0
    det = ContinuousPlasticityController(record_trace=True, seed=7)
    s1 = [det.update(float(v)) for v in x]
    e1 = det.get_trace()["eta"]
    det.reset()
    s2 = [det.update(float(v)) for v in x]
    assert s1 == s2
    assert np.array_equal(e1, det.get_trace()["eta"])

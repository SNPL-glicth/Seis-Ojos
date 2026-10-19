"""Pruebas del RegimeAwarePlasticityController (Fase V): paridad, memoria y causalidad."""

import math
import numpy as np
import pytest

from sixeyes.detectors.base import Detector
from sixeyes.experimental.predictive_readout_esn import PredictiveReadoutESN
from sixeyes.experimental.continuous_plasticity_controller import ContinuousPlasticityController
from sixeyes.experimental.regime_aware_controller import RegimeAwarePlasticityController


def _sine(n, seed=0, noise=0.05):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / 50.0) + rng.normal(0, noise, n)


def test_protocolo_detector():
    det = RegimeAwarePlasticityController()
    assert isinstance(det, Detector)
    assert det.update(1.0) == 0.0


def test_desactivado_es_bit_a_bit_la_linea_base():
    x = _sine(1000, seed=1)
    x[500:700] += 3.0
    base = PredictiveReadoutESN(seed=42)
    ctrl = RegimeAwarePlasticityController(enable_controller=False, seed=42)
    sb = [base.update(float(v)) for v in x]
    sc = [ctrl.update(float(v)) for v in x]
    assert sb == sc
    for hb, hc in ((base._rec, ctrl._rec), (base._fc, ctrl._fc)):
        assert np.array_equal(hb.w, hc.w)
        assert np.array_equal(hb.P, hc.P)
    assert np.array_equal(base._h, ctrl._h)


def test_memoria_desactivada_reproduce_fase_iii():
    x = _sine(1000, seed=2)
    x[500:700] += 3.0
    c3 = ContinuousPlasticityController(enable_controller=True, seed=42)
    c5 = RegimeAwarePlasticityController(enable_controller=True, enable_memory=False, seed=42)
    s3 = [c3.update(float(v)) for v in x]
    s5 = [c5.update(float(v)) for v in x]
    assert s3 == s5
    assert np.array_equal(c3._fc.w, c5._fc.w)


def test_meseta_anomala_mantiene_cuarentena_sin_absorcion():
    """Durante una meseta anómala no cartografiada, eta_t se mantiene en el techo de cuarentena (0.05)."""
    x = _sine(1200, seed=3)
    x[500:700] += 3.0
    ctrl = RegimeAwarePlasticityController(
        enable_controller=True,
        enable_memory=True,
        quarantine_cap=0.05,
        quarantine_steps=300,
        record_trace=True,
        seed=42,
    )
    scores = [ctrl.update(float(v)) for v in x]
    tr = ctrl.get_trace()
    
    # En t in [520, 680]: durante la meseta, el controlador debe reconocer que es un régimen incierto/no cartografiado
    meseta_etas = tr["eta"][520:680]
    assert np.all(meseta_etas <= 0.05 + 1e-6), f"eta en meseta superó el techo de cuarentena: max={np.max(meseta_etas)}"
    
    # Evaluar distorsión al final de la meseta anómala (t=700)
    ctrl_700 = RegimeAwarePlasticityController(
        enable_controller=True, enable_memory=True, quarantine_cap=0.05, seed=42
    )
    base_700 = PredictiveReadoutESN(seed=42)
    for v in x[:700]:
        ctrl_700.update(float(v))
        base_700.update(float(v))
        
    delta_ctrl = np.linalg.norm(ctrl_700._fc.w) - 1.19  # peso nominal ~1.19
    delta_base = np.linalg.norm(base_700._fc.w) - 1.19
    assert delta_ctrl < 0.55 * delta_base, "El controlador debió reducir la deformación al menos un 45%"


def test_reconoce_retorno_a_regimen_nominal():
    """Al regresar a la señal normal tras la meseta (t > 700), reconoce Regime 0 (KNOWN)."""
    x = _sine(1200, seed=4)
    x[500:700] += 3.0
    ctrl = RegimeAwarePlasticityController(
        enable_controller=True,
        enable_memory=True,
        seed=42,
    )
    for v in x[:800]:
        ctrl.update(float(v))
        
    telem = ctrl.get_memory_telemetry()
    assert telem["regime_status"] == "KNOWN"
    assert telem["regime_id"] == 0
    assert telem["regime_distance"] <= 1.25

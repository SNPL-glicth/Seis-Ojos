"""Pruebas automatizadas de diagnóstico temporal y métricas de Fase IV.

Garantiza la reproducibilidad de las 5 afirmaciones clave:
1. El shock reduce la plasticidad según el contrato.
2. Los pesos permanecen protegidos durante el inicio del shock.
3. La persistencia reactiva el aprendizaje bajo condiciones causales exactas (cruce t=512).
4. La cola de recuperación residual se mide de forma determinista y reproducible.
5. Causalidad estricta e invariancia temporal ante intervenciones futuras.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from sixeyes.experimental.continuous_plasticity_controller import (
    ContinuousPlasticityController,
)
from sixeyes.experimental.predictive_readout_esn import PredictiveReadoutESN


def _generate_synthetic_scenario(n: int = 1200, seed: int = 42):
    t = np.arange(n)
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, 0.05, size=n)
    signal = np.sin(2 * np.pi * t / 50.0) + noise
    labels = np.zeros(n, dtype=int)
    signal[500:700] += 3.0
    labels[500:700] = 1
    return signal, labels


def test_shock_reduces_gain_contract():
    """El shock reduce la plasticidad inmediatamente según el contrato v > shock_center."""
    signal, _ = _generate_synthetic_scenario()
    ctrl = ContinuousPlasticityController(
        enable_controller=True, window=50, shock_center=6.0, shock_slope=1.5,
        record_trace=True, seed=42
    )
    for val in signal[:501]:
        ctrl.update(float(val))
    tr = ctrl.get_trace()
    
    # En t=500 (primer paso de la meseta)
    v_500 = tr["shock"][500]
    eta_500 = tr["eta"][500]
    g_500 = tr["g"][500]
    
    assert v_500 > 20.0, f"Shock v_500={v_500} debió ser > 20 ante salto de 3 sigmas"
    assert g_500 > 0.999, f"Memoria de shock g_500={g_500} debió saturar cerca de 1"
    assert eta_500 < 0.10, f"Ganancia eta_500={eta_500} debió caer por debajo de 0.10 (contrato de choque)"


def test_weights_protected_during_shock_onset():
    """Los pesos de RLS quedan protegidos en el primer paso del shock frente a la base."""
    signal, _ = _generate_synthetic_scenario()
    base = PredictiveReadoutESN(seed=42)
    ctrl = ContinuousPlasticityController(
        enable_controller=True, window=50, record_trace=True, seed=42
    )
    
    # Ejecutar hasta t=499
    for val in signal[:500]:
        base.update(float(val))
        ctrl.update(float(val))
        
    w_base_pre = base._fc.w.copy()
    w_ctrl_pre = ctrl._fc.w.copy()
    
    # Dar paso t=500
    base.update(float(signal[500]))
    ctrl.update(float(signal[500]))
    
    delta_w_base = float(np.linalg.norm(base._fc.w - w_base_pre))
    delta_w_ctrl = float(np.linalg.norm(ctrl._fc.w - w_ctrl_pre))
    
    # En la base, el shock deforma fuertemente los pesos inmediatamente.
    # En el controlador, la deformación debe ser un orden de magnitud menor.
    assert delta_w_base > 0.20, f"delta_w_base={delta_w_base} debió ser > 0.20"
    assert delta_w_ctrl < 0.05, f"delta_w_ctrl={delta_w_ctrl} debió ser < 0.05 (protección de pesos)"
    assert delta_w_ctrl < 0.2 * delta_w_base, "El controlador protegió menos del 80% respecto a la base"


def test_persistence_reactivation_crossing_at_step_512():
    """La persistencia direccional q_t reactiva el aprendizaje cruzando eta=0.5 en t=512."""
    signal, _ = _generate_synthetic_scenario()
    ctrl = ContinuousPlasticityController(
        enable_controller=True, window=50, shock_center=6.0, shock_slope=1.5,
        record_trace=True, seed=42
    )
    for val in signal[:530]:
        ctrl.update(float(val))
        
    tr = ctrl.get_trace()
    
    # En t=511 (paso 11 post-shock): eta debe seguir bajo 0.50
    assert tr["eta"][511] < 0.50, f"eta_511={tr['eta'][511]} debió ser < 0.50"
    # En t=512 (paso 12 post-shock): cruce de umbral eta >= 0.50
    assert tr["eta"][512] >= 0.50, f"eta_512={tr['eta'][512]} debió ser >= 0.50"
    
    # Verificar que el cruce es consecuencia analítica de histéresis y persistencia
    g_512 = tr["g"][512]
    q_512 = tr["persistence"][512]
    expected_eta = 1.0 - g_512 * (1.0 - q_512)
    assert math.isclose(tr["eta"][512], expected_eta, rel_tol=1e-12)
    
    # Confirmar que q creció monótonamente entre t=501 y t=512
    q_window = tr["persistence"][501:513]
    assert np.all(np.diff(q_window) >= -1e-12), "q_t debió crecer monótonamente durante el error persistente"


def test_recovery_tail_deterministic_measurement():
    """Medición determinista de la duración de la cola residual tras el retorno a la normalidad."""
    signal, _ = _generate_synthetic_scenario()
    ctrl = ContinuousPlasticityController(
        enable_controller=True, window=50, shock_center=6.0, shock_slope=1.5,
        record_trace=True, seed=42
    )
    scores = [ctrl.update(float(val)) for val in signal]
    
    # Nivel pre-anomalía en régimen estacionario [350, 500)
    baseline_level = float(np.mean(scores[350:500]))
    threshold = 3.0 * baseline_level
    
    # Ventana post-retorno [700, 850)
    post_return_scores = np.array(scores[700:850])
    tail_steps = int(np.sum(post_return_scores > threshold))
    
    # La cola debe ser finita, positiva y reproducible
    assert 30 <= tail_steps <= 90, f"tail_steps={tail_steps} fuera de rango esperado [30, 90]"
    
    # En t=700 (retorno), se produce un segundo shock simétrico
    tr = ctrl.get_trace()
    assert tr["shock"][700] > 15.0, "El retorno debió disparar un shock cinemático v_700 > 15"
    assert tr["eta"][700] < 0.20, "eta_700 debió volver a caer ante el shock inverso de retorno"


def test_strict_causality_under_future_interventions():
    """Modificar la señal en t >= 500 no altera en nada las trazas ni los pesos para t < 500."""
    signal_a, _ = _generate_synthetic_scenario(seed=42)
    signal_b = signal_a.copy()
    
    # Intervención en señal B a partir de t=500: en vez de salto +3, es salto -10 con ruido caótico
    signal_b[500:] = -10.0 + np.random.RandomState(99).normal(0, 5.0, size=len(signal_b) - 500)
    
    ctrl_a = ContinuousPlasticityController(record_trace=True, seed=42)
    ctrl_b = ContinuousPlasticityController(record_trace=True, seed=42)
    
    w_rec_a_at_500 = None
    w_rec_b_at_500 = None

    for i, (va, vb) in enumerate(zip(signal_a, signal_b)):
        ctrl_a.update(float(va))
        ctrl_b.update(float(vb))
        if i == 499:
            w_rec_a_at_500 = ctrl_a._rec.w.copy()
            w_rec_b_at_500 = ctrl_b._rec.w.copy()
            w_fc_a_at_500 = ctrl_a._fc.w.copy()
            w_fc_b_at_500 = ctrl_b._fc.w.copy()
        
    tr_a = ctrl_a.get_trace()
    tr_b = ctrl_b.get_trace()
    
    for key in ("shock", "persistence", "g", "eta"):
        assert np.array_equal(tr_a[key][:500], tr_b[key][:500]), f"Fuga causal detectada en métrica {key}"
        
    # Y los pesos al final del paso t=499 (antes de divergir en t=500) deben ser exactamente idénticos
    assert np.array_equal(w_rec_a_at_500, w_rec_b_at_500)
    assert np.array_equal(w_fc_a_at_500, w_fc_b_at_500)

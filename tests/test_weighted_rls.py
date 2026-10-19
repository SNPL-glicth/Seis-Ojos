"""Pruebas matemáticas del RLS ponderado (Fase III).

Objetivo: intentar refutar la formulación, no solo verificar los extremos.
  1. eta = 0 congela w y P bit a bit.
  2. eta = 1 reproduce RLS estándar (fórmula) y la cabeza es bit a bit la base.
  3. Forma de información: P_t^{-1} = lam_eff P^{-1} + eta h h^T para eta intermedios.
  4. Equivalencia con mínimos cuadrados ponderados por lotes (interpretación estadística).
  5. Positividad, simetría, finitud y condicionamiento en secuencias largas.
  6. Cota de información: P_t <= P_{t-1} / lam_eff (orden de Loewner).
"""

import numpy as np
import pytest

from sixeyes.experimental.predictive_readout_esn import RLS_FORGET, _RLSHead
from sixeyes.experimental.weighted_rls import (
    WeightedRLSHead,
    effective_forgetting,
    weighted_rls_step,
)

LAM = RLS_FORGET


def _warm_state(n: int, steps: int, seed: int):
    """Estado (w, P) realista tras RLS estándar con rasgos tipo reservorio."""
    rng = np.random.default_rng(seed)
    head = _RLSHead(n)
    for _ in range(steps):
        h = np.tanh(rng.normal(size=n))
        y = float(rng.normal())
        head.update(h, y, head.predict(h))
    return head.w.copy(), head.P.copy(), rng


def test_eta_cero_congela_exactamente():
    w, P, rng = _warm_state(16, 300, 0)
    h = np.tanh(rng.normal(size=16))
    w1, P1 = weighted_rls_step(w, P, h, e=5.0, lam=LAM, eta=0.0)
    assert np.array_equal(w1, w)
    assert np.array_equal(P1, P)


def test_eta_uno_reproduce_rls_estandar():
    w, P, rng = _warm_state(16, 300, 1)
    h = np.tanh(rng.normal(size=16))
    y = 0.7
    ref = _RLSHead(16)
    ref.w, ref.P = w.copy(), P.copy()
    pred = ref.predict(h)
    ref.update(h, y, pred)
    w1, P1 = weighted_rls_step(w, P, h, e=y - pred, lam=LAM, eta=1.0, symmetrize=False)
    np.testing.assert_allclose(w1, ref.w, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(P1, ref.P, rtol=1e-12, atol=1e-14)


def test_cabeza_eta_uno_es_bit_a_bit_la_base():
    rng = np.random.default_rng(2)
    a, b = _RLSHead(16), WeightedRLSHead(16)
    for _ in range(500):
        h = np.tanh(rng.normal(size=16))
        y = float(rng.normal())
        a.update(h, y, a.predict(h))
        b.update(h, y, b.predict(h))
    assert np.array_equal(a.w, b.w)
    assert np.array_equal(a.P, b.P)


@pytest.mark.parametrize("eta", [1e-6, 0.01, 0.1, 0.37, 0.5, 0.9, 0.999])
def test_forma_de_informacion(eta):
    w, P, rng = _warm_state(8, 200, 3)
    h = np.tanh(rng.normal(size=8))
    _, P1 = weighted_rls_step(w, P, h, e=1.0, lam=LAM, eta=eta)
    lam_eff = effective_forgetting(LAM, eta)
    info_expected = lam_eff * np.linalg.inv(P) + eta * np.outer(h, h)
    info_got = np.linalg.inv(P1)
    rel = np.linalg.norm(info_got - info_expected) / np.linalg.norm(info_expected)
    assert rel < 1e-9, f"eta={eta}: error relativo {rel:.2e}"


def test_equivalencia_minimos_cuadrados_ponderados_por_lotes():
    """La recursión debe coincidir con la solución cerrada de WLS con descuento."""
    rng = np.random.default_rng(4)
    n, T, delta = 6, 300, 1.0
    H = np.tanh(rng.normal(size=(T, n)))
    w_true = rng.normal(size=n)
    Y = H @ w_true + 0.1 * rng.normal(size=T)
    etas = rng.uniform(0.0, 1.0, size=T)
    etas[rng.random(T) < 0.15] = 0.0
    etas[rng.random(T) < 0.15] = 1.0

    w = np.zeros(n)
    P = np.eye(n) * delta
    for t in range(T):
        e = float(Y[t] - w @ H[t])
        w, P = weighted_rls_step(w, P, H[t], e, LAM, float(etas[t]))

    lam_eff = 1.0 - etas * (1.0 - LAM)
    A = np.prod(lam_eff) * np.eye(n) / delta
    b = np.zeros(n)
    for i in range(T):
        c = np.prod(lam_eff[i + 1:])
        A += c * etas[i] * np.outer(H[i], H[i])
        b += c * etas[i] * Y[i] * H[i]
    w_batch = np.linalg.solve(A, b)
    np.testing.assert_allclose(w, w_batch, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(np.linalg.inv(P), A, rtol=1e-7, atol=1e-9)


def test_positividad_simetria_y_condicionamiento_secuencia_larga():
    rng = np.random.default_rng(5)
    n, T = 16, 5000
    w = np.zeros(n)
    P = np.eye(n)
    w_std = np.zeros(n)
    P_std = np.eye(n)
    max_cond = max_cond_std = 0.0
    for t in range(T):
        h = np.tanh(rng.normal(size=n))
        y = float(rng.normal())
        r = rng.random()
        eta = 0.0 if r < 0.2 else 1.0 if r < 0.4 else 1e-8 if r < 0.5 else float(rng.random())
        w, P = weighted_rls_step(w, P, h, y - w @ h, LAM, eta)
        w_std, P_std = weighted_rls_step(w_std, P_std, h, y - w_std @ h, LAM, 1.0)
        assert np.all(np.isfinite(P)) and np.all(np.isfinite(w))
        assert np.max(np.abs(P - P.T)) == 0.0
        ev = np.linalg.eigvalsh(P)
        assert ev[0] > 0.0, f"P no definida positiva en t={t}: {ev[0]}"
        max_cond = max(max_cond, ev[-1] / ev[0])
        ev_s = np.linalg.eigvalsh(P_std)
        max_cond_std = max(max_cond_std, ev_s[-1] / ev_s[0])
    assert max_cond < 1e8
    # Menos olvido efectivo no debe empeorar el condicionamiento en órdenes de magnitud.
    assert max_cond <= 10.0 * max_cond_std


@pytest.mark.parametrize("eta", [0.05, 0.5, 0.95])
def test_cota_de_loewner(eta):
    """P_t <= P_{t-1}/lam_eff: la observación solo puede añadir información."""
    w, P, rng = _warm_state(8, 200, 6)
    P = 0.5 * (P + P.T)
    h = np.tanh(rng.normal(size=8))
    _, P1 = weighted_rls_step(w, P, h, e=1.0, lam=LAM, eta=eta)
    gap = P / effective_forgetting(LAM, eta) - P1
    assert np.linalg.eigvalsh(0.5 * (gap + gap.T))[0] > -1e-12


def test_eta_fuera_de_rango_rechazado():
    w, P, rng = _warm_state(4, 10, 7)
    h = np.ones(4)
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            weighted_rls_step(w, P, h, 1.0, LAM, bad)


def test_anti_windup_en_ruta_ponderada():
    head = WeightedRLSHead(4)
    head.P[0, 0] = np.inf
    head.eta = 0.5
    head.update(np.ones(4), 1.0, 0.0)
    assert np.array_equal(head.w, np.zeros(4))
    assert np.array_equal(head.P, np.eye(4))

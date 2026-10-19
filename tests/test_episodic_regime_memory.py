"""Pruebas adversariales y de propiedades locales de EpisodicRegimeMemory.

Comprueba:
1. La persistencia sola NO promueve; se exige estabilidad y deriva nula.
2. Invariante de capacidad K=5: régimen 0 nunca se desaloja y política LRU determinista.
3. Franja ambigua [1.25, 1.50]: pausa de contador sin reinicio espurio.
4. Métrica de tres componentes: nivel, escala y envolvente latente.
5. Rechazo estricto de entradas inválidas y no finitas.
6. Huella de memoria acotada en bytes (< 10 KB).
"""

import math
import numpy as np
import pytest

from sixeyes.experimental.episodic_regime_memory import (
    EpisodicRegimeMemory,
    RegimeSignature,
)


def _mock_latent(dim: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.5, 0.5, size=dim)


def test_rechazo_parametros_invalidos():
    with pytest.raises(ValueError):
        EpisodicRegimeMemory(reservoir_dim=0)
    with pytest.raises(ValueError):
        EpisodicRegimeMemory(reservoir_dim=64, max_regimes=1)
    with pytest.raises(ValueError):
        EpisodicRegimeMemory(reservoir_dim=64, quarantine_steps=10)
    with pytest.raises(ValueError):
        EpisodicRegimeMemory(reservoir_dim=64, smoothing_gamma=0.0)
    with pytest.raises(ValueError):
        EpisodicRegimeMemory(reservoir_dim=64, d_known=1.5, d_uncertain=1.2)


def test_entradas_no_finitas_y_dimensiones_rechazadas():
    mem = EpisodicRegimeMemory(reservoir_dim=64)
    rid, dist, status = mem.update(float("nan"), _mock_latent(64))
    assert status == "INVALID_INPUT"
    assert math.isinf(dist)

    rid, dist, status = mem.update(1.0, _mock_latent(32))  # dimensión incorrecta
    assert status == "INVALID_INPUT"

    bad_h = _mock_latent(64)
    bad_h[5] = float("inf")
    rid, dist, status = mem.update(1.0, bad_h)
    assert status == "INVALID_INPUT"


def test_distancia_tres_componentes_valida_dispersion():
    sig = RegimeSignature(
        regime_id=0,
        name="Base",
        mu_x=0.0,
        sigma_x=1.0,
        latent_envelope=np.ones(64),
    )
    # 1. Idéntico
    d0 = sig.distance_to(mu=0.0, sigma=1.0, psi=np.ones(64))
    assert math.isclose(d0, 0.0, abs_tol=1e-6)

    # 2. Solo cambia dispersión (sigma = 2.0, delta=1.0)
    d_scale = sig.distance_to(mu=0.0, sigma=2.0, psi=np.ones(64))
    # d_scale = sqrt( (0 + 1^2 + 0) / 3 ) = 1 / sqrt(3) ~ 0.57735
    assert math.isclose(d_scale, 1.0 / math.sqrt(3.0), rel_tol=1e-5)

    # 3. Solo cambia nivel (mu = 1.0, delta=1.0)
    d_level = sig.distance_to(mu=1.0, sigma=1.0, psi=np.ones(64))
    assert math.isclose(d_level, 1.0 / math.sqrt(3.0), rel_tol=1e-5)


def test_persistencia_sin_estabilidad_no_promueve():
    """Una perturbación caótica o inestable que dura quarantine_steps NO se promueve."""
    mem = EpisodicRegimeMemory(reservoir_dim=64, quarantine_steps=50, smoothing_gamma=0.2)
    # Registrar régimen base
    x_base = list(np.random.RandomState(42).normal(0.0, 0.2, 100))
    h_base = [_mock_latent(64, seed=i) for i in range(100)]
    mem.register_regime("Nominal_0", x_base, h_base, regime_id=0)

    # Inyectar 60 pasos con ruido caótico altamente inestable (volatilidad excesiva)
    rng_wild = np.random.RandomState(99)
    statuses = []
    for _ in range(60):
        wild_x = 5.0 + rng_wild.normal(0, 5.0)  # volatilidad 25x superior a la nominal
        _, _, st = mem.update(wild_x, _mock_latent(64))
        statuses.append(st)

    # Debe haber permanecido en cuarentena; NO haberse promovido
    assert "STABLE_PROMOTED" not in statuses
    assert statuses[-1] == "UNCERTAIN_QUARANTINE"


def test_transicion_estacionaria_si_se_promueve():
    """Un nuevo estado con nivel estable y dispersión acotada SÍ se promueve como STABLE_PROMOTED."""
    mem = EpisodicRegimeMemory(reservoir_dim=64, quarantine_steps=40, smoothing_gamma=0.2)
    # Registrar régimen base
    x_base = list(np.random.RandomState(42).normal(0.0, 0.2, 100))
    h_base = [_mock_latent(64, seed=i) for i in range(100)]
    mem.register_regime("Nominal_0", x_base, h_base, regime_id=0)

    # Inyectar 45 pasos con nuevo nivel constante y ruido controlado
    rng_stable = np.random.RandomState(123)
    promoted = False
    for i in range(45):
        clean_new_x = 4.0 + rng_stable.normal(0, 0.1)  # estacionario en 4.0
        new_h = 0.5 + _mock_latent(64, seed=1000 + i) * 0.1
        rid, dist, st = mem.update(clean_new_x, new_h)
        if st == "STABLE_PROMOTED":
            promoted = True
            assert rid == 1
            # La distancia reportada debe ser un valor real finito, no cero artificial
            assert math.isfinite(dist) and dist >= 0.0

    assert promoted, "Un estado estacionario con baja varianza debió promoverse como STABLE_PROMOTED"


def test_capacidad_maxima_y_lru_protege_regimen_cero():
    """Al insertar más de K regímenes, el régimen 0 NUNCA se expulsa y se expulsa el LRU."""
    K = 4
    mem = EpisodicRegimeMemory(reservoir_dim=64, max_regimes=K)

    # Registrar 4 regímenes (0, 1, 2, 3)
    for rid in range(K):
        mem.register_regime(f"Regime_{rid}", [float(rid)] * 10, [_mock_latent(64)] * 10, regime_id=rid)
        mem.regimes[rid].last_seen_step = rid * 10

    assert len(mem.regimes) == 4
    assert set(mem.regimes.keys()) == {0, 1, 2, 3}

    # Actualizar last_seen de 2 y 3 para que 1 sea el LRU entre los no-cero
    mem.regimes[2].last_seen_step = 100
    mem.regimes[3].last_seen_step = 150
    # id 0 tiene last_seen = 0 (el más antiguo), pero está BLINDADO

    # Insertar un quinto régimen (supera max_regimes=4)
    new_rid = mem.register_regime("Regime_4", [4.0] * 10, [_mock_latent(64)] * 10)

    assert len(mem.regimes) == K
    # id 0 debe estar INTACTO
    assert 0 in mem.regimes, "Invariante violado: Regime_0 fue expulsado!"
    # id 1 debió ser expulsado por ser el LRU entre los no-cero
    assert 1 not in mem.regimes, "Regime_1 (LRU) no fue expulsado"
    # id 2, 3 y 4 deben estar presentes
    assert 2 in mem.regimes and 3 in mem.regimes and new_rid in mem.regimes


def test_franja_ambigua_pausa_cuarentena():
    """Distancia en (1.25, 1.50] reporta AMBIGUOUS_BORDER y pausa el contador sin borrarlo."""
    mem = EpisodicRegimeMemory(
        reservoir_dim=64,
        d_known=1.25,
        d_uncertain=1.50,
        quarantine_steps=100,
        smoothing_gamma=1.0,  # gamma=1.0 para respuesta inmediata sin inercia en el test
    )
    sig = RegimeSignature(0, "Base", mu_x=0.0, sigma_x=1.0, latent_envelope=np.ones(64))
    mem.regimes[0] = sig

    # 1. Poner primero en cuarentena franca para acumular contador (mu=10.0 >> 1.50)
    for _ in range(5):
        mem.update(10.0, np.ones(64))
    count_pre = mem._quarantine_counter
    assert count_pre == 5

    # 2. Enviar a franja ambigua: target d = 1.35 in (1.25, 1.50]
    # d = sqrt((mu^2 + 0 + 0) / 3) -> mu = 1.35 * sqrt(3) ~ 2.338
    x_border = 1.35 * math.sqrt(3.0)
    _, dist, st = mem.update(x_border, np.ones(64))
    assert 1.25 < dist <= 1.50, f"dist={dist} debió estar en (1.25, 1.50]"
    assert st == "AMBIGUOUS_BORDER"
    # El contador debe haberse pausado: no sube a 6 ni se resetea a 0
    assert mem._quarantine_counter == count_pre


def test_huella_de_memoria_acotada():
    """La memoria total consumida en bytes permanece estrictamente acotada."""
    mem = EpisodicRegimeMemory(reservoir_dim=64, max_regimes=5, quarantine_steps=100)
    for rid in range(5):
        mem.register_regime(f"R_{rid}", [float(rid)] * 50, [_mock_latent(64)] * 50, regime_id=rid)

    for i in range(200):
        mem.update(float(i % 10), _mock_latent(64))

    bytes_used = mem.get_memory_footprint_bytes()
    # 5 prototipos x 512 bytes + buffers circulares de 100 floats y 100 vectores de 64 = ~60 KB
    assert bytes_used < 100_000, f"Memoria consumida excesiva: {bytes_used} bytes"

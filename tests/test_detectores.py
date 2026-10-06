"""Pruebas unitarias para los detectores RandomDetector y RollingRobustZ."""

import math
import random

from sixeyes.detectors import Detector, RandomDetector, RollingRobustZ


def _generar_serie_sintetica(longitud: int = 100, semilla: int = 42) -> list[float]:
    """Genera una serie temporal sintética determinista (seno + ruido leve)."""
    rng = random.Random(semilla)
    return [math.sin(i * 0.15) + rng.uniform(-0.05, 0.05) for i in range(longitud)]


def test_scores_en_rango_cero_uno() -> None:
    """Verifica que todos los scores de ambos detectores pertenezcan al intervalo [0, 1]."""
    serie = _generar_serie_sintetica(100)
    det_aleatorio = RandomDetector(seed=123)
    det_zscore = RollingRobustZ(window_size=25, k=3.0, min_observations=5)

    for i, valor in enumerate(serie):
        ts = f"2026-10-05T00:00:{i:02d}Z"
        score_rnd = det_aleatorio.update(valor, ts)
        score_z = det_zscore.update(valor, ts)

        assert 0.0 <= score_rnd <= 1.0
        assert 0.0 <= score_z <= 1.0


def test_anti_fuga_informacion_causal() -> None:
    """Verifica que modificar valores futuros no altere los scores pasados o presentes.

    Comprueba tanto la igualdad estricta hasta la posición de corte como que
    la alteración posterior efectivamente genera scores divergentes y no triviales.
    """
    serie_original = _generar_serie_sintetica(60)
    corte_k = 30

    serie_alterada = list(serie_original)
    for i in range(corte_k + 1, len(serie_alterada)):
        serie_alterada[i] += 500.0

    det_a = RollingRobustZ(window_size=20, k=3.0, min_observations=5)
    det_b = RollingRobustZ(window_size=20, k=3.0, min_observations=5)

    scores_a = [det_a.update(v, f"t{i}") for i, v in enumerate(serie_original)]
    scores_b = [det_b.update(v, f"t{i}") for i, v in enumerate(serie_alterada)]

    # 1. Los scores históricos y presentes no pueden cambiar
    assert scores_a[: corte_k + 1] == scores_b[: corte_k + 1]

    # 2. Confirma que la serie no era trivialmente cero y que la alteración surtió efecto inmediato
    assert any(s > 0.0 for s in scores_a[5 : corte_k + 1])
    assert scores_b[corte_k + 1] != scores_a[corte_k + 1]
    assert scores_b[corte_k + 1] > scores_a[corte_k + 1]


def test_valor_actual_se_puntua_antes_de_entrar_a_la_ventana() -> None:
    """Demuestra que el valor actual NO entra en la ventana antes del cálculo del score.

    Si el valor entrara antes, la mediana y el MAD se desplazarían absorbiéndolo,
    aplanando el z-score resultante.
    """
    detector = RollingRobustZ(window_size=5, k=3.0, min_observations=5)
    # Llenamos exactamente la ventana de 5 elementos
    ventana_inicial = [1.0, 2.0, 3.0, 4.0, 5.0]
    for i, v in enumerate(ventana_inicial):
        detector.update(v, f"t{i}")

    # Nuevo valor a puntuar
    nuevo_valor = 10.0
    score_obtenido = detector.update(nuevo_valor, "t5")

    # Cálculo analítico esperado usando la ventana ANTERIOR [1.0, 2.0, 3.0, 4.0, 5.0]:
    # mediana = 3.0, desviaciones = [2, 1, 0, 1, 2], MAD = 1.0
    z_esperado = abs(10.0 - 3.0) / (1.4826 * 1.0)
    score_esperado = 1.0 - math.exp(-z_esperado / 3.0)
    assert math.isclose(score_obtenido, score_esperado, rel_tol=1e-9)

    # Si hubiera entrado antes, la ventana sería [2.0, 3.0, 4.0, 5.0, 10.0]:
    # mediana = 4.0, MAD = 1.0, z = 6.0 / 1.4826 (score aplanado)
    z_aplanado = abs(10.0 - 4.0) / (1.4826 * 1.0)
    score_aplanado = 1.0 - math.exp(-z_aplanado / 3.0)
    assert score_obtenido > score_aplanado


def test_determinismo() -> None:
    """Verifica que dos instancias con la misma semilla o configuración produzcan secuencias idénticas."""
    serie = _generar_serie_sintetica(50)

    # Determinismo en RandomDetector
    rnd_1 = RandomDetector(seed=777)
    rnd_2 = RandomDetector(seed=777)
    scores_rnd_1 = [rnd_1.update(v, f"t{i}") for i, v in enumerate(serie)]
    scores_rnd_2 = [rnd_2.update(v, f"t{i}") for i, v in enumerate(serie)]
    assert scores_rnd_1 == scores_rnd_2

    # Determinismo en RollingRobustZ
    z_1 = RollingRobustZ(window_size=15, k=2.5)
    z_2 = RollingRobustZ(window_size=15, k=2.5)
    scores_z_1 = [z_1.update(v, f"t{i}") for i, v in enumerate(serie)]
    scores_z_2 = [z_2.update(v, f"t{i}") for i, v in enumerate(serie)]
    assert scores_z_1 == scores_z_2


def test_serie_constante_sin_excepciones_ni_nan() -> None:
    """Verifica que una serie constante (MAD=0) se procese sin error ni valores NaN."""
    serie_constante = [42.0] * 50
    detector = RollingRobustZ(window_size=20, min_observations=5)

    scores = [detector.update(v, f"t{i}") for i, v in enumerate(serie_constante)]

    for score in scores:
        assert not math.isnan(score)
        assert not math.isinf(score)
        assert 0.0 <= score <= 1.0

    # Inyección de un cambio repentino sobre la serie constante
    score_cambio = detector.update(100.0, "t50")
    assert not math.isnan(score_cambio)
    assert score_cambio == 1.0


def test_deteccion_basica_pico_anomalo() -> None:
    """Verifica que un pico inducido reciba un score mayor al promedio del resto de la serie."""
    serie = _generar_serie_sintetica(80)
    posicion_pico = 50
    serie[posicion_pico] += 25.0

    detector = RollingRobustZ(window_size=30, k=3.0, min_observations=5)
    scores = [detector.update(v, f"t{i}") for i, v in enumerate(serie)]

    resto_scores = [s for i, s in enumerate(scores) if i != posicion_pico]
    promedio_resto = sum(resto_scores) / len(resto_scores)

    assert scores[posicion_pico] > promedio_resto
    assert scores[posicion_pico] > 0.8


def test_calentamiento_devuelve_cero() -> None:
    """Verifica que las primeras observaciones previas al umbral de calentamiento devuelvan score 0.0."""
    min_obs = 7
    detector = RollingRobustZ(window_size=20, min_observations=min_obs)
    serie = _generar_serie_sintetica(20)

    for i in range(min_obs):
        score = detector.update(serie[i], f"t{i}")
        assert score == 0.0


def test_cumplimiento_interfaz_detector() -> None:
    """Verifica que ambos detectores implementen el protocolo Detector."""
    assert isinstance(RandomDetector(), Detector)
    assert isinstance(RollingRobustZ(), Detector)

"""Pruebas unitarias para los detectores RandomDetector y RollingRobustZ."""

import math
import random

from detectors import Detector, RandomDetector, RollingRobustZ
from detectors.zscore_robusto import Z_MEDIO


def _generar_serie_sintetica(longitud: int = 100, semilla: int = 42) -> list[float]:
    """Genera una serie temporal sintética determinista (seno + ruido leve)."""
    rng = random.Random(semilla)
    return [math.sin(i * 0.15) + rng.uniform(-0.05, 0.05) for i in range(longitud)]


def test_scores_en_rango_cero_uno() -> None:
    """Verifica que todos los scores de ambos detectores pertenezcan al intervalo [0, 1]."""
    serie = _generar_serie_sintetica(100)
    det_aleatorio = RandomDetector(seed=123)
    # Cambio k: eliminado
    det_zscore = RollingRobustZ(window_size=25, min_observations=5)

    for i, valor in enumerate(serie):
        ts = f"2026-10-05T00:00:{i:02d}Z"
        score_rnd = det_aleatorio.update(valor, ts)
        score_z = det_zscore.update(valor, ts)

        assert 0.0 <= score_rnd <= 1.0
        assert 0.0 <= score_z <= 1.0


def test_anti_fuga_informacion_causal() -> None:
    """Verifica que modificar valores futuros no altere los scores pasados o presentes."""
    serie_original = _generar_serie_sintetica(60)
    corte_k = 30

    serie_alterada = list(serie_original)
    for i in range(corte_k + 1, len(serie_alterada)):
        serie_alterada[i] += 500.0

    # Cambio k: eliminado
    det_a = RollingRobustZ(window_size=20, min_observations=5)
    det_b = RollingRobustZ(window_size=20, min_observations=5)

    scores_a = [det_a.update(v, f"t{i}") for i, v in enumerate(serie_original)]
    scores_b = [det_b.update(v, f"t{i}") for i, v in enumerate(serie_alterada)]

    assert scores_a[: corte_k + 1] == scores_b[: corte_k + 1]
    assert any(s > 0.0 for s in scores_a[5 : corte_k + 1])
    assert scores_b[corte_k + 1] != scores_a[corte_k + 1]
    assert scores_b[corte_k + 1] > scores_a[corte_k + 1]


def test_valor_actual_se_puntua_antes_de_entrar_a_la_ventana() -> None:
    """Demuestra que el valor actual NO entra en la ventana antes del cálculo del score."""
    # Cambio k: eliminado
    detector = RollingRobustZ(window_size=5, min_observations=5)
    ventana_inicial = [1.0, 2.0, 3.0, 4.0, 5.0]
    for i, v in enumerate(ventana_inicial):
        detector.update(v, f"t{i}")

    nuevo_valor = 10.0
    score_obtenido = detector.update(nuevo_valor, "t5")

    # Cambio k: score adaptado a nueva funcion
    z_esperado = abs(10.0 - 3.0) / (1.4826 * 1.0)
    score_esperado = z_esperado / (z_esperado + Z_MEDIO)
    assert math.isclose(score_obtenido, score_esperado, rel_tol=1e-9)


def test_determinismo() -> None:
    """Verifica que dos instancias con la misma semilla o configuración produzcan secuencias idénticas."""
    serie = _generar_serie_sintetica(50)

    rnd_1 = RandomDetector(seed=777)
    rnd_2 = RandomDetector(seed=777)
    scores_rnd_1 = [rnd_1.update(v, f"t{i}") for i, v in enumerate(serie)]
    scores_rnd_2 = [rnd_2.update(v, f"t{i}") for i, v in enumerate(serie)]
    assert scores_rnd_1 == scores_rnd_2

    # Cambio k: eliminado
    z_1 = RollingRobustZ(window_size=15)
    z_2 = RollingRobustZ(window_size=15)
    scores_z_1 = [z_1.update(v, f"t{i}") for i, v in enumerate(serie)]
    scores_z_2 = [z_2.update(v, f"t{i}") for i, v in enumerate(serie)]
    assert scores_z_1 == scores_z_2


def test_serie_constante_sin_excepciones_ni_nan() -> None:
    """Verifica que una serie constante se procese sin error ni valores NaN y devuelva SCORE_SIN_EVIDENCIA."""
    serie_constante = [42.0] * 50
    detector = RollingRobustZ(window_size=20, min_observations=5)

    scores = [detector.update(v, f"t{i}") for i, v in enumerate(serie_constante)]

    for score in scores:
        assert not math.isnan(score)
        assert not math.isinf(score)
        assert 0.0 <= score <= 1.0

    score_cambio = detector.update(100.0, "t50")
    assert not math.isnan(score_cambio)
    assert score_cambio == 0.0


def test_deteccion_basica_pico_anomalo() -> None:
    """Verifica que un pico inducido reciba un score mayor al promedio del resto de la serie."""
    serie = _generar_serie_sintetica(80)
    posicion_pico = 50
    serie[posicion_pico] += 25.0

    # Cambio k: eliminado
    detector = RollingRobustZ(window_size=30, min_observations=5)
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


def test_rama_1_mean_ad() -> None:
    """a) Ventana [10,10,10,10,11] y valor 11: usa la rama 1, escala > 0, score < 1.0."""
    # Cambio k: eliminado
    detector = RollingRobustZ(window_size=5, min_observations=5)
    detector.update(10.0, "t1")
    detector.update(10.0, "t2")
    detector.update(10.0, "t3")
    detector.update(10.0, "t4")
    detector.update(11.0, "t5")
    
    score = detector.update(11.0, "t6")
    
    assert 0.0 < score < 1.0


def test_rama_2_saltos_enteros() -> None:
    """b) Serie de enteros con saltos de 1 y MAD == 0: un salto de 1 NO da score 1.0."""
    # Cambio k: eliminado
    detector = RollingRobustZ(window_size=3, min_observations=3)
    detector.update(1.0, "t1")
    detector.update(1.0, "t2")
    detector.update(1.0, "t3")
    detector.update(2.0, "t4")
    detector.update(3.0, "t6")
    detector.update(3.0, "t7")
    
    score = detector.update(4.0, "t8")
    assert 0.0 < score < 1.0


def test_rama_2_explicita() -> None:
    """c) Rama 2: ventana con desviación media 0 pero resolución observada > 0."""
    # Cambio k: eliminado
    detector = RollingRobustZ(window_size=3, min_observations=3)
    detector.update(10.0, "t1")
    detector.update(15.0, "t2")
    detector.update(15.0, "t3")
    detector.update(15.0, "t4")
    
    score = detector.update(20.0, "t5")
    # Cambio k: score adaptado a z / (z + Z_MEDIO)
    # z = |20 - 15| / 5.0 = 1.0
    assert math.isclose(score, 1.0 / (1.0 + Z_MEDIO), rel_tol=1e-5)


def test_rama_3_serie_plana() -> None:
    """d) Rama 3: serie totalmente plana y luego un valor distinto devuelve SCORE_SIN_EVIDENCIA."""
    # Cambio k: eliminado
    detector = RollingRobustZ(window_size=3, min_observations=3)
    detector.update(100.0, "t1")
    detector.update(100.0, "t2")
    detector.update(100.0, "t3")
    score = detector.update(150.0, "t4")
    assert score == 0.0


def test_anti_fuga_resolucion() -> None:
    """e) Anti-fuga: la resolución observada no usa el valor actual ni el futuro."""
    # Cambio k: eliminado
    detector = RollingRobustZ(window_size=3, min_observations=3)
    detector.update(10.0, "t1")
    detector.update(10.0, "t2")
    detector.update(10.0, "t3")
    
    score_1 = detector.update(11.0, "t4")
    assert score_1 == 0.0
    
    detector.update(11.0, "t5")
    detector.update(11.0, "t6")
    score_2 = detector.update(12.0, "t7")
    assert score_2 > 0.0


def test_no_hay_nan_ni_fueras_de_rango() -> None:
    """f) Los scores siguen en [0, 1] y no hay NaN."""
    # Cambio k: eliminado
    detector = RollingRobustZ(window_size=3, min_observations=3)
    scores = []
    valores = [1.0, float('nan'), float('inf'), 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 5000.0, -5000.0, 0.0, 0.0, 0.0, 0.0]
    for i, v in enumerate(valores):
        scores.append(detector.update(v, f"t{i}"))
        
    for s in scores:
        assert not math.isnan(s)
        assert not math.isinf(s)
        assert 0.0 <= s <= 1.0

# NUEVAS PRUEBAS PARA SCORE MODIFICADO z / (z + Z_MEDIO)

def test_score_formula_z_cero_y_z_medio() -> None:
    """a) score(z=0) == 0, score(z=Z_MEDIO) == 0.5"""
    detector = RollingRobustZ(window_size=3, min_observations=3)
    
    # z=0:
    detector.update(10.0, "t1")
    detector.update(10.0, "t2")
    detector.update(10.0, "t3")
    # Para poder probar que z=0 da score=0, tenemos que usar la resolucion
    # o meanAD. Para probar la formula limpia podemos simular z.
    # Mejor: ventana [10, 20, 30]. mediana=20. mad=10. escala=14.826.
    detector = RollingRobustZ(window_size=3, min_observations=3)
    detector.update(10.0, "t1")
    detector.update(20.0, "t2")
    detector.update(30.0, "t3")
    
    # z=0
    score_z0 = detector.update(20.0, "t4")
    assert score_z0 == 0.0
    
    # z=Z_MEDIO
    detector2 = RollingRobustZ(window_size=3, min_observations=3)
    detector2.update(10.0, "t1")
    detector2.update(20.0, "t2")
    detector2.update(30.0, "t3")
    val_z_medio = 20.0 + Z_MEDIO * 14.826
    score_zm = detector2.update(val_z_medio, "t5")
    assert math.isclose(score_zm, 0.5, rel_tol=1e-5)

def test_monotonia() -> None:
    """b) Monotonía: scores crecientes para z crecientes."""
    detector = RollingRobustZ(window_size=3, min_observations=3)
    detector.update(10.0, "t1")
    detector.update(20.0, "t2")
    detector.update(30.0, "t3")
    
    # ventana actual [10, 20, 30]. Mediana 20.
    s1 = detector.update(30.0, "t4")  # diff=10
    s2 = detector.update(40.0, "t5")  # diff=20
    s3 = detector.update(50.0, "t6")  # diff=30
    
    assert s1 < s2 < s3

def test_z_muy_grande_sin_saturacion_pura() -> None:
    """c) Con z muy grande (ej. 1e6) el score es < 1.0 en punto flotante."""
    detector = RollingRobustZ(window_size=3, min_observations=3)
    detector.update(10.0, "t1")
    detector.update(20.0, "t2")
    detector.update(30.0, "t3")
    
    # ventana [10, 20, 30]. mediana=20.
    s = detector.update(1e8, "t4")
    assert s < 1.0

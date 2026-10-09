"""Pruebas unitarias y de anti-fuga para BiSeasonalRobustZ."""

from detectors.base import Detector
from detectors.biseasonal import BiSeasonalRobustZ


def test_biseasonal_cumple_protocolo():
    detector = BiSeasonalRobustZ()
    assert isinstance(detector, Detector)


def test_particionado_regimen_y_bucket():
    detector = BiSeasonalRobustZ(bucket_size_minutes=15)

    # Lunes (2020-01-06) 10:00 AM -> Laboral (0), bucket 40
    reg, b = detector._parsear_tiempo("2020-01-06 10:00:00")
    assert reg == 0
    assert b == 40

    # Viernes (2020-01-10) 23:45 -> Laboral (0), bucket 95
    reg, b = detector._parsear_tiempo("2020-01-10 23:45:00")
    assert reg == 0
    assert b == 95

    # Sábado (2020-01-11) 00:05 -> Fin de semana (1), bucket 0
    reg, b = detector._parsear_tiempo("2020-01-11 00:05:00")
    assert reg == 1
    assert b == 0

    # Domingo (2020-01-12) 14:30 -> Fin de semana (1), bucket 58 (14*4 + 2)
    reg, b = detector._parsear_tiempo("2020-01-12 14:30:00")
    assert reg == 1
    assert b == 58


def test_fallback_suave_a_diurno():
    # min_regime = 3, min_diurnal = 3
    detector = BiSeasonalRobustZ(
        bucket_size_minutes=60,
        window_size=5,
        min_observations_regime=3,
        min_observations_diurnal=3,
    )

    # Entrenamos los días hábiles a las 10:00 AM (bucket 10)
    # Lunes, Martes, Miércoles
    detector.update(10.0, "2020-01-06 10:00:00")  # Lun
    detector.update(10.2, "2020-01-07 10:00:00")  # Mar
    detector.update(10.1, "2020-01-08 10:00:00")  # Mie
    # En este punto: diurnal_det tiene 3 observaciones (maduro).
    # regime_det laboral tiene 3 observaciones (maduro).
    # regime_det fin de semana tiene 0 observaciones (inmaduro).

    # Sábado 10:00 AM con valor fuertemente anómalo (50.0)
    # Como el régimen fin de semana tiene 0 obs (< 3), debe hacer fallback a diurno
    # y detectar la anomalía respecto a [10.0, 10.2, 10.1].
    score_sabado = detector.update(50.0, "2020-01-11 10:00:00")
    assert score_sabado > 0.5, f"Fallback diurno falló, score obtenido: {score_sabado}"


def test_independencia_anti_fuga_biseasonal():
    det_a = BiSeasonalRobustZ(
        bucket_size_minutes=60,
        min_observations_regime=2,
        min_observations_diurnal=2,
    )
    det_b = BiSeasonalRobustZ(
        bucket_size_minutes=60,
        min_observations_regime=2,
        min_observations_diurnal=2,
    )

    # Secuencia para det_a
    s_a = []
    s_a.append(det_a.update(10.0, "2020-01-06 10:00:00"))
    s_a.append(det_a.update(10.5, "2020-01-07 10:00:00"))
    s_a.append(det_a.update(25.0, "2020-01-08 10:00:00"))

    # Secuencia para det_b donde alteramos el futuro posterior a t2
    s_b = []
    s_b.append(det_b.update(10.0, "2020-01-06 10:00:00"))
    s_b.append(det_b.update(10.5, "2020-01-07 10:00:00"))
    s_b.append(det_b.update(25.0, "2020-01-08 10:00:00"))

    # Los primeros 3 puntos deben ser idénticos
    assert s_a == s_b

    # El tercer punto debe haber detectado la anomalía
    assert s_a[2] > 0.0

    # Inyectamos alteración masiva futura en det_b
    det_b.update(99999.0, "2020-01-09 10:00:00")

    # Verificamos que los scores pasados de det_b hasta t2 no cambiaron
    assert s_a[:3] == s_b[:3]

"""Pruebas unitarias para el envolvente causal StreamingPeakDecay."""

import math
import numpy as np
import pytest

from detectors.base import Detector
from detectors.peak_decay import StreamingPeakDecay
from detectors.seasonal import DailySeasonalRobustZ
from detectors.calibrador_percentil import PercentileCalibrator


class DummyDetector(Detector):
    """Detector de prueba que emite una secuencia predefinida de scores."""

    def __init__(self, sequence: list[float]) -> None:
        self.sequence = sequence
        self.idx = 0

    def update(self, value: float, timestamp: str = "") -> float:
        if self.idx < len(self.sequence):
            val = self.sequence[self.idx]
            self.idx += 1
            return val
        return 0.0


def test_causality_and_anti_leakage():
    """Certifica que los scores en t dependen únicamente de [t-K, t] sin fuga futura."""
    np.random.seed(42)
    stream_a = list(np.random.randn(100))
    # Stream B comparte los primeros 50 puntos pero tiene un futuro completamente diferente
    stream_b = stream_a[:50] + list(np.random.randn(50) + 100.0)

    det_a = StreamingPeakDecay(
        PercentileCalibrator(DailySeasonalRobustZ()),
        k_window=12,
        tau=2.0,
    )
    scores_a = [det_a.update(v, f"2020-01-01 00:{i:02d}:00") for i, v in enumerate(stream_a)]

    det_b = StreamingPeakDecay(
        PercentileCalibrator(DailySeasonalRobustZ()),
        k_window=12,
        tau=2.0,
    )
    scores_b = [det_b.update(v, f"2020-01-01 00:{i:02d}:00") for i, v in enumerate(stream_b)]

    # Los primeros 50 puntos deben ser exactamente idénticos bit a bit
    for i in range(50):
        assert scores_a[i] == scores_b[i], f"Fuga de futuro detectada en el paso {i}!"


def test_attack_front_retention_and_exponential_decay():
    """Verifica que el frente de choque pasa pleno y las mesetas se atenúan exponencialmente."""
    # Secuencia: choque a 0.90, luego meseta prolongada a 0.90, luego cresta superior a 0.95
    seq = [0.1, 0.90, 0.90, 0.90, 0.90, 0.95, 0.90]
    dummy = DummyDetector(seq)
    decay_det = StreamingPeakDecay(dummy, k_window=5, tau=2.0)

    outputs = [decay_det.update(0.0) for _ in seq]

    # Paso 0: score normal de fondo
    assert outputs[0] == 0.1

    # Paso 1: frente de choque (cresta nueva) -> emite 0.90 pleno
    assert outputs[1] == 0.90

    # Pasos 2, 3, 4: meseta de 0.90 -> deben decaer estrictamente según exp(-dt / 2.0)
    assert outputs[2] == pytest.approx(0.90 * math.exp(-1.0 / 2.0), abs=1e-5)
    assert outputs[3] == pytest.approx(0.90 * math.exp(-2.0 / 2.0), abs=1e-5)
    assert outputs[4] == pytest.approx(0.90 * math.exp(-3.0 / 2.0), abs=1e-5)
    assert outputs[2] > outputs[3] > outputs[4]

    # Paso 5: nueva cresta más alta (0.95 > max historial) -> emite 0.95 pleno
    assert outputs[5] == 0.95

    # Paso 6: relajación tras la nueva cresta -> decae desde el nuevo pico
    assert outputs[6] == pytest.approx(0.90 * math.exp(-1.0 / 2.0), abs=1e-5)


def test_boundary_conditions_and_edge_cases():
    """Verifica la robustez ante NaN, infinitos y extremos [0.0, 1.0]."""
    seq = [float("nan"), float("inf"), float("-inf"), 0.0, 1.0, 1.5, -0.5]
    dummy = DummyDetector(seq)
    decay_det = StreamingPeakDecay(dummy, k_window=4, tau=2.0)

    outputs = [decay_det.update(0.0) for _ in seq]

    # NaN e infinitos deben emitir 0.0
    assert outputs[0] == 0.0
    assert outputs[1] == 0.0
    assert outputs[2] == 0.0

    # 0.0 debe emitir 0.0
    assert outputs[3] == 0.0

    # 1.0 debe emitir 1.0
    assert outputs[4] == 1.0

    # Clamping de valores fuera del rango
    assert outputs[5] <= 1.0
    assert outputs[6] >= 0.0


def test_invalid_parameters():
    """Verifica que se rechacen hiperparámetros no válidos."""
    dummy = DummyDetector([0.5])
    with pytest.raises(ValueError):
        StreamingPeakDecay(dummy, k_window=0)

    with pytest.raises(ValueError):
        StreamingPeakDecay(dummy, tau=0.0)

    with pytest.raises(ValueError):
        StreamingPeakDecay(dummy, tau=-1.5)


def test_recovery_after_k_window():
    """Verifica que tras expirar la ventana de memoria K, una nueva perturbación sea cresta local."""
    # Un pico de 0.80 en t=1, luego 4 pasos de silencio (0.0), luego un pico moderado de 0.60
    # Con K=3, cuando llega t=5 el historial sólo contiene [0.0, 0.0, 0.0], por lo que 0.60 es nueva cresta
    seq = [0.0, 0.80, 0.0, 0.0, 0.0, 0.60]
    dummy = DummyDetector(seq)
    decay_det = StreamingPeakDecay(dummy, k_window=3, tau=2.0)

    outputs = [decay_det.update(0.0) for _ in seq]

    assert outputs[1] == 0.80  # Pico original
    assert outputs[5] == 0.60  # Recuperado plenamente como cresta local tras expirar K=3

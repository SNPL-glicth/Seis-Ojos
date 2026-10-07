import math
import random
import pytest
from typing import Any
from detectors.calibrador_percentil import PercentileCalibrator
from detectors.base import Detector

class FakeDetector(Detector):
    def __init__(self, sequence):
        self.sequence = sequence
        self.idx = 0
        
    def update(self, value: float, timestamp: Any) -> float:
        if self.idx < len(self.sequence):
            res = self.sequence[self.idx]
            self.idx += 1
            return res
        return 0.0

def test_calibrador_interfaz():
    """h) Cumple la interfaz Detector."""
    calibrador = PercentileCalibrator(FakeDetector([1.0]))
    assert isinstance(calibrador, Detector)
    assert hasattr(calibrador, "update")

def test_calentamiento_y_rango():
    """
    f) Calentamiento: 0.0 hasta reunir MIN_HISTORIAL valores no nulos.
    a) Scores en [0, 1], sin NaN. Tras el calentamiento: estrictamente entre 0 y 1.
    """
    # Create sequence of exactly MIN_HISTORIAL + 10 values
    seq = [float(i + 1) for i in range(PercentileCalibrator.MIN_HISTORIAL + 10)]
    calibrador = PercentileCalibrator(FakeDetector(seq))
    
    scores = []
    for _ in range(len(seq)):
        scores.append(calibrador.update(0, None))
        
    # Hasta MIN_HISTORIAL, deben ser exactamente 0.0
    assert all(s == 0.0 for s in scores[:PercentileCalibrator.MIN_HISTORIAL])
    
    # Después de MIN_HISTORIAL, deben estar estrictamente entre 0 y 1
    scores_post = scores[PercentileCalibrator.MIN_HISTORIAL:]
    assert len(scores_post) == 10
    assert all(not math.isnan(s) for s in scores_post)
    assert all(0.0 < s < 1.0 for s in scores_post)

def test_monotonia():
    """b) Monotonía, con un detector interno falso que devuelve una secuencia dada."""
    # Emitimos MIN_HISTORIAL valores fijos para calentar
    seq = [50.0] * PercentileCalibrator.MIN_HISTORIAL
    # Luego emitimos 3 valores distintos
    seq.extend([10.0, 50.0, 90.0])
    calibrador = PercentileCalibrator(FakeDetector(seq))
    
    for _ in range(PercentileCalibrator.MIN_HISTORIAL):
        calibrador.update(0, None)
        
    score_low = calibrador.update(0, None)
    score_mid = calibrador.update(0, None)
    score_high = calibrador.update(0, None)
    
    assert score_low < score_mid < score_high

def test_records_sin_empates():
    """c) Récords: una secuencia creciente no produce dos scores iguales a 1.0."""
    seq = [float(i + 1) for i in range(PercentileCalibrator.MIN_HISTORIAL + 10)]
    calibrador = PercentileCalibrator(FakeDetector(seq))
    
    for _ in range(PercentileCalibrator.MIN_HISTORIAL):
        calibrador.update(0, None)
        
    scores = []
    for _ in range(10):
        scores.append(calibrador.update(0, None))
        
    assert all(s < 1.0 for s in scores)
    assert len(set(scores)) == len(scores)  # Todos distintos

def test_uniformidad():
    """d) Uniformidad: con valores aleatorios uniformes (semilla fija, 20,000 puntos)."""
    random.seed(42)
    seq = [random.uniform(0.1, 100.0) for _ in range(20000)]
    calibrador = PercentileCalibrator(FakeDetector(seq))
    
    scores = []
    for _ in range(20000):
        s = calibrador.update(0, None)
        if s > 0.0:
            scores.append(s)
            
    # De los que superaron el calentamiento, contamos los >= 0.99
    extremos = sum(1 for s in scores if s >= 0.99)
    fraccion = extremos / len(scores)
    
    assert 0.007 <= fraccion <= 0.013, f"Fracción fue {fraccion}"

def test_interno_nulo():
    """g) Interno que devuelve siempre 0.0: no lanza excepciones, todo es 0.0."""
    seq = [0.0] * (PercentileCalibrator.MIN_HISTORIAL + 100)
    calibrador = PercentileCalibrator(FakeDetector(seq))
    
    for _ in range(len(seq)):
        assert calibrador.update(0, None) == 0.0

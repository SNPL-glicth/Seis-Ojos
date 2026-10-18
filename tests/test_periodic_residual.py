import math
import numpy as np
import pytest

from sixeyes.experimental.periodic_residual import PeriodicResidualDetector, MaxFusion
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.base import Detector


def test_periodic_detector_implements_protocol():
    assert isinstance(PeriodicResidualDetector(), Detector)
    assert isinstance(MaxFusion([PeriodicResidualDetector()]), Detector)


def test_aperiodic_noise_stays_gated():
    """Ruido blanco aleatorio no tiene estructura periódica (R² < 0.25). Debe devolver 0.0."""
    rng = np.random.default_rng(42)
    det = PeriodicResidualDetector(periods=(48, 288), gate_min=0.25)
    
    scores = []
    for _ in range(1000):
        val = rng.standard_normal()
        scores.append(det.update(val))
        
    # En ruido blanco, casi el 100% de los puntos deben quedar en 0.0 por compuerta cerrada
    zero_frac = np.mean(np.array(scores) == 0.0)
    assert zero_frac > 0.95


def test_periodic_signal_opens_gate_and_detects_phase_anomaly():
    """Onda periódica P=48: tras el calentamiento, debe detectar una anomalía que no rompa el rango global."""
    det = PeriodicResidualDetector(periods=(48,), min_visits=3, gate_min=0.25)
    
    P = 48
    # Rango global será [-10, 10]
    # Inyectaremos una anomalía a mitad de ciclo donde normalmente vale -10 pero le ponemos +5
    # (no rompe el máximo global de +10, por lo que SpatialSpikeBypass no la vería).
    
    scores = []
    # 5 ciclos limpios (240 puntos)
    for t in range(240):
        val = 10.0 * math.sin(2 * math.pi * t / P)
        scores.append(det.update(val))
        
    # La compuerta ya debería estar abierta y el residuo normal ser muy bajo
    score_normal = det.update(10.0 * math.sin(2 * math.pi * 240 / P))
    
    # Inyectamos en t=241 una perturbación de fase
    # Esperado: sin(2pi*1/48) ~ 1.30, inyectamos 8.0
    val_anom = 8.0
    score_anom = det.update(val_anom)
    
    assert score_anom > 20.0  # El residuo al cuadrado estandarizado debe dispararse
    assert score_anom > score_normal * 10


def test_random_walk_gate_stays_closed():
    """Un paseo aleatorio sin periodicidad no debe engañar a la compuerta."""
    rng = np.random.default_rng(42)
    rw = np.cumsum(rng.standard_normal(4000))
    det = PeriodicResidualDetector()
    scores = [det.update(v) for v in rw]
    zero_frac = np.mean(np.array(scores) == 0.0)
    assert zero_frac > 0.95


def test_step_changes_gate_stays_closed():
    """Saltos de nivel bruscos no deben abrir la compuerta si no hay periodicidad."""
    rng = np.random.default_rng(42)
    steps = np.zeros(4000)
    steps[1000:2000] = 50.0
    steps[3000:] = -30.0
    steps += rng.standard_normal(4000)
    det = PeriodicResidualDetector()
    scores = [det.update(v) for v in steps]
    zero_frac = np.mean(np.array(scores) == 0.0)
    assert zero_frac > 0.95


def test_calibrator_ignores_zeros_from_gated_detector():
    """Verificar que el PercentileCalibrator no ingiere ceros de la compuerta cerrada en su historial."""
    det = PeriodicResidualDetector(periods=(48,), gate_min=0.25)
    cal = PercentileCalibrator(det, max_history=1000)
    
    # 350 puntos de ceros (compuerta cerrada)
    for _ in range(350):
        s = cal.update(0.0)
        assert s == 0.0
        
    # El historial del calibrador debe estar vacío porque raw == 0.0 se salta antes de bisect.insort
    assert len(cal._historial) == 0

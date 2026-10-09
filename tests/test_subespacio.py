import numpy as np
import pytest
from src.detectors.subespacio import SubspaceResidualDetector

def test_subspace_limits():
    det = SubspaceResidualDetector(ventana=5, buffer_max=20, min_ventanas=10, reajuste_cada=5)
    # a) Scores en [0, 1], sin NaN. d) Serie constante: todo 0.0.
    scores = []
    for i in range(50):
        scores.append(det.update(10.0, str(i)))
    
    scores = np.array(scores)
    assert not np.isnan(scores).any()
    assert (scores >= 0.0).all() and (scores < 1.0).all()
    assert (scores == 0.0).all()

def test_anti_fuga_y_determinismo():
    # b) Anti-fuga y c) Determinismo
    # Configuramos parámetros pequeños para calentar rápido
    k_calentamiento = 20
    
    # Serie original
    np.random.seed(42)
    s1 = np.random.randn(50)
    det1 = SubspaceResidualDetector(ventana=5, buffer_max=20, min_ventanas=10, reajuste_cada=5)
    scores1 = [det1.update(v, str(i)) for i, v in enumerate(s1)]
    
    # Serie alterada después del calentamiento
    s2 = s1.copy()
    s2[30:] = s2[30:] * 10
    det2 = SubspaceResidualDetector(ventana=5, buffer_max=20, min_ventanas=10, reajuste_cada=5)
    scores2 = [det2.update(v, str(i)) for i, v in enumerate(s2)]
    
    # Anti-fuga: hasta el punto de alteración, scores idénticos
    assert np.allclose(scores1[:30], scores2[:30])
    
    # Hay scores > 0 antes del k=30
    assert (np.array(scores1[:30]) > 0).any()
    
    # Determinismo
    det3 = SubspaceResidualDetector(ventana=5, buffer_max=20, min_ventanas=10, reajuste_cada=5)
    scores3 = [det3.update(v, str(i)) for i, v in enumerate(s1)]
    assert np.allclose(scores1, scores3)

def test_seno_con_pico():
    # e) Seno con pico inyectado: score del pico mayor que el promedio
    t = np.linspace(0, 4*np.pi, 200)
    s = np.sin(t)
    # inyectar pico en el paso 150
    s[150] = 5.0
    
    det = SubspaceResidualDetector(ventana=5, buffer_max=50, min_ventanas=20, reajuste_cada=10)
    scores = [det.update(v, str(i)) for i, v in enumerate(s)]
    
    pico_score = scores[150]
    promedio_sin_pico = np.mean(scores[50:149])
    
    assert pico_score > promedio_sin_pico
    assert pico_score > 0.5

def test_mecanismo_con_varianza_alta():
    # f) Un cambio de FORMA (seno que pasa a diente de sierra, misma amplitud).
    # Con el valor por defecto (0.90) este test no pasa; requiere varianza alta (0.999)
    # para forzar al SVD a capturar la curvatura del seno en pocas dimensiones.
    t = np.linspace(0, 10*np.pi, 500)
    s1 = np.sin(t[:250])
    s2 = (t[250:] % (2*np.pi)) / (2*np.pi) * 2 - 1  # Diente de sierra en [-1, 1]
    s = np.concatenate([s1, s2])
    
    det = SubspaceResidualDetector(ventana=10, buffer_max=100, min_ventanas=50, reajuste_cada=25, varianza_objetivo=0.999)
    scores = [det.update(v, str(i)) for i, v in enumerate(s)]
    
    max_seno = np.max(scores[100:200])
    max_cambio = np.max(scores[255:270])
    
    assert max_cambio > max_seno

def test_cumple_interfaz():
    # g) Cumple la interfaz
    from src.detectors.base import Detector
    det = SubspaceResidualDetector()
    assert isinstance(det, Detector)

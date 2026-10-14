"""
Pruebas unitarias para CausalSTFTDetector.
"""

import numpy as np
import pytest
import math

from sixeyes.detectors.base import Detector
from sixeyes.experimental.causal_spectral import CausalSTFTDetector


class TestCausalSTFTDetector:
    def test_protocolo_detector(self):
        """Verifica que hereda de Detector y cumple la interfaz."""
        det = CausalSTFTDetector()
        assert isinstance(det, Detector)
        score = det.update(10.0, "2026-10-12T00:00:00")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_causalidad_estricta(self):
        """Verifica causalidad: score en t no depende de futuro."""
        np.random.seed(42)
        
        # Serie base
        base = [10.0 + np.random.randn() * 0.5 for _ in range(100)]
        det1 = CausalSTFTDetector()
        scores1 = [det1.update(x) for x in base]
        
        # Serie con perturbación FUTURA (después del paso 50)
        perturbed = base.copy()
        for i in range(50, 100):
            perturbed[i] += 5.0
        
        det2 = CausalSTFTDetector()
        scores2 = [det2.update(x) for x in perturbed]
        
        # Primeros 50 scores deben ser idénticos (causalidad)
        for i in range(50):
            assert abs(scores1[i] - scores2[i]) < 1e-10, f"Fuga de futuro en paso {i}"

    def test_determinismo(self):
        """Mismas entradas -> mismos scores."""
        d1 = CausalSTFTDetector()
        d2 = CausalSTFTDetector()
        
        for x in [1.0, 2.0, 1.5, 0.5, 3.0, 2.5, 1.0]:
            s1 = d1.update(x)
            s2 = d2.update(x)
            assert abs(s1 - s2) < 1e-10

    def test_nan_inf_handling(self):
        """NaN e Inf producen 0.0."""
        det = CausalSTFTDetector()
        assert det.update(float('nan')) == 0.0
        assert det.update(float('inf')) == 0.0
        assert det.update(float('-inf')) == 0.0

    def test_compuerta_ruido_en_serie_tranquila(self):
        """En serie estable, score debe ser 0.0 (compuerta activa)."""
        det = CausalSTFTDetector(noise_gate_sigma=3.5)
        
        # 500 pasos de ruido gaussiano pequeño
        np.random.seed(42)
        for _ in range(500):
            x = 10.0 + np.random.randn() * 0.1
            s = det.update(x)
        
        # Score debe ser 0.0 en régimen estacionario
        # (compuerta de ruido filtra fluctuaciones normales)
        assert s == 0.0, f"Compuerta falló en serie tranquila: score={s}"

    def test_deteccion_cambio_frecuencia(self):
        """Debe detectar cambio abrupto de frecuencia (con latencia de ventana STFT)."""
        # Usar min_history bajo y compuerta de ruido baja para test
        det = CausalSTFTDetector(noise_gate_sigma=0.5, calibrator_min_history=100)
        
        # Fondo: seno 0.1 Hz (suficientes pasos para calibrador)
        for i in range(250):
            x = 10.0 + 2.0 * np.sin(2 * np.pi * 0.1 * i * 0.1)
            det.update(x)
        
        # Cambio: seno 0.5 Hz (frecuencia 5x mayor)
        # Ventana STFT=64 tarda ~64 pasos en purgar señal vieja
        scores = []
        for i in range(100):
            x = 10.0 + 2.0 * np.sin(2 * np.pi * 0.5 * (i + 250) * 0.1)
            s = det.update(x)
            scores.append(s)
        
        # Debe detectar el cambio (después de ~64 pasos de latencia STFT)
        assert max(scores) > 0.5, f"No detectó cambio de frecuencia: max={max(scores):.4f}"

    def test_anti_fuga_causal(self):
        """El score en t no cambia si se altera el futuro."""
        det1 = CausalSTFTDetector()
        det2 = CausalSTFTDetector()
        
        # Mismas observaciones hasta paso 80
        np.random.seed(42)
        stream = [10.0 + np.random.randn() * 0.5 for _ in range(100)]
        
        scores1 = [det1.update(x) for x in stream]
        
        # Alterar futuro en stream2
        stream2 = stream.copy()
        for i in range(80, 100):
            stream2[i] += 10.0
        
        scores2 = [det2.update(x) for x in stream2]
        
        # Primeros 80 deben ser idénticos
        for i in range(80):
            assert abs(scores1[i] - scores2[i]) < 1e-10

    def test_reset_limpia_estado(self):
        """reset() debe dejar detector en estado inicial."""
        det = CausalSTFTDetector()
        
        # Calentar
        for _ in range(200):
            det.update(10.0 + np.random.randn() * 0.5)
        
        det.reset()
        
        # Debe volver a estado inicial
        s = det.update(10.0)
        assert s == 0.0

    def test_rango_score(self):
        """Scores siempre en [0, 1]."""
        det = CausalSTFTDetector()
        for _ in range(200):
            x = 10.0 + np.random.randn() * 10.0
            s = det.update(x)
            assert 0.0 <= s <= 1.0

    def test_serie_constante_score_cero(self):
        """Serie constante -> score 0.0 tras warmup."""
        det = CausalSTFTDetector()
        for _ in range(100):
            s = det.update(100.0)
        # En serie constante, residuo espectral = 0
        for _ in range(50):
            s = det.update(100.0)
            assert s == 0.0

    def test_pico_transitorio_no_genera_falsa_alarma(self):
        """Un solo pico transitorio no debe disparar compuerta espectral."""
        det = CausalSTFTDetector(noise_gate_sigma=3.5)
        
        # Fondo estable
        for _ in range(100):
            det.update(10.0 + np.random.randn() * 0.5)
        
        # Pico único
        s = det.update(50.0)
        
        # Un solo pico no debe generar residuo espectral sostenido
        assert s == 0.0 or s < 0.1, f"Falsa alarma en pico transitorio: {s}"
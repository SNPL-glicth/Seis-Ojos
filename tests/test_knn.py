"""Tests k-NN streaming sobre retardos."""

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.experimental.knn_detector import StreamingKNNDetector


class TestStreamingKNN:
    def test_protocolo(self):
        det = StreamingKNNDetector()
        assert isinstance(det, Detector)
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0  # warmup

    def test_motivo_nuevo_puntua_alto(self):
        det = StreamingKNNDetector(dimension=8, buffer_size=200, k=3)
        base = [50.0 + np.sin(i / 5.0) for i in range(300)]
        for v in base:
            det.update(float(v), "2024-01-01T00:00:00")
        s_comun = det.update(float(50.0 + np.sin(300 / 5.0)), "2024-01-01T01:00:00")
        s_raro = det.update(500.0, "2024-01-01T01:01:00")
        assert s_raro > s_comun, f"raro={s_raro:.2f} comun={s_comun:.2f}"

    def test_sin_automatch(self):
        # Repetir el último vector no debe dar distancia 0 (consulta previa
        # a la ingesta).
        det = StreamingKNNDetector(dimension=8, buffer_size=200, k=3)
        for i in range(100):
            det.update(float(i % 10), "2024-01-01T00:00:00")
        assert det.update(9.0, "2024-01-01T01:00:00") > 0.0

    def test_buffer_acotado(self):
        det = StreamingKNNDetector(dimension=4, buffer_size=50, k=2)
        for i in range(300):
            det.update(float(i % 7), "2024-01-01T00:00:00")
        assert len(det._buf) <= 50

    def test_reset_replay_invariance(self):
        vals = [float(50 + (i % 11)) for i in range(120)]
        d = StreamingKNNDetector(dimension=8, buffer_size=200, k=3)
        s1 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        d.reset()
        s2 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2

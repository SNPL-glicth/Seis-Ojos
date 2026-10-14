"""Tests predictor Holt + likelihood de error."""

from sixeyes.detectors.base import Detector
from sixeyes.experimental.predictor import HoltPredictor, ErrorLikelihood, PredictiveDetector


class TestHolt:
    def test_pronostica_antes_de_observar(self):
        h = HoltPredictor()
        assert h.forecast() == 0.0
        h.observe(10.0)
        assert h.forecast() == 10.0
        h.observe(12.0)  # nivel sube, tendencia positiva
        assert h.forecast() > 12.0 - 3.0

    def test_sigue_rampa(self):
        h = HoltPredictor()
        for i in range(50):
            h.observe(100.0 + 2.0 * i)
        assert abs(h.forecast() - 202.0) < 5.0


class TestLikelihood:
    def test_ruptura_sostenida_puntua_alto(self):
        lik = ErrorLikelihood()
        for _ in range(500):
            lik.update(0.0, "2024-01-01T00:00:00")
        scores = [lik.update(5.0, "2024-01-01T01:00:00") for _ in range(10)]
        assert scores[-1] > 0.9, f"sin likelihood: {scores[-1]}"

    def test_ruido_estacionario_puntua_bajo(self):
        import numpy as np
        rng = np.random.RandomState(11)
        lik = ErrorLikelihood()
        scores = [lik.update(float(rng.randn()), "2024-01-01T00:00:00") for _ in range(1000)]
        assert sum(s > 0.9 for s in scores[500:]) < 25


class TestPredictive:
    def test_protocolo(self):
        det = PredictiveDetector()
        assert isinstance(det, Detector)
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0

    def test_cambio_de_nivel_reacciona_en_pocos_puntos(self):
        det = PredictiveDetector()
        for _ in range(500):
            det.update(50.0, "2024-01-01T00:00:00")
        scores = [det.update(80.0, f"2024-01-01T01:{i:02d}:00") for i in range(10)]
        assert max(scores[:3]) > 0.5, f"tardío: {scores[:3]}"

    def test_flatline_tras_cambio_puntua(self):
        # Patrón invisible a la densidad: valor "normal" en contexto imposible.
        import numpy as np
        rng = np.random.RandomState(5)
        det = PredictiveDetector()
        for _ in range(500):
            det.update(float(50.0 + rng.randn()), "2024-01-01T00:00:00")
        flat = [det.update(50.0, f"2024-01-01T01:{i:02d}:00") for i in range(30)]
        assert max(flat) > 0.5, f"flatline invisible: {max(flat):.3f}"

    def test_reset_replay_invariance(self):
        vals = [50.0 + (i % 7) for i in range(100)]
        d = PredictiveDetector()
        s1 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        d.reset()
        s2 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2

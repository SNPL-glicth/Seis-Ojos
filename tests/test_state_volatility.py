"""Tests para StateVolatilityDetector (Fase 16: latent state volatility)."""

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.experimental.state_volatility import StateVolatilityDetector


class TestStateVolatility:
    def test_protocolo_detector(self):
        """Implementa la interfaz Detector con update/reset."""
        det = StateVolatilityDetector()
        assert isinstance(det, Detector)
        # Primeros puntos: warmup del embedding → 0.0.
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0

    def test_determinismo_semilla_fija(self):
        """Dos instancias con la misma semilla producen scores idénticos."""
        a, b = StateVolatilityDetector(), StateVolatilityDetector()
        vals = list(np.sin(np.arange(200) / 5.0) * 10.0 + 50.0)
        sa = [a.update(float(v), "2024-01-01T00:00:00") for v in vals]
        sb = [b.update(float(v), "2024-01-01T00:00:00") for v in vals]
        assert sa == sb

    def test_scores_no_negativos(self):
        """La norma euclídea es ≥0 siempre."""
        det = StateVolatilityDetector()
        vals = list(np.random.RandomState(7).randn(300) * 5 + 50)
        scores = [det.update(float(v), "2024-01-01T00:00:00") for v in vals]
        assert all(s >= 0.0 for s in scores)

    def test_ruptura_dispara_pico(self):
        """Un salto brusco en la entrada causa pico en Δh."""
        det = StateVolatilityDetector()
        # 800 puntos de sinusoide estable.
        vals = [float(50.0 + 10.0 * np.sin(i / 5.0)) for i in range(800)]
        scores = [det.update(v, "2024-01-01T00:00:00") for v in vals]
        tail = float(np.median(scores[-50:]))
        # Ruptura masiva.
        s = det.update(500.0, "2024-01-01T01:00:00")
        assert s > tail * 2, f"ruptura={s:.4f} cola={tail:.4f}"

    def test_causalidad_estricta(self):
        """Divergencia futura no afecta scores pasados."""
        rng = np.random.RandomState(21)
        a = list(rng.randn(80))
        b = a[:40] + list(rng.randn(40) + 50.0)
        da, db = StateVolatilityDetector(), StateVolatilityDetector()
        sa = [da.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(a)]
        sb = [db.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(b)]
        assert sa[:40] == sb[:40]

    def test_reset_replay_invariance(self):
        """reset() + replay == replay desde cero."""
        vals = [float(50 + (i % 9)) for i in range(120)]
        d = StateVolatilityDetector()
        s1 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        d.reset()
        s2 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2

    def test_nan_inf_producen_cero(self):
        """NaN e Inf no propagan."""
        det = StateVolatilityDetector()
        assert det.update(float("nan")) == 0.0
        assert det.update(float("inf")) == 0.0
        assert det.update(float("-inf")) == 0.0

    def test_scores_finitos(self):
        """Ningún score es NaN o Inf en secuencia larga."""
        det = StateVolatilityDetector()
        rng = np.random.RandomState(0)
        vals = list(rng.randn(500) * 100 + 50)
        scores = [det.update(float(v), "2024-01-01T00:00:00") for v in vals]
        assert all(s == s and s != float("inf") for s in scores)

    def test_semilla_distinta_da_resultados_distintos(self):
        """Semillas distintas → reservorios distintos → scores distintos."""
        a = StateVolatilityDetector(seed=42)
        b = StateVolatilityDetector(seed=99)
        vals = [float(50 + (i % 7)) for i in range(200)]
        sa = [a.update(v, "2024-01-01T00:00:00") for v in vals]
        sb = [b.update(v, "2024-01-01T00:00:00") for v in vals]
        assert sa != sb

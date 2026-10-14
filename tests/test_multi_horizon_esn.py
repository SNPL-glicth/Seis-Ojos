"""Tests para MultiHorizonESN (Fase 16b: multi-horizon predictor)."""

import numpy as np
import pytest

from sixeyes.detectors.base import Detector
from sixeyes.experimental.multi_horizon_esn import MultiHorizonESN


class TestMultiHorizonESN:
    def test_protocolo_detector(self):
        """Implementa la interfaz Detector."""
        det = MultiHorizonESN()
        assert isinstance(det, Detector)
        # Primeros puntos: warmup embedding → 0.0
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0

    def test_determinismo_semilla_fija(self):
        """Dos instancias misma semilla → scores idénticos."""
        a, b = MultiHorizonESN(), MultiHorizonESN()
        vals = list(np.sin(np.arange(200) / 5.0) * 10.0 + 50.0)
        sa = [a.update(float(v), "2024-01-01T00:00:00") for v in vals]
        sb = [b.update(float(v), "2024-01-01T00:00:00") for v in vals]
        assert sa == sb

    def test_scores_no_negativos(self):
        """La norma euclídea es ≥ 0 siempre."""
        det = MultiHorizonESN()
        rng = np.random.RandomState(7)
        vals = list(rng.randn(300) * 5 + 50)
        scores = [det.update(float(v), "2024-01-01T00:00:00") for v in vals]
        assert all(s >= 0.0 for s in scores)

    def test_ruptura_dispara_pico(self):
        """Un salto brusco causa pico en la norma de error."""
        det = MultiHorizonESN()
        vals = [float(50.0 + 10.0 * np.sin(i / 5.0)) for i in range(800)]
        scores = [det.update(v, "2024-01-01T00:00:00") for v in vals]
        tail = float(np.median(scores[-50:]))
        s = det.update(500.0, "2024-01-01T01:00:00")
        assert s > tail * 2, f"ruptura={s:.4f} cola={tail:.4f}"

    def test_causalidad_estricta(self):
        """Divergencia futura no afecta scores pasados."""
        rng = np.random.RandomState(21)
        a = list(rng.randn(80))
        b = a[:40] + list(rng.randn(40) + 50.0)
        da, db = MultiHorizonESN(), MultiHorizonESN()
        sa = [da.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(a)]
        sb = [db.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(b)]
        assert sa[:40] == sb[:40]

    def test_reset_replay_invariance(self):
        """reset() + replay == replay desde cero."""
        vals = [float(50 + (i % 9)) for i in range(150)]
        d = MultiHorizonESN()
        s1 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        d.reset()
        s2 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2

    def test_nan_inf_producen_cero(self):
        """NaN e Inf no propagan."""
        det = MultiHorizonESN()
        assert det.update(float("nan")) == 0.0
        assert det.update(float("inf")) == 0.0
        assert det.update(float("-inf")) == 0.0

    def test_scores_finitos(self):
        """Ningún score es NaN o Inf en secuencia larga."""
        det = MultiHorizonESN()
        rng = np.random.RandomState(0)
        vals = list(rng.randn(500) * 100 + 50)
        scores = [det.update(float(v), "2024-01-01T00:00:00") for v in vals]
        assert all(s == s and s != float("inf") for s in scores)

    def test_horizontes_customizables(self):
        """Horizontes distintos generan scores distintos."""
        a = MultiHorizonESN(horizons=(1, 3, 6))
        b = MultiHorizonESN(horizons=(1, 5, 10))
        vals = [float(50 + (i % 7)) for i in range(200)]
        sa = [a.update(v, "2024-01-01T00:00:00") for v in vals]
        sb = [b.update(v, "2024-01-01T00:00:00") for v in vals]
        assert sa != sb

    def test_warmup_multi_horizonte(self):
        """Antes de max_horizon puntos, algunos errores son 0 (buffer no lleno)."""
        det = MultiHorizonESN(horizons=(1, 3, 6))
        # Los primeros 6 pasos después del warmup del embedding (16 pts)
        # no tienen todos los horizontes disponibles — el score debe ser ≥ 0
        vals = [float(50 + i) for i in range(30)]
        scores = [det.update(v, "2024-01-01T00:00:00") for v in vals]
        assert all(s >= 0.0 for s in scores)

"""Tests para BOCPDDetector (Paradigma II: cambio de distribución)."""

import math
import numpy as np
import pytest

from sixeyes.detectors.base import Detector
from sixeyes.experimental.bocpd import BOCPDDetector


class TestBOCPD:
    def test_protocolo_detector(self):
        """Implementa la interfaz Detector. Score = sorpresa logarítmica ≥ 0."""
        det = BOCPDDetector()
        assert isinstance(det, Detector)
        # Primer punto: score no-negativo y finito.
        s = det.update(50.0, "2024-01-01T00:00:00")
        assert s >= 0.0 and s == s and s != float("inf")

    def test_scores_en_rango(self):
        """Todos los scores son no-negativos y finitos (sorpresa log)."""
        det = BOCPDDetector()
        rng = np.random.RandomState(7)
        vals = list(rng.randn(300) * 5 + 50)
        scores = [det.update(float(v), "2024-01-01T00:00:00") for v in vals]
        assert all(s >= 0.0 and s == s and s != float("inf") for s in scores)

    def test_changepoint_dispara_pico(self):
        """Un salto brusco de media eleva P(changepoint)."""
        det = BOCPDDetector()
        rng = np.random.RandomState(0)
        # 300 puntos de régimen A (media=50, std=1).
        for _ in range(300):
            det.update(float(rng.randn() + 50), "2024-01-01T00:00:00")
        # Score en régimen estable.
        stable = [det.update(float(rng.randn() + 50), "2024-01-01T00:00:00")
                  for _ in range(20)]
        baseline = float(np.median(stable))
        # Primer punto del régimen B (media=200, std=1).
        s_change = det.update(200.0, "2024-01-01T01:00:00")
        assert s_change > baseline + 1.0 or s_change > 3.0, (
            f"cambio={s_change:.4f} baseline={baseline:.4f}")

    def test_causalidad_estricta(self):
        """Divergencia futura no afecta scores pasados."""
        rng = np.random.RandomState(21)
        a = list(rng.randn(80) + 50)
        b = a[:40] + list(rng.randn(40) + 200.0)
        da, db = BOCPDDetector(), BOCPDDetector()
        sa = [da.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(a)]
        sb = [db.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(b)]
        assert sa[:40] == sb[:40]

    def test_reset_replay_invariance(self):
        """reset() + replay produce scores idénticos."""
        vals = [float(50 + (i % 9)) for i in range(120)]
        d = BOCPDDetector()
        s1 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        d.reset()
        s2 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2

    def test_nan_inf_producen_cero(self):
        """NaN e Inf no propagan."""
        det = BOCPDDetector()
        assert det.update(float("nan")) == 0.0
        assert det.update(float("inf")) == 0.0
        assert det.update(float("-inf")) == 0.0

    def test_scores_finitos(self):
        """Ningún score es NaN o Inf en secuencia larga."""
        det = BOCPDDetector()
        rng = np.random.RandomState(0)
        vals = list(rng.randn(500) * 100 + 50)
        scores = [det.update(float(v), "2024-01-01T00:00:00") for v in vals]
        assert all(s >= 0.0 and s == s and s != float("inf") for s in scores)

    def test_truncacion_no_explota(self):
        """Con max_run_length bajo, no hay explosión de memoria."""
        det = BOCPDDetector(max_run_length=50)
        vals = list(np.random.RandomState(1).randn(200) + 50)
        scores = [det.update(float(v), "2024-01-01T00:00:00") for v in vals]
        # El buffer nunca supera max_run_length + 1.
        assert len(det._log_probs) <= 51
        assert all(s >= 0.0 and s == s for s in scores)

    def test_cambio_varianza_detectado(self):
        """Un cambio de varianza (no de media) también eleva P(changepoint)."""
        det = BOCPDDetector()
        rng = np.random.RandomState(5)
        # 300 pts con std=0.5.
        for _ in range(300):
            det.update(float(rng.randn() * 0.5 + 50), "2024-01-01T00:00:00")
        stable = [det.update(float(rng.randn() * 0.5 + 50), "2024-01-01T00:00:00")
                  for _ in range(20)]
        baseline = float(np.mean(stable))
        # Régimen de alta varianza (std=20).
        high_var = [det.update(float(rng.randn() * 20 + 50), "2024-01-01T01:00:00")
                    for _ in range(10)]
        assert max(high_var) > baseline + 0.5, (
            f"max_cambio={max(high_var):.4f} baseline={baseline:.4f}")

"""Tests votante de densidad y consenso por conteo."""

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.experimental.density_voter import HistogramDensityVoter
from sixeyes.experimental.consensus import CountConsensus


class TestDensityVoter:
    def test_protocolo(self):
        det = HistogramDensityVoter()
        assert isinstance(det, Detector)
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0  # warmup

    def test_nivel_nuevo_vota_uno(self):
        det = HistogramDensityVoter(history=500)
        for _ in range(400):
            det.update(50.0, "2024-01-01T00:00:00")
        # tail_avg mira la cola previa: el voto llega cuando el nivel
        # persiste (retardo 1-3 puntos, como skyline).
        det.update(500.0, "2024-01-01T01:00:00")
        assert det.update(500.0, "2024-01-01T01:01:00") == 1.0

    def test_regimen_habitual_vota_cero(self):
        det = HistogramDensityVoter(history=500)
        rng = np.random.RandomState(3)
        vals = list(50.0 + rng.randn(400))
        for v in vals:
            det.update(float(v), "2024-01-01T00:00:00")
        assert det.update(50.0, "2024-01-01T01:00:00") == 0.0

    def test_reset_replay_invariance(self):
        vals = [50.0 + (i % 5) for i in range(60)]
        d = HistogramDensityVoter()
        s1 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        d.reset()
        s2 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2


class TestConsensus:
    def test_protocolo_y_rango(self):
        det = CountConsensus()
        assert isinstance(det, Detector)
        assert det.n_voters == 4
        for v in [50.0, 51.0, 49.0, 50.5]:
            s = det.update(v, "2024-01-01T00:00:00")
            assert 0.0 <= s <= 1.0

    def test_fracciones_discretas(self):
        det = CountConsensus()
        seen = set()
        for i in range(300):
            s = det.update(50.0 + (i % 9), f"2024-01-02T{i // 60:02d}:{i % 60:02d}:00")
            seen.add(round(s, 6))
        # Fracciones k/4, con atenuación ×0.5 cuando no hay confirmación.
        allowed = {0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0}
        assert seen <= allowed, f"fuera de rejilla: {sorted(seen)}"

    def test_todos_actualizan_sin_cortocircuito(self):
        det = CountConsensus()
        for _ in range(400):
            det.update(50.0, "2024-01-01T00:00:00")
        # Tras ruptura sostenida de nivel, al menos densidad+nivel votan
        # (fracción ≥0.5). La densidad necesita 2 puntos de persistencia.
        scores = [det.update(500.0, f"2024-01-01T01:{i:02d}:00") for i in range(5)]
        assert max(scores) >= 0.5, f"sin confirmación: {scores}"

    def test_reset_replay_invariance(self):
        vals = [50.0 + (i % 7) for i in range(80)]
        ts = [f"2024-01-01T00:{i % 60:02d}:00" for i in range(80)]
        d1, d2 = CountConsensus(), CountConsensus()
        s1 = [d1.update(v, t) for v, t in zip(vals, ts)]
        d2.reset()
        s2 = [d2.update(v, t) for v, t in zip(vals, ts)]
        assert s1 == s2

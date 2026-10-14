"""Tests arquitectura densa (encoder + novedad de trayectoria)."""

import time

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.dense_arch.encoder import PhaseSpaceEncoder
from sixeyes.dense_arch.trajectory import TrajectoryNovelty, DualTrajectoryNovelty


class TestEncoder:
    def test_dimension_y_determinismo(self):
        a, b = PhaseSpaceEncoder(), PhaseSpaceEncoder()
        vals = [float(i % 7) for i in range(60)]
        ra = [a.update(v, "2024-01-01T00:00:00") for v in vals]
        rb = [b.update(v, "2024-01-01T00:00:00") for v in vals]
        assert ra[-1][0] is not None and ra[-1][0].shape == (32,)
        assert np.array_equal(ra[-1][0], rb[-1][0])

    def test_geometria_preservada(self):
        # Vectores cercanos en fases → latentes cercanos (sin cajas).
        enc = PhaseSpaceEncoder()
        for v in [50.0] * 40:
            enc.update(v, "2024-01-01T00:00:00")
        z1, _ = enc.update(50.1, "2024-01-01T00:01:00")
        z2, _ = enc.update(90.0, "2024-01-01T00:02:00")
        z0, _ = enc.update(50.05, "2024-01-01T00:03:00")
        assert np.linalg.norm(z1 - z0) < np.linalg.norm(z2 - z0)


class TestTrajectory:
    def test_protocolo(self):
        det = TrajectoryNovelty()
        assert isinstance(det, Detector)
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0

    def test_novedad_estructural(self):
        det = TrajectoryNovelty()
        base = [det.update(50.0 + (i % 5), "2024-01-01T00:00:00") for i in range(500)]
        pre = float(np.median(base[-50:]))
        s = det.update(500.0, "2024-01-01T01:00:00")
        assert s > pre

    def test_estabilidad_ruido_largo(self):
        # 5000 puntos de ruido: salida finita y acotada (sin explosión).
        rng = np.random.RandomState(0)
        det = TrajectoryNovelty()
        worst = 0.0
        for _ in range(5000):
            s = det.update(float(50.0 + rng.randn()), "2024-01-01T00:00:00")
            assert np.isfinite(s)
            worst = max(worst, s)
        assert worst < 1e6

    def test_tiempo_por_paso(self):
        det = TrajectoryNovelty()
        vals = [float(50 + (i % 13)) for i in range(2000)]
        t0 = time.perf_counter()
        for v in vals:
            det.update(v, "2024-01-01T00:00:00")
        ms = 1000 * (time.perf_counter() - t0) / len(vals)
        print(f"\nTrajectoryNovelty: {ms:.3f} ms/pt")
        assert ms < 1.0

    def test_causalidad(self):
        rng = np.random.RandomState(21)
        a = list(rng.randn(80))
        b = a[:40] + list(rng.randn(40) + 50.0)
        da, db = TrajectoryNovelty(), TrajectoryNovelty()
        sa = [da.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(a)]
        sb = [db.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(b)]
        assert sa[:40] == sb[:40]

    def test_reset_replay_invariance(self):
        vals = [50.0 + (i % 7) for i in range(120)]
        d = TrajectoryNovelty()
        s1 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        d.reset()
        s2 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2


class TestDualTrajectory:
    def test_rige_el_maximo(self):
        det = DualTrajectoryNovelty()
        for i in range(300):
            det.update(50.0 + (i % 5), "2024-01-01T00:00:00")
        s = det.update(500.0, "2024-01-01T01:00:00")
        assert s > 0.0
        assert det._fast is not None and det._slow is not None

    def test_determinismo(self):
        vals = [float(50 + (i % 11)) for i in range(150)]
        a, b = DualTrajectoryNovelty(), DualTrajectoryNovelty()
        sa = [a.update(v, "2024-01-01T00:00:00") for v in vals]
        sb = [b.update(v, "2024-01-01T00:00:00") for v in vals]
        assert sa == sb

    def test_reset(self):
        det = DualTrajectoryNovelty()
        for v in [50.0] * 60:
            det.update(v, "2024-01-01T00:00:00")
        det.reset()
        assert det._fast is None and det._slow is None
        assert det.update(50.0, "2024-01-01T00:00:00") == 0.0

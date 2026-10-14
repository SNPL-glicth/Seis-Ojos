"""Tests DualScaleRobustZ."""

from sixeyes.detectors.base import Detector
from sixeyes.experimental.dual_scale_z import DualScaleRobustZ, SeasonalDualScaleRobustZ


class TestDualScale:
    def test_protocolo(self):
        det = DualScaleRobustZ()
        assert isinstance(det, Detector)
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0  # warmup

    def test_frente_de_choque_puntua_alto(self):
        det = DualScaleRobustZ()
        for i in range(100):
            det.update(50.0 + (0.1 if i % 2 else -0.1), "2024-01-01T00:00:00")
        s = det.update(200.0, "2024-01-01T01:00:00")
        assert s > 0.8, f"frente atenuado: {s}"

    def test_meseta_se_atenua_vs_escala_lenta(self):
        det = DualScaleRobustZ()
        for i in range(100):
            det.update(50.0 + (0.1 if i % 2 else -0.1), "2024-01-01T00:00:00")
        s1 = det.update(200.0, "2024-01-01T01:00:00")
        ruined = [det.update(200.0, "2024-01-01T02:00:00") for _ in range(30)]
        # La escala lenta absorbe la meseta: el punto 30 puntúa menos que el 1.
        assert ruined[-1] < s1

    def test_causalidad(self):
        import numpy as np
        rng = np.random.RandomState(7)
        a = list(rng.randn(60))
        b = a[:30] + [x + 50.0 for x in rng.randn(30)]
        da, db = DualScaleRobustZ(), DualScaleRobustZ()
        sa = [da.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(a)]
        sb = [db.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(b)]
        assert sa[:30] == sb[:30]

    def test_reset_replay_invariance(self):
        vals = [50.0 + (i % 11) for i in range(120)]
        d1 = DualScaleRobustZ()
        s1 = [d1.update(v, "2024-01-01T00:00:00") for v in vals]
        d1.reset()
        s2 = [d1.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2

    def test_bordes(self):
        det = DualScaleRobustZ()
        assert det.update(float("nan")) == 0.0
        assert det.update(float("inf")) == 0.0
        for _ in range(10):
            assert det.update(50.0, "2024-01-01T00:00:00") >= 0.0
        # Flatline perfecto: sin variación, score 0.
        assert det.update(50.0, "2024-01-01T00:00:00") == 0.0


class TestSeasonalDualScale:
    def test_estacional_supera_a_plano_en_serie_diurna(self):
        import numpy as np
        # Serie con oscilación diurna fuerte: el plano la sufre, el
        # estacional la absorbe (centro por bucket).
        ts = [f"2024-01-{(d // 288) + 1:02d}T{(d % 288) * 5 // 60:02d}:{(d % 288) * 5 % 60:02d}:00"
              for d in range(2000)]
        vals = [50.0 + 20.0 * np.sin(2 * np.pi * (i % 288) / 288) for i in range(2000)]
        plano = DualScaleRobustZ()
        est = SeasonalDualScaleRobustZ()
        sp = np.mean([plano.update(v, t) for v, t in zip(vals, ts)][500:])
        se = np.mean([est.update(v, t) for v, t in zip(vals, ts)][500:])
        assert se < sp, f"estacional={se:.4f} plano={sp:.4f}"

    def test_reset_replay_invariance(self):
        vals = [50.0 + (i % 7) for i in range(120)]
        ts = [f"2024-01-01T00:{i % 60:02d}:00" for i in range(120)]
        d = SeasonalDualScaleRobustZ()
        s1 = [d.update(v, t) for v, t in zip(vals, ts)]
        d.reset()
        s2 = [d.update(v, t) for v, t in zip(vals, ts)]
        assert s1 == s2

    def test_fallback_sin_timestamp(self):
        d = SeasonalDualScaleRobustZ()
        for _ in range(30):
            s = d.update(50.0, "")
        assert 0.0 <= s <= 1.0

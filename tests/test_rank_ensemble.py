"""Tests ensemble por rangos (réplicas ESN)."""

from sixeyes.detectors.base import Detector
from sixeyes.experimental.rank_ensemble import RankEnsembleESN


class TestRankEnsemble:
    def test_protocolo(self):
        det = RankEnsembleESN()
        assert isinstance(det, Detector)
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0  # warmup

    def test_todos_actualizan_en_warmup(self):
        # Ningún miembro queda congelado: tras el warmup, estados vivos.
        det = RankEnsembleESN()
        for _ in range(300):
            det.update(50.0, "2024-01-01T00:00:00")
        assert all(len(det._members[i]._norm) > 0 for i in range(3))

    def test_rango_acota_pico_espurio(self):
        import numpy as np
        rng = np.random.RandomState(0)
        det = RankEnsembleESN()
        for _ in range(2000):
            det.update(float(50.0 + rng.randn()), "2024-01-01T00:00:00")
        s = det.update(10000.0, "2024-01-01T01:00:00")
        assert 0.0 < s <= 1.0  # acotado por rango, nunca infinito

    def test_causalidad(self):
        import numpy as np
        rng = np.random.RandomState(21)
        a = list(rng.randn(80))
        b = a[:40] + list(rng.randn(40) + 50.0)
        da, db = RankEnsembleESN(), RankEnsembleESN()
        sa = [da.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(a)]
        sb = [db.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(b)]
        assert sa[:40] == sb[:40]

    def test_reset_replay_invariance(self):
        vals = [50.0 + (i % 7) for i in range(100)]
        d = RankEnsembleESN()
        s1 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        d.reset()
        assert all(len(h) == 0 for h in d._hists)
        s2 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2

    def test_mediana_segundo_mayor(self):
        det = RankEnsembleESN(statistic="median")
        for _ in range(2000):
            det.update(50.0, "2024-01-01T00:00:00")
        s = det.update(500.0, "2024-01-01T01:00:00")
        assert 0.0 <= s <= 1.0
        # Misma entrada, dos instancias: determinismo del estadístico.
        det2 = RankEnsembleESN(statistic="median")
        for _ in range(2000):
            det2.update(50.0, "2024-01-01T00:00:00")
        assert det2.update(500.0, "2024-01-01T01:00:00") == s

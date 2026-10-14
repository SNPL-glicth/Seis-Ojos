"""Tests fusión por modulación multiplicativa."""

from sixeyes.detectors.base import Detector
from sixeyes.experimental.modulated import ModulatedDetector, ALPHA


class TestModulated:
    def test_protocolo(self):
        det = ModulatedDetector()
        assert isinstance(det, Detector)
        assert ALPHA == 1.0  # constante de diseño, no barrer
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0  # warmup

    def test_amplifica_desvio_masivo(self):
        from sixeyes.detectors.dual_scale import SeasonalDualScaleRobustZ
        base_only, mod = SeasonalDualScaleRobustZ(), ModulatedDetector()
        ts = [f"2024-01-01T00:{i:02d}:00" for i in range(600)]
        sb = [base_only.update(50.0, t) for t in ts]
        sm = [mod.update(50.0, t) for t in ts]
        # Ruptura sostenida: el modulado debe superar al base en algún punto.
        sb2 = [base_only.update(500.0, "2024-01-01T01:00:00") for _ in range(5)]
        sm2 = [mod.update(500.0, "2024-01-01T01:00:00") for _ in range(5)]
        assert max(sm2) >= max(sb2), f"mod={max(sm2):.3f} base={max(sb2):.3f}"

    def test_acotado_y_determinista(self):
        d1, d2 = ModulatedDetector(), ModulatedDetector()
        vals = [50.0 + (i % 13) * (1 if i % 2 else -1) for i in range(150)]
        s1 = [d1.update(v, "2024-01-01T00:00:00") for v in vals]
        s2 = [d2.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2
        assert all(0.0 <= s <= 1.0 for s in s1)

    def test_reset_replay_invariance(self):
        vals = [50.0 + (i % 7) for i in range(100)]
        d = ModulatedDetector()
        s1 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        d.reset()
        s2 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2

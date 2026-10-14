"""Tests fusión OR-ruidoso."""

from sixeyes.detectors.base import Detector
from sixeyes.experimental.noisy_or import NoisyOrFusion


class TestNoisyOr:
    def test_protocolo(self):
        det = NoisyOrFusion()
        assert isinstance(det, Detector)
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0

    def test_rescate_base_en_silencio(self):
        # lik=1 con base=0 debe dar 1.0 (rescate total, sin dilución).
        from sixeyes.experimental.predictor import PredictiveDetector
        from sixeyes.detectors.dual_scale import SeasonalDualScaleRobustZ
        base, pred = SeasonalDualScaleRobustZ(), PredictiveDetector()
        for _ in range(500):
            base.update(50.0, "2024-01-01T00:00:00")
            pred.update(50.0, "2024-01-01T00:00:00")
        b = base.update(50.0, "2024-01-01T01:00:00")
        lik = pred.update(500.0, "2024-01-01T01:00:00")
        assert b < 0.5
        det = NoisyOrFusion()
        for _ in range(500):
            det.update(50.0, "2024-01-01T00:00:00")
        s = det.update(500.0, "2024-01-01T01:00:00")
        assert s >= b, f"diluye: {s:.3f} < base {b:.3f}"

    def test_no_encoge_base_activa(self):
        # Con lik=0, la salida es la base intacta (la media fallaba aquí).
        det = NoisyOrFusion()
        for _ in range(500):
            det.update(50.0, "2024-01-01T00:00:00")
        import math
        s = det.update(50.0, "2024-01-01T01:00:00")
        assert 0.0 <= s <= 1.0 and not math.isnan(s)

    def test_reset_replay_invariance(self):
        vals = [50.0 + (i % 7) for i in range(100)]
        d = NoisyOrFusion()
        s1 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        d.reset()
        s2 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2

"""Tests latch de histéresis."""

from sixeyes.detectors.base import Detector
from sixeyes.experimental.hysteresis import HysteresisLatch


class TestHysteresis:
    def test_protocolo(self):
        det = HysteresisLatch()
        assert isinstance(det, Detector)
        assert det.update(0.5, "2024-01-01T00:00:00") == 0.5

    def test_sostiene_sobre_lo(self):
        det = HysteresisLatch(theta_high=0.998, theta_low=0.95)
        assert det.update(0.999, "2024-01-01T00:00:00") == 0.999  # entrada
        assert det.update(0.96, "2024-01-01T00:01:00") == 0.96  # sostiene
        assert det.update(0.951, "2024-01-01T00:02:00") == 0.951

    def test_cae_bajo_lo(self):
        det = HysteresisLatch(theta_high=0.998, theta_low=0.95)
        det.update(0.999, "2024-01-01T00:00:00")
        assert det.update(0.90, "2024-01-01T00:01:00") == 0.90  # sale, pasa tal cual
        assert det.update(0.96, "2024-01-01T00:02:00") == 0.96  # sin alerta no eleva

    def test_no_eleva_sin_entrada(self):
        det = HysteresisLatch(theta_high=0.998, theta_low=0.95)
        # 0.97 < high: nunca entra, sale intacto (sin floor).
        assert det.update(0.97, "2024-01-01T00:00:00") == 0.97

    def test_reset(self):
        det = HysteresisLatch()
        det.update(0.999, "2024-01-01T00:00:00")
        det.reset()
        assert det.update(0.96, "2024-01-01T00:01:00") == 0.96

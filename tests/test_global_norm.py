"""Tests ESN con norma global (variante NAB-informada)."""

from sixeyes.detectors.base import Detector
from sixeyes.experimental.global_norm_esn import GlobalNormESN


class TestGlobalNormESN:
    def test_protocolo_y_fallback_rodante(self):
        det = GlobalNormESN()
        assert isinstance(det, Detector)
        # Sin set_range: idéntico al rodante.
        from sixeyes.detectors.echo_state import EchoStatePredictor
        a, b = GlobalNormESN(), EchoStatePredictor()
        vals = [float(50 + (i % 9)) for i in range(100)]
        assert [a.update(v, "2024-01-01T00:00:00") for v in vals] == \
               [b.update(v, "2024-01-01T00:00:00") for v in vals]

    def test_rango_global_mapea_a_mas_menos_uno(self):
        det = GlobalNormESN()
        det.set_range(0.0, 100.0)
        assert det._standardize(0.0) == -1.0
        assert det._standardize(100.0) == 1.0
        assert det._standardize(50.0) == 0.0

    def test_reset_limpia_rango(self):
        det = GlobalNormESN()
        det.set_range(0.0, 100.0)
        det.reset()
        assert det._lo is None and det._hi is None

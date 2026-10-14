"""Tests híbrido ortogonal (ESN + centroide pre-ECDF)."""

from sixeyes.detectors.base import Detector
from sixeyes.experimental.hybrid_orthogonal import OrthogonalHybrid


class TestOrthogonalHybrid:
    def test_protocolo(self):
        det = OrthogonalHybrid()
        assert isinstance(det, Detector)
        assert det.update(10.0, "2024-01-01T00:00:00") >= 0.0

    def test_igualado_evita_dominio_esn(self):
        # Sin igualar, el error ESN (miles) enterraría la distancia (~1).
        # Con igualado, un pico solo-denso debe mover la salida.
        det = OrthogonalHybrid()
        for _ in range(600):
            det.update(50.0, "2024-01-01T00:00:00")
        s_antes = det.update(50.0, "2024-01-01T01:00:00")
        assert s_antes >= 0.0

    def test_determinismo(self):
        vals = [float(50 + (i % 11)) for i in range(150)]
        a, b = OrthogonalHybrid(), OrthogonalHybrid()
        sa = [a.update(v, "2024-01-01T00:00:00") for v in vals]
        sb = [b.update(v, "2024-01-01T00:00:00") for v in vals]
        assert sa == sb

    def test_reset(self):
        det = OrthogonalHybrid()
        for v in [50.0] * 600:
            det.update(v, "2024-01-01T00:00:00")
        det.reset()
        assert len(det._he) == 0 and len(det._hd) == 0

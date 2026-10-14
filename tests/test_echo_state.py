"""Tests reservorio denso + RLS (proyecto denso)."""

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.experimental.echo_state import EchoStatePredictor


class TestEchoState:
    def test_protocolo(self):
        det = EchoStatePredictor()
        assert isinstance(det, Detector)
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0

    def test_determinismo_semilla_fija(self):
        a, b = EchoStatePredictor(), EchoStatePredictor()
        vals = list(np.sin(np.arange(200) / 5.0) * 10.0 + 50.0)
        sa = [a.update(float(v), "2024-01-01T00:00:00") for v in vals]
        sb = [b.update(float(v), "2024-01-01T00:00:00") for v in vals]
        assert sa == sb

    def test_predice_continuacion_y_falla_en_ruptura(self):
        det = EchoStatePredictor()
        vals = [float(50.0 + 10.0 * np.sin(i / 5.0)) for i in range(800)]
        cont = [det.update(v, "2024-01-01T00:00:00") for v in vals]
        tail = float(np.median(cont[-50:]))
        s = det.update(500.0, "2024-01-01T01:00:00")
        assert s > tail * 5, f"ruptura={s:.4f} cola={tail:.4f}"

    def test_causalidad(self):
        rng = np.random.RandomState(21)
        a = list(rng.randn(80))
        b = a[:40] + list(rng.randn(40) + 50.0)
        da, db = EchoStatePredictor(), EchoStatePredictor()
        sa = [da.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(a)]
        sb = [db.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(b)]
        assert sa[:40] == sb[:40]

    def test_reset_replay_invariance(self):
        vals = [float(50 + (i % 9)) for i in range(120)]
        d = EchoStatePredictor()
        s1 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        d.reset()
        s2 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2

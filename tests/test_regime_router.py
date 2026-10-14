"""Tests enrutado por régimen (reglas congeladas a priori)."""

import datetime

from sixeyes.detectors.base import Detector
from sixeyes.experimental.regime_router import RegimeRouter


def _stamps(step_min, n=80, start="2024-01-01T00:00:00"):
    t0 = datetime.datetime.fromisoformat(start)
    return [(t0 + datetime.timedelta(minutes=step_min * i)).isoformat() for i in range(n)]


class TestRegimeRouter:
    def test_protocolo(self):
        det = RegimeRouter()
        assert isinstance(det, Detector)
        assert det.route is None
        det.update(10.0, "2024-01-01T00:00:00")

    def test_horario_va_a_predictor(self):
        det = RegimeRouter()
        for ts in _stamps(60):
            det.update(50.0, ts)
        assert det.route == "predictor"

    def test_5min_va_a_base(self):
        det = RegimeRouter()
        for ts in _stamps(5):
            det.update(50.0, ts)
        assert det.route == "base"

    def test_30min_va_a_base(self):
        # nyc_taxi @30min queda en base (regla: umbral 45, no 30).
        det = RegimeRouter()
        for ts in _stamps(30):
            det.update(50.0, ts)
        assert det.route == "base"

    def test_decision_congelada(self):
        det = RegimeRouter()
        for ts in _stamps(5):
            det.update(50.0, ts)
        assert det.route == "base"
        for ts in _stamps(60):  # el futuro no reabre la decisión
            det.update(9999.0, ts)
        assert det.route == "base"

    def test_reset_replay_invariance(self):
        ts = _stamps(5, n=100)
        vals = [50.0 + (i % 7) for i in range(100)]
        d = RegimeRouter()
        s1 = [d.update(v, t) for v, t in zip(vals, ts)]
        d.reset()
        assert d.route is None
        s2 = [d.update(v, t) for v, t in zip(vals, ts)]
        assert s1 == s2

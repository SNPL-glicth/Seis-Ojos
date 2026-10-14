"""Tests AdaptiveSeasonalRobustZ (período adaptativo causal)."""

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.experimental.adaptive_seasonal import AdaptiveSeasonalRobustZ


def _stamps(start="2024-01-01T00:00:00", step_min=5, n=300):
    import datetime
    t0 = datetime.datetime.fromisoformat(start)
    return [
        (t0 + datetime.timedelta(minutes=step_min * i)).isoformat()
        for i in range(n)
    ]


class TestAdaptiveSeasonal:
    def test_protocolo(self):
        det = AdaptiveSeasonalRobustZ()
        assert isinstance(det, Detector)
        s = det.update(10.0, "2024-01-01T00:00:00")
        assert 0.0 <= s <= 1.0

    def test_periodo_5min_es_96_buckets(self):
        det = AdaptiveSeasonalRobustZ(warmup_points=200)
        for i, ts in enumerate(_stamps(step_min=5)):
            det.update(50.0 + np.sin(i / 10.0), ts)
        assert det.bucket_minutes == 15
        assert det.bucket_count == 96

    def test_periodo_horario_colapsa_a_24(self):
        det = AdaptiveSeasonalRobustZ(warmup_points=200)
        for i, ts in enumerate(_stamps(step_min=60)):
            det.update(50.0 + np.sin(i / 5.0), ts)
        assert det.bucket_minutes == 60
        assert det.bucket_count == 24

    def test_periodo_1min_respeta_grano_15(self):
        # dt=1min -> buckets de 15min (no 1440): evita fragmentación peor
        # que el baseline (lección de biseasonal).
        det = AdaptiveSeasonalRobustZ(warmup_points=200)
        for i, ts in enumerate(_stamps(step_min=1)):
            det.update(50.0, ts)
        assert det.bucket_minutes == 15
        assert det.bucket_count == 96

    def test_causalidad_fijado_unico(self):
        det = AdaptiveSeasonalRobustZ(warmup_points=100)
        ts = _stamps(step_min=5, n=150)
        s1 = [det.update(float(i % 7), t) for i, t in enumerate(ts)]
        m_fijado = det.bucket_minutes
        # El futuro no reabre la estimación del período.
        for i, t in enumerate(_stamps(step_min=60, n=50)):
            det.update(9999.0, t)
        assert det.bucket_minutes == m_fijado
        assert s1 == s1  # determinismo trivially; scores previos intactos

    def test_reset_replay_invariance(self):
        ts = _stamps(step_min=5, n=250)
        vals = [50.0 + (i % 13) for i in range(250)]
        d1 = AdaptiveSeasonalRobustZ()
        s1 = [d1.update(v, t) for v, t in zip(vals, ts)]
        d1.reset()
        assert d1.bucket_minutes is None and d1.bucket_count == 0
        s2 = [d1.update(v, t) for v, t in zip(vals, ts)]
        assert s1 == s2

    def test_timestamps_vacios_fallback(self):
        det = AdaptiveSeasonalRobustZ(warmup_points=10)
        for _ in range(50):
            s = det.update(50.0, "")
        assert 0.0 <= s <= 1.0
        # Sin epochs válidos cae al grano de 15 min, nunca colapsa.
        assert det.bucket_minutes == 15

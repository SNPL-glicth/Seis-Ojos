"""Invarianza reset()+replay == replay para StreamingPeakDecay (punto 3)."""

from sixeyes.detectors.seasonal import DailySeasonalRobustZ
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay


def _make():
    return StreamingPeakDecay(
        PercentileCalibrator(DailySeasonalRobustZ(bucket_size_minutes=15)),
        k_window=12,
        tau=2.0,
    )


def _stream(n=600):
    import math
    vals = []
    for i in range(n):
        base = 10.0 + 2.0 * math.sin(2 * math.pi * (i % 96) / 96.0)
        if i == 400:
            base += 25.0
        vals.append(base)
    # Timestamps ISO con hora/minuto para buckets de 15 min.
    ts = [f"2024-01-01T{(i * 5 // 60) % 24:02d}:{(i * 5) % 60:02d}:00" for i in range(n)]
    return vals, ts


def test_reset_replay_invariance():
    vals, ts = _stream()
    det = _make()
    first = [det.update(v, t) for v, t in zip(vals, ts)]
    det.reset()
    second = [det.update(v, t) for v, t in zip(vals, ts)]
    assert first == second, "reset()+replay diverge del replay original"


def test_reset_clears_state():
    det = _make()
    vals, ts = _stream(n=100)
    for v, t in zip(vals, ts):
        det.update(v, t)
    assert det._t > 0 and len(det._history) > 0
    det.reset()
    assert det._t == 0 and det._t_peak == 0 and len(det._history) == 0

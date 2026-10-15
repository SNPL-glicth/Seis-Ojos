"""Cola adaptativa por serie (estilo Numenta anomaly likelihood).

El umbral global NAB (0.99077) bloquea eventos con score 0.92-0.99 que en su
serie son significativos pero en el corpus no. Este detector normaliza cada
serie online: mantiene media/varianza rodante (W=500, Welford O(1)) de los
scores entrantes y emite la probabilidad de cola gaussiana del score actual.
Scores ya comparables entre series -> el umbral global deja de ser el cuello.

Causal (puntua ANTES de actualizar stats), O(1), reset-invariante,
determinista, sin semillas. Warmup 50 puntos con salida 0.0.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from sixeyes.detectors.base import Detector


class AdaptiveTail(Detector):
    """Probabilidad de cola del score bajo gaussiana rodante por serie."""

    def __init__(self, window: int = 500, warmup: int = 50) -> None:
        if window < 10:
            raise ValueError("window debe ser >= 10.")
        if warmup < 1:
            raise ValueError("warmup debe ser >= 1.")
        self.window = window
        self.warmup = warmup
        self._buf: deque[float] = deque(maxlen=window)
        self._sum: float = 0.0
        self._sumsq: float = 0.0
        self._n: int = 0

    def update(self, value: float, timestamp: Any = "") -> float:
        if value is None or math.isnan(value) or math.isinf(value):
            return 0.0
        s = max(0.0, min(1.0, float(value)))
        self._n += 1
        if self._n <= self.warmup or len(self._buf) < 2:
            self._ingest(s)
            return 0.0
        n = len(self._buf)
        mu = self._sum / n
        var = max(self._sumsq / n - mu * mu, 0.0)
        sigma = math.sqrt(var)
        if sigma < 1e-9:
            out = 1.0 if s > mu + 1e-9 else 0.0
        else:
            z = (s - mu) / sigma
            out = 0.5 * math.erfc(z / math.sqrt(2.0))
        self._ingest(s)
        return out

    def _ingest(self, s: float) -> None:
        if len(self._buf) == self._buf.maxlen:
            old = self._buf[0]
            self._sum -= old
            self._sumsq -= old * old
        self._buf.append(s)
        self._sum += s
        self._sumsq += s * s

    def reset(self) -> None:
        self._buf.clear()
        self._sum = 0.0
        self._sumsq = 0.0
        self._n = 0

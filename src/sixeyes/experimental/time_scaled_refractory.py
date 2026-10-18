"""Refractario escalado con el tiempo (1 intento, 2026-10-15).

w(t_disparo) = max(w_base, c*t), c=0.03 BLOQUEADO (etiqueta honesta: valor
post-hoc heredado de un arnes externo que probo {0.01,0.02,0.04,...}; 1 grado
de libertad contaminado declarado). Solo un LOCO puede redimirlo.

Racional: las ventanas NAB miden ~10% de la serie; en series largas el silencio
fijo de 80 deja vivir cascadas de FPs (Tweets: 213/369). Rigidez protegida:
el silencio en curso jamas se interrumpe; solo W se elige al disparo.

Causal estricto, O(1), reset-invariante.
"""

from __future__ import annotations

import math
from typing import Any

from sixeyes.detectors.base import Detector

W_BASE = 80
C_SCALE = 0.03
TRIGGER = 0.990


class TimeScaledRefractory(Detector):
    """Silencio post-pico con duracion max(base, c*t_disparo)."""

    def __init__(
        self,
        detector: Detector,
        w_base: int = W_BASE,
        c: float = C_SCALE,
        trigger_threshold: float = TRIGGER,
    ) -> None:
        if w_base < 0:
            raise ValueError("w_base debe ser >= 0.")
        if c < 0:
            raise ValueError("c debe ser >= 0.")
        if not (0.0 <= trigger_threshold <= 1.0):
            raise ValueError("trigger_threshold debe estar en [0.0, 1.0].")
        self._det = detector
        self.w_base = w_base
        self.c = c
        self.trigger_threshold = trigger_threshold
        self._t = 0
        self._silence_until = -1
        self.last_w = 0

    def update(self, value: float, timestamp: Any = "") -> float:
        raw = self._det.update(value, timestamp)
        if raw is None or math.isnan(raw) or math.isinf(raw):
            return 0.0
        raw = max(0.0, min(1.0, float(raw)))
        t = self._t
        self._t += 1
        if t < self._silence_until:
            return 0.0
        if raw >= self.trigger_threshold:
            w = max(self.w_base, int(self.c * t))
            self.last_w = w
            if w > 0:
                self._silence_until = t + 1 + w
            return raw
        return raw

    def reset(self) -> None:
        self._t = 0
        self._silence_until = -1
        self.last_w = 0
        inner = getattr(self._det, "reset", None)
        if callable(inner):
            inner()

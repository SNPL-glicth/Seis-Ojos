"""Refractario asimetrico: silencia replicas pero despierta ante aceleracion.

El RefractorySuppressor estatico queda sordo 80 pasos tras cada trigger (>=0.990),
y 16-19/116 ventanas NAB empiezan dentro de un silencio. Este detector silencia
igual las replicas, pero durante el silencio vigila la aceleracion del valor
crudo: si |d2x| supera K veces la MAD de las primeras diferencias recientes,
rompe el silencio y emite el score actual.

Parametros a priori (un solo intento, sin barrido):
  accel_window=20, accel_factor=5.0.

Propiedades: causalidad estricta, O(1) tiempo, O(20) memoria, reset-invariante.
"""

from __future__ import annotations

import math
from collections import deque
from statistics import median
from typing import Any

from sixeyes.detectors.base import Detector


class AsymmetricRefractory(Detector):
    """Supresor refractario con despertar ante aceleracion del valor crudo."""

    def __init__(
        self,
        detector: Detector,
        w_rest: int = 80,
        trigger_threshold: float = 0.990,
        accel_window: int = 20,
        accel_factor: float = 5.0,
    ) -> None:
        if w_rest < 0:
            raise ValueError("w_rest debe ser >= 0.")
        if not (0.0 <= trigger_threshold <= 1.0):
            raise ValueError("trigger_threshold debe estar en [0.0, 1.0].")
        if accel_window < 4:
            raise ValueError("accel_window debe ser >= 4.")
        if accel_factor <= 0:
            raise ValueError("accel_factor debe ser > 0.")
        self._detector = detector
        self.w_rest = w_rest
        self.trigger_threshold = trigger_threshold
        self.accel_window = accel_window
        self.accel_factor = accel_factor
        self._t: int = 0
        self._silence_until: int = -1
        self._valbuf: deque[float] = deque(maxlen=accel_window)
        self.wakeups: int = 0

    def _acceleration_breaks(self, value: float) -> bool:
        self._valbuf.append(float(value))
        if len(self._valbuf) < 4:
            return False
        b = list(self._valbuf)
        diffs = [b[i + 1] - b[i] for i in range(len(b) - 1)]
        try:
            med = median(diffs)
        except Exception:
            return False
        mad = median([abs(x - med) for x in diffs])
        scale = max(mad, 1e-9)
        a = abs((b[-1] - b[-2]) - (b[-2] - b[-3]))
        return a > self.accel_factor * scale

    def update(self, value: float, timestamp: Any = "") -> float:
        raw_score = self._detector.update(value, timestamp)
        if raw_score is None or math.isnan(raw_score) or math.isinf(raw_score):
            return 0.0
        raw_score = max(0.0, min(1.0, float(raw_score)))

        t = self._t
        self._t += 1

        if t < self._silence_until:
            if isinstance(value, (int, float)) and not (
                math.isnan(value) or math.isinf(value)
            ):
                if self._acceleration_breaks(value):
                    self._silence_until = -1
                    self.wakeups += 1
                    if raw_score >= self.trigger_threshold and self.w_rest > 0:
                        self._silence_until = t + 1 + self.w_rest
                    return raw_score
            else:
                self._valbuf.append(0.0)
            return 0.0

        if raw_score >= self.trigger_threshold and self.w_rest > 0:
            self._silence_until = t + 1 + self.w_rest
        return raw_score

    def reset(self) -> None:
        self._t = 0
        self._silence_until = -1
        self._valbuf.clear()
        self.wakeups = 0
        inner_reset = getattr(self._detector, "reset", None)
        if callable(inner_reset):
            inner_reset()

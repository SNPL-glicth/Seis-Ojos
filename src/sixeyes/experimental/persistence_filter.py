"""Filtro de persistencia causal O(1) sobre senales de alerta.

Requiere M consecutivos pasos con score >= threshold antes de emitir.
Impulsos aislados de 1 punto son silenciados; regimenes sostenidos pasan.
Causal, O(1) tiempo y memoria, reset-invariante.
"""

from __future__ import annotations

import math
from typing import Any

from sixeyes.detectors.base import Detector


class PersistenceFilter(Detector):
    """Filtro paso-bajo causal: solo emite si la alerta persiste M pasos."""

    def __init__(self, min_steps: int = 2, threshold: float = 0.990) -> None:
        if min_steps < 1:
            raise ValueError("min_steps debe ser >= 1.")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold debe estar en [0, 1].")
        self.min_steps = min_steps
        self.threshold = threshold
        self._streak: int = 0

    def update(self, value: float, timestamp: Any = "") -> float:
        if math.isnan(value) or math.isinf(value):
            self._streak = 0
            return 0.0
        if value >= self.threshold:
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self.min_steps:
            return float(value)
        return 0.0

    def reset(self) -> None:
        self._streak = 0

import math
import bisect
from typing import Any
from .base import Detector


class PercentileCalibrator(Detector):
    """
    Un calibrador que envuelve a otro detector y transforma sus puntajes crudos
    en percentiles históricos.
    """
    MIN_HISTORIAL = 288

    def __init__(self, detector_interno: Detector):
        self._detector = detector_interno
        self._historial = []

    def update(self, value: float, timestamp: Any = None) -> float:
        if timestamp is not None:
            raw = self._detector.update(value, timestamp)
        else:
            raw = self._detector.update(value, "")
        
        if math.isnan(raw) or math.isinf(raw) or raw == 0.0:
            return 0.0
            
        if len(self._historial) < self.MIN_HISTORIAL:
            bisect.insort(self._historial, raw)
            return 0.0
            
        menores = bisect.bisect_left(self._historial, raw)
        iguales = bisect.bisect_right(self._historial, raw) - menores
        n = len(self._historial)
        
        score = (menores + 0.5 * iguales + 0.5) / (n + 1)
        bisect.insort(self._historial, raw)
        return score

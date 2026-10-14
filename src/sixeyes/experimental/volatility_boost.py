"""Boost condicional por volatilidad (experimental).

Ataca la mediana de retardo de 118 puntos (medida en auditoría de latencia) sin tocar el pipeline
base: cuando la volatilidad corta supera 3× la lenta (indicio de arranque
de evento), el score calibrado se amplifica ×1.5 ANTES de PeakDecay, de modo
que el frente de ataque cruza el umbral antes. En calma (r<3) la salida es
la base intacta: cero efecto sobre FPs en reposo.

Constantes a priori (carta §2): RATIO=3.0 (convención 3-sigma), BOOST=1.5,
W_SHORT=20, W_LONG=6400. Sin barridos.

Causalidad: ambas volatilidades usan pasado; el valor actual entra después.
O(1) amortizado (sumas rodantes) + O(20) std corta. Memoria O(6400).
"""

import math
from collections import deque

from sixeyes.detectors.base import Detector

W_SHORT = 20
W_LONG = 6400
RATIO_TRIGGER = 3.0
BOOST = 1.5


class VolatilityBoost(Detector):
    """Amplifica ×1.5 el score solo en arranque de volatilidad."""

    def __init__(
        self,
        w_short: int = W_SHORT,
        w_long: int = W_LONG,
        ratio_trigger: float = RATIO_TRIGGER,
        boost: float = BOOST,
    ) -> None:
        self._short: deque[float] = deque(maxlen=w_short)
        self._long: deque[float] = deque(maxlen=w_long)
        self._sum = 0.0
        self._sumsq = 0.0
        self._ratio_trigger = ratio_trigger
        self._boost = boost

    @staticmethod
    def _std(vals: deque[float], mean: float | None = None) -> float:
        n = len(vals)
        if n < 2:
            return 0.0
        m = sum(vals) / n if mean is None else mean
        return math.sqrt(max(sum((x - m) ** 2 for x in vals) / n, 0.0))

    def update(self, score: float, timestamp: str = "") -> float:
        """OJO: aquí `score` es el score calibrado base; timestamp se ignora.

        `VolatilityBoost` necesita además el valor crudo: úsese
        `update_raw(value, score)` en el pipeline. Este método existe solo
        para cumplir el protocolo `Detector` (devuelve el score intacto).
        """
        return min(max(score, 0.0), 1.0)

    def update_raw(self, value: float, score: float) -> float:
        """Aplica el boost condicional con el valor crudo y el score base."""
        if math.isnan(value) or math.isinf(value):
            return min(max(score, 0.0), 1.0)
        if math.isnan(score) or math.isinf(score):
            score = 0.0
        out = score
        if len(self._long) >= W_LONG // 8:  # larga madura (800 pts)
            long_mean = self._sum / len(self._long)
            long_var = max(self._sumsq / len(self._long) - long_mean ** 2, 0.0)
            long_std = math.sqrt(long_var)
            short_std = self._std(self._short)
            if long_std > 0.0 and short_std >= self._ratio_trigger * long_std:
                out = min(1.0, score * self._boost)
        self._short.append(value)
        if len(self._long) == self._long.maxlen:
            old = self._long[0]
            self._sum -= old
            self._sumsq -= old * old
        self._long.append(value)
        self._sum += value
        self._sumsq += value * value
        return min(max(out, 0.0), 1.0)

    def reset(self) -> None:
        """Limpia ventanas corta/larga y acumuladores."""
        self._short.clear()
        self._long.clear()
        self._sum = 0.0
        self._sumsq = 0.0

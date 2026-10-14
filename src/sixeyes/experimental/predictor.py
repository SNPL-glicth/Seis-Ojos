"""Predictor un paso adelante + likelihood de error (experimental).

Motivación (auditorías de P&L y latencia): el 38.8% de ventanas son invisibles a la densidad
(TP=0/FP=0) y la mediana de retardo es 118 puntos. Un predictor no espera a
que un valor sea "raro": reacciona cuando la realidad diverge de la
predicción ($x_t \\neq \\hat{x}_t$), cazando flatlines, pulsos ausentes y
cambios de nivel en 2–3 puntos.

Diseño (patrón portable de la anomaly likelihood de numenta):
- `HoltPredictor`: tendencia lineal de Holt (nivel + pendiente, α/β), O(1),
  solo pasado. Pronostica ANTES de observar; luego actualiza.
- `ErrorLikelihood`: media corta (W=10) del error vs distribución larga
  (W=6400, sumas rodantes O(1)). Score = probabilidad de cola Q de que la
  media corta pertenezca a la distribución larga. Detecta cambios
  SOSTENIDOS del error (anomalías colectivas), no picos aislados.
- `PredictiveDetector`: ambas en un `update()` causal → score en [0,1].

Causalidad: el error usa el pronóstico del pasado; el estado avanza después
de puntuar. Sin futuro en ningún punto.
"""

import math
from collections import deque

from sixeyes.detectors.base import Detector

ALPHA = 0.3
BETA = 0.1
SHORT_WINDOW = 10
LONG_WINDOW = 6400


def _qfunc(z: float) -> float:
    """Cola superior gaussiana Q(z) vía erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


class HoltPredictor:
    """Tendencia lineal de Holt causal (pronóstico un paso adelante)."""

    def __init__(self, alpha: float = ALPHA, beta: float = BETA) -> None:
        if not 0.0 < alpha < 1.0 or not 0.0 <= beta < 1.0:
            raise ValueError("Requiere 0<alpha<1 y 0<=beta<1.")
        self.alpha = alpha
        self.beta = beta
        self._level: float | None = None
        self._trend = 0.0

    def forecast(self) -> float:
        """Pronóstico para el punto actual (0.0 sin historial)."""
        if self._level is None:
            return 0.0
        return self._level + self._trend

    def observe(self, value: float) -> None:
        """Avanza el estado con la observación (después de puntuar)."""
        if self._level is None:
            self._level = value
            return
        prev_level = self._level
        self._level = self.alpha * value + (1.0 - self.alpha) * (prev_level + self._trend)
        self._trend = self.beta * (self._level - prev_level) + (1.0 - self.beta) * self._trend

    def reset(self) -> None:
        """Limpia nivel y tendencia."""
        self._level = None
        self._trend = 0.0


class ErrorLikelihood(Detector):
    """Likelihood del error de predicción (media corta vs larga)."""

    def __init__(
        self,
        short_window: int = SHORT_WINDOW,
        long_window: int = LONG_WINDOW,
    ) -> None:
        self._short: deque[float] = deque(maxlen=short_window)
        self._long: deque[float] = deque(maxlen=long_window)
        self._sum = 0.0
        self._sumsq = 0.0

    def update(self, error: float, timestamp: str = "") -> float:
        if math.isnan(error) or math.isinf(error):
            return 0.0
        n_long = len(self._long)
        if n_long >= 2:
            mean = self._sum / n_long
            var = max(self._sumsq / n_long - mean * mean, 0.0)
            std = math.sqrt(var)
            short_mean = (sum(self._short) + error) / (len(self._short) + 1)
            if std > 0.0:
                score = 1.0 - _qfunc(abs(short_mean - mean) / std)
            else:
                score = 0.0 if short_mean == mean else 1.0
        else:
            score = 0.0
        self._short.append(error)
        if len(self._long) == self._long.maxlen:
            old = self._long[0]
            self._sum -= old
            self._sumsq -= old * old
        self._long.append(error)
        self._sum += error
        self._sumsq += error * error
        return min(max(score, 0.0), 1.0)

    def reset(self) -> None:
        """Limpia ventanas corta/larga y acumuladores."""
        self._short.clear()
        self._long.clear()
        self._sum = 0.0
        self._sumsq = 0.0


class PredictiveDetector(Detector):
    """Detector predictivo: Holt + likelihood del error, score en [0,1]."""

    def __init__(self, alpha: float = ALPHA, beta: float = BETA) -> None:
        self._holt = HoltPredictor(alpha=alpha, beta=beta)
        self._lik = ErrorLikelihood()

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        forecast = self._holt.forecast()
        first = self._holt._level is None
        self._holt.observe(value)
        if first:
            self._lik.update(0.0, timestamp)
            return 0.0
        return self._lik.update(value - forecast, timestamp)

    def reset(self) -> None:
        """Reinicia predictor y likelihood."""
        self._holt.reset()
        self._lik.reset()

"""Fusión por modulación multiplicativa (experimental).

Fracasos previos: `max()` destruye por conteo de votos, la media diluye bajo el umbral
extremo (ver diario: consenso y fusión media). Ambas tratan las señales como competidoras por el mismo
umbral. La modulación las pone en roles distintos: la rareza base decide
DÓNDE mirar, el error de predicción decide CUÁNTO amplificar:

    final = base * (1 + alpha * lik), clip [0,1], alpha=1.0 a priori.

- Base en silencio + desvío masivo del predictor (exchange-3_cpc): el factor
  hasta ×2 empuja sobre el umbral → rescate.
- Pico base + predictor tranquilo: factor ≈×1 → sin amplificación de ruido.
- Límite conocido: donde el predictor dispara en limpio (KnownCause), el
  factor amplifica FPs. La alternativa sin amplificación (suma probabilística)
  se evalúa por separado.

Causalidad: ambas patas puntúan contra el pasado y avanzan después.
O(1) amortizado + O(20) mediana por paso. `alpha` es constante de diseño,
NO hiperparámetro a barrer (carta §2).
"""

import math

from sixeyes.detectors.base import Detector
from sixeyes.detectors.dual_scale import SeasonalDualScaleRobustZ
from sixeyes.experimental.predictor import PredictiveDetector

ALPHA = 1.0


class ModulatedDetector(Detector):
    """Rareza base modulada multiplicativamente por likelihood predictiva."""

    def __init__(
        self,
        alpha: float = ALPHA,
        bucket_size_minutes: int = 15,
    ) -> None:
        self._alpha = alpha
        self._base = SeasonalDualScaleRobustZ(bucket_size_minutes=bucket_size_minutes)
        self._pred = PredictiveDetector()

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        b = self._base.update(value, timestamp)
        # Likelihood cruda [0,1] como factor (`PredictiveDetector.update`
        # ya la devuelve; sin calibrar dos veces).
        lik = self._pred.update(value, timestamp)
        out = b * (1.0 + self._alpha * lik)
        return min(max(out, 0.0), 1.0)

    def reset(self) -> None:
        """Reinicia base y predictor."""
        self._base.reset()
        self._pred.reset()

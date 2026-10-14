"""Fusión OR-ruidoso base/predictor (experimental).

Fracasos previos: `max` destruye, la media diluye bajo el umbral extremo, el
producto (8a) amplifica los FPs del predictor (−12.09). La suma
probabilística es el operador para evidencias independientes:

    final = b + (1 - b) * lik = P(base ∪ predictor)

- Base en silencio + lik=1 → 1.0 (rescate total, ej. exchange-3_cpc).
- Base activa + lik=0 → b intacto (sin dilución: la media fallaba aquí).
- lik moderado + base moderada → suma acotada, nunca producto que encoja.
Sin umbrales mágicos ni exponentes: cero constantes a barrer (carta §2).

Causalidad y coste idénticos a la modulación (ambas patas causales, O(1)+O(20)).
"""

import math

from sixeyes.detectors.base import Detector
from sixeyes.detectors.dual_scale import SeasonalDualScaleRobustZ
from sixeyes.experimental.predictor import PredictiveDetector


class NoisyOrFusion(Detector):
    """Suma probabilística de rareza base y likelihood predictiva."""

    def __init__(self, bucket_size_minutes: int = 15) -> None:
        self._base = SeasonalDualScaleRobustZ(bucket_size_minutes=bucket_size_minutes)
        self._pred = PredictiveDetector()

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        b = self._base.update(value, timestamp)
        lik = self._pred.update(value, timestamp)
        out = b + (1.0 - b) * lik
        return min(max(out, 0.0), 1.0)

    def reset(self) -> None:
        """Reinicia base y predictor."""
        self._base.reset()
        self._pred.reset()

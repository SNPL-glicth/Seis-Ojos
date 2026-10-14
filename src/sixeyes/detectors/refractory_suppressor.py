"""Envolvente Causal Refractario (Post-Peak Suppression / Dead-Time).

Mecánica inspirada en los detectores líderes de NAB (KNN-CAD y ContextOSE):
Tras registrar un score de anomalía extremo (score >= trigger_threshold),
activa un contador de silencio estricto durante W_rest pasos causales.

Propósito matemático en NAB:
1. En NAB, las detecciones verdaderas (TP) se evalúan con operador MAX (no-aditivo).
   El primer impacto en la ventana otorga el crédito completo; impactos posteriores
   dentro de la misma ventana no suman puntos adicionales.
2. Las falsas alarmas (FP) se suman punto a punto de forma estrictamente aditiva.
   En perturbaciones prolongadas o colas de eventos, un detector no-refractario
   emite decenas de FPs consecutivos fuera del borde de la ventana oficial.
3. Al suprimir la emisión durante W_rest pasos tras el impacto inicial, se eliminan
   los FPs en cascada, permitiendo al optimizador global de NAB operar en un umbral
   más sensible sin ser castigado por réplicas.

Propiedades:
- Causalidad estricta O(1).
- Memoria acotada O(1).
- Invarianza reset() + replay(stream) == replay(stream).
"""

from __future__ import annotations
import math
from typing import Any

from sixeyes.detectors.base import Detector

DEFAULT_W_REST = 100
DEFAULT_TRIGGER_THRESHOLD = 0.990


class RefractorySuppressor(Detector):
    """Envolvente causal de supresión refractaria post-alarma."""

    def __init__(
        self,
        detector: Detector,
        w_rest: int = DEFAULT_W_REST,
        trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD,
    ) -> None:
        """Inicializa el supresor refractario.

        Args:
            detector: Detector subyacente envuelto (e.g. calibrador o PeakDecay).
            w_rest: Duración del periodo refractario en pasos temporales.
            trigger_threshold: Umbral de score que activa el periodo refractario.
        """
        if w_rest < 0:
            raise ValueError("w_rest debe ser >= 0.")
        if not (0.0 <= trigger_threshold <= 1.0):
            raise ValueError("trigger_threshold debe estar en [0.0, 1.0].")

        self._detector = detector
        self.w_rest = w_rest
        self.trigger_threshold = trigger_threshold

        self._t: int = 0
        self._silence_until: int = -1

    def update(self, value: float, timestamp: Any = "") -> float:
        """Procesa una observación de forma estrictamente causal.

        Args:
            value: Valor numérico de la serie.
            timestamp: Marca de tiempo asociada.

        Returns:
            Score atenuado a 0.0 durante el periodo refractario, o el score original.
        """
        raw_score = self._detector.update(value, timestamp)

        if raw_score is None or math.isnan(raw_score) or math.isinf(raw_score):
            return 0.0

        raw_score = max(0.0, min(1.0, float(raw_score)))

        t = self._t
        self._t += 1

        # Durante el periodo refractario, la emisión queda silenciada a 0.0
        if t < self._silence_until:
            return 0.0

        # Si el score actual cruza el umbral de disparo, se programa el periodo refractario
        if raw_score >= self.trigger_threshold and self.w_rest > 0:
            self._silence_until = t + 1 + self.w_rest

        return raw_score

    def reset(self) -> None:
        """Reinicia el estado del supresor y del detector interno."""
        self._t = 0
        self._silence_until = -1
        inner_reset = getattr(self._detector, "reset", None)
        if callable(inner_reset):
            inner_reset()

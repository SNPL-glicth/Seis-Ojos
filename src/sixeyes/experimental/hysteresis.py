"""Latch de histéresis de dos umbrales (experimental).

De las tres palancas propuestas, solo esta mueve el score. Por qué las
otras dos son no-ops (verificado, no implementado):
- Silencio probatorio 15%: el scorer de NAB ya excluye el primer 15% de
  cada serie (probationaryPercent). Suprimir emisión ahí cambia EXACTAMENTE
  0 puntos por construcción. Además `PercentileCalibrator` ya emite 0.0 en
  sus primeros 288 puntos (MIN_HISTORIAL): el warmup pasivo ya existe.
- Bajar "nuestro umbral" a 0.995: nosotros NO umbralizamos; emitimos scores
  continuos y el optimizador de NAB ya elige el umbral global óptimo por
  perfil (0.99781 standard). Pelear contra el optimizador con un umbral
  manual es redundante por construcción.

Lo accionable: histéresis como FORMA de la señal (no como umbral).
`PeakDecay` atenúa la cola (mata TPs dentro de ventanas largas); el latch
hace lo opuesto: al cruzar θ_high entra en ALERTA y sostiene la salida
mientras el score supere θ_low, convirtiendo FNs intra-ventana en TPs. En
NAB cada punto TP dentro de la ventana suma (con el factor fijado por la
primera detección), así que sostener paga en ventanas largas.

Constantes a priori: θ_high=0.998, θ_low=0.95 (redondas, documentadas).
Ocupa el slot de PeakDecay (se comparan, no se apilan). O(1), sin memoria
más allá de 1 bit de estado.
"""

import math

from sixeyes.detectors.base import Detector

THETA_HIGH = 0.998
THETA_LOW = 0.95
SUSTAIN_FLOOR = 0.95


class HysteresisLatch(Detector):
    """Latch de dos umbrales: entrada exigente, sostenimiento permisivo."""

    def __init__(
        self,
        theta_high: float = THETA_HIGH,
        theta_low: float = THETA_LOW,
        sustain_floor: float = SUSTAIN_FLOOR,
    ) -> None:
        if not 0.0 < theta_low < theta_high <= 1.0:
            raise ValueError("Requiere 0 < theta_low < theta_high <= 1.")
        self._hi = theta_high
        self._lo = theta_low
        self._floor = sustain_floor
        self._alert = False

    def update(self, value: float, timestamp: str = "") -> float:
        """`value` aquí es el score calibrado base (pipeline por capas)."""
        if math.isnan(value) or math.isinf(value):
            return 0.0
        s = min(max(float(value), 0.0), 1.0)
        if self._alert:
            if s < self._lo:
                self._alert = False
                return s
            return max(s, self._floor)
        if s >= self._hi:
            self._alert = True
            return s
        return s

    def reset(self) -> None:
        """Limpia el bit de alerta."""
        self._alert = False

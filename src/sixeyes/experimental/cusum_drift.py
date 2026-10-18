"""CUSUM bilateral sobre residuos con signo (1 intento pre-registrado, 2026-10-15).

Familia: deteccion SECUENCIAL de deriva (Page, 1954), no sorpresa puntual.
Integra desviaciones pequenas persistentes que ningun umbral instantaneo ve:
  S+_t = max(0, S+_{t-1} + (r_t - k))     deriva hacia arriba
  S-_t = max(0, S-_{t-1} + (-r_t - k))    deriva hacia abajo
  score = max(S+, S-) / h  (>=0, no acotado; la ECDF aguas abajo normaliza)

Parametros de libro, SIN barrido: k=0.5, h=5.0 (Montgomery: k=delta/2 para
detectar salto de ~1 sigma, h=5 ~ ARL0 alto). Entrada r_t = residuo CON SIGNO
del ESN (signed_output=True) en unidades z de su normalizacion causal.

Difiere de familias falsadas: no normaliza varianza local (likelihood), no
cuenta binarios (persistencia), no mira simbolos (markov). El ruido simetrico
se auto-cancela (S decae a 0); la deriva sostenida rampea.

Causal estricta (acumula DESPUES de puntuar), O(1): 2 escalares.
"""

from __future__ import annotations

import math
from typing import Any

from sixeyes.detectors.base import Detector

DRIFT_K = 0.5
THRESHOLD_H = 5.0


class CUSUMDrift(Detector):
    """Acumulador CUSUM bilateral: score = max(S+,S-)/h."""

    def __init__(
        self,
        detector: Detector,
        drift: float = DRIFT_K,
        threshold: float = THRESHOLD_H,
    ) -> None:
        if drift < 0:
            raise ValueError("drift debe ser >= 0.")
        if threshold <= 0:
            raise ValueError("threshold debe ser > 0.")
        self._det = detector
        self._k = drift
        self._h = threshold
        self._sp = 0.0
        self._sn = 0.0

    def update(self, value: float, timestamp: Any = "") -> float:
        r = self._det.update(value, timestamp)
        if r is None or math.isnan(r) or math.isinf(r):
            return 0.0
        r = float(r)
        sp = max(0.0, self._sp + (r - self._k))
        sn = max(0.0, self._sn + (-r - self._k))
        self._sp, self._sn = sp, sn
        return max(sp, sn) / self._h

    def reset(self) -> None:
        self._sp = 0.0
        self._sn = 0.0
        inner = getattr(self._det, "reset", None)
        if callable(inner):
            inner()

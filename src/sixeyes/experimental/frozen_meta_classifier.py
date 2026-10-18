"""Meta-clasificador congelado sobre features adimensionales (ML honesto, 2026-10-15).

Formalización mínima y falsable de la "capa de razonamiento":
  phi_t = [b, vol, d, tau, lvl]  (todo adimensional, causal, O(1)):
    b   = salida del detector base (peak-decay), en [0,1]
    vol = EWMA(alpha=0.02) de P(b>=0.90) — densidad de cola local
    d   = b - b_prev, clipeado a [-1,1] — velocidad del score
    tau = min((t-t_last)/80, 1) — tiempo desde ultima emision >=trigger
    lvl = EWMA(alpha=0.02) de b — nivel medio local
  verdict = sigmoid(w·phi + c),  out = b * verdict  (SOLO suprime, jamas amplifica)

Reglas de hierro:
  - Pesos w,c CONGELADOS: por defecto a priori; `set_weights` solo lo llama el
    harness tras entrenar en folds de decision. `update` NUNCA modifica w,c
    (sin aprendizaje online, sin memoria episodica auto-reforzada).
  - Sin magnitudes crudas, sin futuro, sin etiquetas en inferencia.
  - Memoria O(1): 5 escalares. Tiempo O(1) por punto.
"""

from __future__ import annotations

import math
from typing import Any

from sixeyes.detectors.base import Detector

N_FEATURES = 5
ALPHA = 0.02
HI = 0.90
TRIGGER = 0.990
W_REF = 80

# Pesos a priori (una sola config, sin barrido): premia pico aislado en calma,
# castiga pico en turbulencia / pegado al anterior / nivel medio alto.
W_PRIOR = (2.0, -1.5, -0.5, 0.8, -1.0)
C_PRIOR = -2.0


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class FrozenMetaClassifier(Detector):
    """Supresor contextual con pesos congelados: out = b * sigmoid(w.phi + c)."""

    def __init__(
        self,
        detector: Detector,
        weights: tuple[float, ...] = W_PRIOR,
        bias: float = C_PRIOR,
        alpha: float = ALPHA,
        hi: float = HI,
        trigger: float = TRIGGER,
    ) -> None:
        if len(weights) != N_FEATURES:
            raise ValueError(f"weights debe tener {N_FEATURES} elementos.")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha debe estar en (0, 1].")
        if not (0.0 <= hi <= 1.0):
            raise ValueError("hi debe estar en [0.0, 1.0].")
        if not (0.0 <= trigger <= 1.0):
            raise ValueError("trigger debe estar en [0.0, 1.0].")
        self._det = detector
        self._w = tuple(float(x) for x in weights)
        self._c = float(bias)
        self._alpha = alpha
        self._hi = hi
        self._trigger = trigger
        self._t = 0
        self._vol = 0.0
        self._lvl = 0.0
        self._prev = 0.0
        self._t_last = -10**9

    @property
    def weights(self) -> tuple[float, ...]:
        return self._w

    @property
    def bias(self) -> float:
        return self._c

    def set_weights(self, weights: tuple[float, ...], bias: float) -> None:
        """Fija pesos entrenados offline (solo el harness, nunca en streaming)."""
        if len(weights) != N_FEATURES:
            raise ValueError(f"weights debe tener {N_FEATURES} elementos.")
        self._w = tuple(float(x) for x in weights)
        self._c = float(bias)

    def features(self, b: float) -> tuple[float, float, float, float, float]:
        """Vector phi (puro, testeable). Requiere estado actualizado hasta t-1."""
        d = max(-1.0, min(1.0, b - self._prev))
        tau = min(max(self._t - self._t_last, 0) / W_REF, 1.0)
        return (b, self._vol, d, tau, self._lvl)

    def verdict(self, phi: tuple[float, ...]) -> float:
        return sigmoid(sum(w * x for w, x in zip(self._w, phi)) + self._c)

    def update(self, value: float, timestamp: Any = "") -> float:
        b = self._det.update(value, timestamp)
        if b is None or math.isnan(b) or math.isinf(b):
            return 0.0
        b = max(0.0, min(1.0, float(b)))
        phi = self.features(b)
        v = self.verdict(phi)
        out = b * v
        # Estado DESPUES de puntuar (causalidad). Pesos intactos.
        hit = 1.0 if b >= self._hi else 0.0
        self._vol += self._alpha * (hit - self._vol)
        self._lvl += self._alpha * (b - self._lvl)
        self._prev = b
        if b >= self._trigger:
            self._t_last = self._t
        self._t += 1
        return max(0.0, min(1.0, out))

    def reset(self) -> None:
        self._t = 0
        self._vol = 0.0
        self._lvl = 0.0
        self._prev = 0.0
        self._t_last = -10**9
        inner = getattr(self._det, "reset", None)
        if callable(inner):
            inner()

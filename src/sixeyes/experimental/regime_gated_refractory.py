"""Refractario gated por regimen (1 intento pre-registrado, 2026-10-15).

Direccion: adaptar la DURACION del silencio, nunca la magnitud del score,
sin romper comparabilidad global ni despertar antes (rigidez protegida).

Diseno a priori (sin barrido):
  - Estado S_t = EWMA causal de excedencia de cola: hit=1 si raw>=hi(0.90).
    vol += alpha*(hit-vol), alpha=0.02 (~W efectiva 100). O(1).
  - Al disparar (raw>=trigger 0.990): W = 40 si vol<0.15 (calma),
    80 si <0.35 (base), 160 si >=0.35 (turbulencia).
  - Durante el silencio: se sigue actualizando vol con el raw subyacente
    (no con el 0.0 emitido) y JAMAS se despierta antes. Solo se elige W
    en el instante del disparo.

Correcciones al sketch original (documentadas, no silenciosas):
  1. gamma=0.98 como tasa EWMA es tasa instantanea (vol~=raw). Se usa
     alpha=0.02 (=1-gamma): memoria real ~100 pasos.
  2. EWMA de |raw| es inutil: el calibrador ECDF emite ~Uniforme[0,1]
     con media ~0.5, luego vol~0.5 siempre >=0.35 -> siempre W=160.
     Se trackea densidad de cola P(raw>=0.90): calma ~0.10, racha ~0.3+.

Propiedades: causalidad estricta, O(1) tiempo/memoria, reset-invariante.
"""

from __future__ import annotations

import math
from typing import Any

from sixeyes.detectors.base import Detector

W_CALM = 40
W_BASE = 80
W_TURB = 160
TRIGGER = 0.990
ALPHA = 0.02
HI = 0.90
LO_VOL = 0.15
HI_VOL = 0.35


class RegimeGatedRefractory(Detector):
    """Silencio post-pico con duracion modulada por turbulencia local."""

    def __init__(
        self,
        detector: Detector,
        w_calm: int = W_CALM,
        w_base: int = W_BASE,
        w_turb: int = W_TURB,
        trigger_threshold: float = TRIGGER,
        alpha: float = ALPHA,
        hi: float = HI,
        lo_vol: float = LO_VOL,
        hi_vol: float = HI_VOL,
    ) -> None:
        if w_calm < 0 or w_base < 0 or w_turb < 0:
            raise ValueError("w_* deben ser >= 0.")
        if not (0.0 <= trigger_threshold <= 1.0):
            raise ValueError("trigger_threshold debe estar en [0.0, 1.0].")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha debe estar en (0, 1].")
        if not (0.0 <= hi <= 1.0):
            raise ValueError("hi debe estar en [0.0, 1.0].")
        if not (0.0 <= lo_vol < hi_vol <= 1.0):
            raise ValueError("se requiere 0 <= lo_vol < hi_vol <= 1.")
        self._detector = detector
        self.w_calm = w_calm
        self.w_base = w_base
        self.w_turb = w_turb
        self.trigger_threshold = trigger_threshold
        self.alpha = alpha
        self.hi = hi
        self.lo_vol = lo_vol
        self.hi_vol = hi_vol
        self._t = 0
        self._silence_until = -1
        self._vol = 0.0
        self.last_w = 0  # auditoria: W elegida en el ultimo disparo

    def update(self, value: float, timestamp: Any = "") -> float:
        raw = self._detector.update(value, timestamp)
        if raw is None or math.isnan(raw) or math.isinf(raw):
            return 0.0
        raw = max(0.0, min(1.0, float(raw)))

        # 1. Estado de regimen con el raw SUBYACENTE (antes del silencio).
        hit = 1.0 if raw >= self.hi else 0.0
        self._vol += self.alpha * (hit - self._vol)

        t = self._t
        self._t += 1

        # 2. Rigidez: silencio en curso jamas se interrumpe.
        if t < self._silence_until:
            return 0.0

        # 3. Disparo: elegir W segun regimen, programar, emitir.
        if raw >= self.trigger_threshold:
            if self._vol < self.lo_vol:
                w = self.w_calm
            elif self._vol < self.hi_vol:
                w = self.w_base
            else:
                w = self.w_turb
            self.last_w = w
            if w > 0:
                self._silence_until = t + 1 + w
            return raw
        return raw

    def reset(self) -> None:
        self._t = 0
        self._silence_until = -1
        self._vol = 0.0
        self.last_w = 0
        inner = getattr(self._detector, "reset", None)
        if callable(inner):
            inner_reset = inner
            inner_reset()

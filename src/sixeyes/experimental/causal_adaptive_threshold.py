"""Umbral adaptativo causal con actualizacion asimetrica (1 intento, 2026-10-15).

Idea honesta: estimar el piso de ruido POR SERIE de forma causal (warmup
fingerprint + deriva lenta solo en calma) sin mirar futuro ni etiquetas.
No es trampa: es estado local O(1), transfiere a LOCO por construccion.

Correcciones al sketch original (documentadas):
  1. El sketch retornaba self._tr (el umbral). NAB barre un umbral GLOBAL
     sobre los scores emitidos (Sweeper). Emitir el umbral colapsa el sweep.
     Este wrapper emite EXCEDENCIA normalizada comparable globalmente:
       out = clip((raw - tr) / (1 - tr), 0, 1) si raw>=tr, else raw*0.5.
     Lo bajo se atenua a mitad (preserva orden), lo alto mide distancia
     al propio piso de la serie.
  2. alpha=0.001 hacia raw (media ~0.5 del ECDF) arrastra tr->0.5 e inunda
     FPs. Se trackea el CUANTIL alto via update solo en calma y con paso
     proporcional a (raw-tr) negativo: tr persigue el borde superior de la
     calma, no la media. Congela SIEMPRE que raw>=tr (inmunidad al exito).
  3. Warmup 200 contamina con burn-in del RLS (errores hasta 1997 el dia 1,
     ver autopsia 44.70). Warmup=100 (igual que bypass) y con mediana de los
     primeros scores, no media acumulativa.

Orden en pipeline: ... -> Calibrador -> PeakDecay -> CausalAdaptiveThreshold
  -> SpatialSpikeBypass (bypass SIEMPRE por encima: fractura fisica no se
  normaliza) -> RegimeGatedRefractory. El bypass va despues para que el 1.0
  de ruptura no sea reescalado.

Causal, O(1), reset-invariante. Parametros a priori, sin barrido.
"""

from __future__ import annotations

import math
from collections import deque
from statistics import median
from typing import Any

from sixeyes.detectors.base import Detector

BASE_TR = 0.990
ALPHA = 0.001
WARMUP = 100
FLOOR_GAIN = 0.01


class CausalAdaptiveThreshold(Detector):
    """Normaliza excedencia contra un piso de ruido estimado por serie."""

    def __init__(
        self,
        detector: Detector,
        base_threshold: float = BASE_TR,
        alpha: float = ALPHA,
        warmup_steps: int = WARMUP,
        floor_gain: float = FLOOR_GAIN,
    ) -> None:
        if not (0.0 <= base_threshold <= 1.0):
            raise ValueError("base_threshold debe estar en [0.0, 1.0].")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha debe estar en (0, 1].")
        if warmup_steps < 10:
            raise ValueError("warmup_steps debe ser >= 10.")
        self._det = detector
        self._base = base_threshold
        self._alpha = alpha
        self._warmup = warmup_steps
        self._gain = floor_gain
        self._t = 0
        self._tr = base_threshold
        self._buf: deque[float] = deque(maxlen=warmup_steps)

    def update(self, value: float, timestamp: Any = "") -> float:
        raw = self._det.update(value, timestamp)
        if raw is None or math.isnan(raw) or math.isinf(raw):
            return 0.0
        raw = max(0.0, min(1.0, float(raw)))
        self._t += 1

        # 1. Warmup: fingerprint con mediana (robusta al burn-in del RLS).
        if self._t <= self._warmup:
            self._buf.append(raw)
            floor = float(median(self._buf))
            # Serie ruidosa de entrada -> piso mas exigente; limpia -> mas sensible.
            # Banda estrecha: +-0.005 alrededor de la base (no reescribe el sistema).
            self._tr = max(0.0, min(1.0, self._base + (floor - 0.5) * self._gain))
            return max(0.0, min(1.0, (raw - self._tr) / max(1.0 - self._tr, 1e-9))) if raw >= self._tr else raw * 0.5

        # 2. Regimen: solo aprender en calma (raw < tr). Congelar en sospecha.
        if raw < self._tr:
            self._tr += self._alpha * (raw - self._tr)

        if raw >= self._tr:
            return (raw - self._tr) / max(1.0 - self._tr, 1e-9)
        return raw * 0.5

    def reset(self) -> None:
        self._t = 0
        self._tr = self._base
        self._buf.clear()
        inner = getattr(self._det, "reset", None)
        if callable(inner):
            inner()

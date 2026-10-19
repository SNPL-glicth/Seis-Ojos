"""Estimadores de escala de dispersión cinemática (Aislamiento Experimental).

Compara de manera controlada y sin alterar el código de producción:
1. WindowedMADEstimator: Estimador actual de ContinuousPlasticityController (ventana móvil, mediana de desviaciones absolutas).
   - Ventaja: Breakdown point del 50% (insensible a ráfagas de outliers aislados hasta W/2).
   - Inconveniente: Coste computacional O(W) con doble np.median y np.fromiter.
2. StreamingEWMAEstimator: Estimador recursivo de media y desviación absoluta media (EMA / L1).
   - Ventaja: O(1) tiempo y memoria, cero asignaciones de arrays.
   - Inconveniente: Breakdown point 0% (un outlier masivo infla inmediatamente la escala).
3. StreamingHuberEstimator: Estimador recursivo con corte robusto (Huber / Winsorizado).
   - Ventaja: O(1) streaming pero con saturación de outliers (|d - mu| <= k * s).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Protocol

import numpy as np

MAD_TO_SIGMA = 1.4826
NORMAL_MEAN_DEV_TO_SIGMA = 1.253314  # sqrt(pi / 2)


class ScaleEstimator(Protocol):
    """Protocolo de estimador de escala cinemática s_t."""

    def update(self, delta: float, prev_x: float) -> float:
        """Actualiza el estimador con d_t = x_t - x_{t-1} y retorna la escala s_t."""
        ...

    def reset(self) -> None:
        ...


class WindowedMADEstimator:
    """Implementación exacta actual de ContinuousPlasticityController._shock()."""

    def __init__(self, window: int = 50, min_history: int = 25) -> None:
        self.window = int(window)
        self.min_history = int(min_history)
        self._dbuf: deque[float] = deque(maxlen=self.window)

    def update(self, delta: float, prev_x: float) -> float:
        if len(self._dbuf) < self.min_history:
            self._dbuf.append(float(delta))
            return 1.0

        arr = np.fromiter(self._dbuf, dtype=float, count=len(self._dbuf))
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        s = max(
            MAD_TO_SIGMA * mad,
            float(np.mean(np.abs(arr))),
            1e-6 * max(1.0, abs(prev_x)),
        )
        self._dbuf.append(float(delta))
        return s

    def reset(self) -> None:
        self._dbuf.clear()


class StreamingEWMAEstimator:
    """Estimador puramente recursivo O(1) de desviación absoluta media."""

    def __init__(self, half_life: float = 25.0) -> None:
        self.half_life = float(half_life)
        self.alpha = 1.0 - 0.5 ** (1.0 / self.half_life)
        self._mu: float = 0.0
        self._s: float = 1.0
        self._count: int = 0

    def update(self, delta: float, prev_x: float) -> float:
        d = float(delta)
        self._count += 1
        if self._count <= 1:
            self._mu = d
            self._s = max(abs(d), 1e-4)
            return self._s

        # Escala previa estrictamente antes del incremento d_t (causalidad temporal)
        s_prior = max(self._s, 1e-6 * max(1.0, abs(prev_x)))

        err = d - self._mu
        # Actualización de media y escala L1 para el siguiente paso t+1
        self._mu += self.alpha * err
        abs_err = abs(err)
        self._s = (1.0 - self.alpha) * self._s + self.alpha * (abs_err * NORMAL_MEAN_DEV_TO_SIGMA)
        return s_prior

    def reset(self) -> None:
        self._mu = 0.0
        self._s = 1.0
        self._count = 0


class StreamingHuberEstimator:
    """Estimador recursivo O(1) con corte robusto (Huber clipping k=2.5 sigmas)."""

    def __init__(self, half_life: float = 25.0, clip_sigmas: float = 2.5) -> None:
        self.half_life = float(half_life)
        self.clip_sigmas = float(clip_sigmas)
        self.alpha = 1.0 - 0.5 ** (1.0 / self.half_life)
        self._mu: float = 0.0
        self._s: float = 1.0
        self._count: int = 0

    def update(self, delta: float, prev_x: float) -> float:
        d = float(delta)
        self._count += 1
        if self._count <= 1:
            self._mu = d
            self._s = max(abs(d), 1e-4)
            return self._s

        # Escala previa estrictamente antes del incremento d_t (causalidad temporal)
        s_prior = max(self._s, 1e-6 * max(1.0, abs(prev_x)))

        err = d - self._mu
        limit = self.clip_sigmas * self._s
        # Recorte robusto de Huber para que outliers masivos no contaminen s_{t+1}
        clipped_err = max(-limit, min(limit, err))
        self._mu += self.alpha * clipped_err
        
        abs_clipped = abs(clipped_err)
        # Factor de corrección asintótica de Huber para k=2.5
        huber_scale_factor = 1.28
        self._s = (1.0 - self.alpha) * self._s + self.alpha * (abs_clipped * huber_scale_factor)
        return s_prior

    def reset(self) -> None:
        self._mu = 0.0
        self._s = 1.0
        self._count = 0

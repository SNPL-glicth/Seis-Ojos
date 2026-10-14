"""Novedad de trayectoria latente (memoria densa continua + salida graduada).

El centroide recorre la trayectoria latente con media móvil exponencial
(EWMA, gamma configurable): es memoria de DÓNDE estuvo el sistema, no un
conteo de símbolos. Score = distancia euclídea del latente actual al
centroide: continuo, graduado, sin colapso a binario — el optimizador
opera en su zona útil.

Causalidad: puntúa contra el centroide del pasado y lo actualiza DESPUÉS.
O(m) por paso, memoria O(m). Sin futuro en ningún punto.
"""

import math

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.dense_arch.encoder import PhaseSpaceEncoder

EWMA_GAMMA = 0.99


class TrajectoryNovelty(Detector):
    """Distancia al centroide EWMA de la trayectoria latente (score ≥0)."""

    def __init__(self, gamma: float = EWMA_GAMMA, **encoder_kwargs) -> None:
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma debe estar en (0, 1).")
        self._gamma = gamma
        self._enc = PhaseSpaceEncoder(**encoder_kwargs)
        self._centroid: np.ndarray | None = None

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        z, ready = self._enc.update(value, timestamp)
        if not ready or z is None:
            return 0.0
        if self._centroid is None:
            self._centroid = z.copy()
            return 0.0
        score = float(np.linalg.norm(z - self._centroid))
        self._centroid = self._gamma * self._centroid + (1.0 - self._gamma) * z
        return max(score, 0.0)

    def reset(self) -> None:
        """Limpia encoder y centroide."""
        self._enc.reset()
        self._centroid = None


class DualTrajectoryNovelty(Detector):
    """Novedad dual: max(distancia a centroide rápido, a centroide lento).

    Análogo denso de la doble escala que sumó +5.17: el centroide rápido
    (γ=0.98, ~50 pts) reacciona a rupturas; el lento (γ=0.999, ~1000 pts)
    estabiliza el régimen. Exige rareza bajo ambas memorias.
    """

    def __init__(self, gamma_fast: float = 0.98, gamma_slow: float = 0.999,
                 **encoder_kwargs) -> None:
        for g in (gamma_fast, gamma_slow):
            if not 0.0 < g < 1.0:
                raise ValueError("gamma debe estar en (0, 1).")
        self._enc = PhaseSpaceEncoder(**encoder_kwargs)
        self._gf = gamma_fast
        self._gs = gamma_slow
        self._fast: np.ndarray | None = None
        self._slow: np.ndarray | None = None

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        z, ready = self._enc.update(value, timestamp)
        if not ready or z is None:
            return 0.0
        if self._fast is None:
            self._fast = z.copy()
            self._slow = z.copy()
            return 0.0
        score = max(float(np.linalg.norm(z - self._fast)),
                    float(np.linalg.norm(z - self._slow)))
        self._fast = self._gf * self._fast + (1.0 - self._gf) * z
        self._slow = self._gs * self._slow + (1.0 - self._gs) * z
        return max(score, 0.0)

    def reset(self) -> None:
        """Limpia encoder y ambos centroides."""
        self._enc.reset()
        self._fast = None
        self._slow = None

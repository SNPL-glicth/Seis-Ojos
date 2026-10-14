"""Fusión ortogonal pre-ECDF: sorpresa temporal ESN + novedad espacial centroide.

Difiere de la media falsada en dos puntos: (1) fusiona CRUDOS antes de UN
solo calibrador (la media unía salidas ya calibradas); (2) iguala escalas
primero —cada crudo dividido por su mediana trailing W=500— porque el error
ESN (colas hasta miles) dominaría a la distancia centroide por magnitud y
no se mediría nada.

    híbrido = 0.5·(esn/mediana_esn) + 0.5·(denso/mediana_denso)

Peso 0.5/0.5 y W=500 fijos a priori, sin barrer. Causalidad: medianas e
historias solo del pasado; ambas patas puntúan antes de actualizarse.
O(500) medianas + O(4096) RLS por paso.
"""

import math
import statistics
from collections import deque

from sixeyes.detectors.base import Detector
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.dense_arch.trajectory import TrajectoryNovelty

WEIGHT_ESN = 0.5
SCALE_WINDOW = 500
SCALE_MIN = 50


class OrthogonalHybrid(Detector):
    """Híbrido temporal+espacial con igualado robusto de escala."""

    def __init__(self, weight_esn: float = WEIGHT_ESN) -> None:
        self._w = weight_esn
        self._esn = EchoStatePredictor()
        self._dense = TrajectoryNovelty()
        self._he: deque[float] = deque(maxlen=SCALE_WINDOW)
        self._hd: deque[float] = deque(maxlen=SCALE_WINDOW)

    @staticmethod
    def _scale(hist: deque[float]) -> float:
        if len(hist) < SCALE_MIN:
            return 1.0
        med = statistics.median(hist)
        return med if med > 0.0 else 1.0

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        e = self._esn.update(value, timestamp)
        d = self._dense.update(value, timestamp)
        out = self._w * (e / self._scale(self._he)) + (1.0 - self._w) * (d / self._scale(self._hd))
        self._he.append(e)
        self._hd.append(d)
        return max(out, 0.0)

    def reset(self) -> None:
        """Reinicia patas e historiales de escala."""
        self._esn.reset()
        self._dense.reset()
        self._he.clear()
        self._hd.clear()

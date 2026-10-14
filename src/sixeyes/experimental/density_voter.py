"""Votante binario de densidad cruda (experimental).

Patrón portable de `histogram_bins` de skyline
(`NAB/nab/detectors/earthgecko_skyline/algorithms.py:226-251`) y de la pata
univariada de RCF (`shingleSize=1`): la rareza del VALOR CRUDO, no del
z-score. Detecta cambios de nivel que el z local normaliza y perdona.

Regla (idéntica a skyline): `tail_avg` (media de los últimos 3 puntos) cae en
un bin del histograma (15 bines) con ≤20 ocupantes → voto True.

Diferencias honestas vs skyline: historial acotado (500, no serie completa)
y sin acceso al futuro. Coste O(H)=O(500) por paso con `numpy.histogram`.
Causalidad estricta: el punto actual vota contra el histograma del pasado y
entra al buffer DESPUÉS.
"""

import math
from collections import deque

import numpy as np

from sixeyes.detectors.base import Detector

HISTORY = 500
N_BINS = 15
MIN_BIN_COUNT = 20
TAIL = 3


class HistogramDensityVoter(Detector):
    """Voto binario 0/1 por rareza de densidad del valor crudo."""

    def __init__(
        self,
        history: int = HISTORY,
        n_bins: int = N_BINS,
        min_bin_count: int = MIN_BIN_COUNT,
    ) -> None:
        self._history = history
        self._n_bins = n_bins
        self._min_bin = min_bin_count
        self._buf: deque[float] = deque(maxlen=history)

    def _vote(self, tail: float) -> float:
        series = np.array(self._buf, dtype=float)
        counts, edges = np.histogram(series, bins=self._n_bins)
        for index, bin_size in enumerate(counts):
            if bin_size <= self._min_bin:
                if index == 0:
                    if tail <= edges[0]:
                        return 1.0
                elif tail >= edges[index] and tail < edges[index + 1]:
                    return 1.0
        return 0.0

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        if len(self._buf) < TAIL + 1:
            self._buf.append(value)
            return 0.0
        tail = (self._buf[-1] + self._buf[-2] + self._buf[-3]) / 3.0
        vote = self._vote(tail)
        self._buf.append(value)
        return vote

    def reset(self) -> None:
        """Limpia el historial de densidad."""
        self._buf.clear()

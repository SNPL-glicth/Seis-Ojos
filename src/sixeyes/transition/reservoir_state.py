"""Estado latente del reservorio ESN (huella no lineal h_t en R^64).

Replica la dinamica del reservorio de EchoStatePredictor (mismas constantes
importadas, misma construccion de matrices con la misma semilla) pero SIN
lectura RLS: solo expone el estado. Normalizacion causal rodante identica
(W=6400, sumas O(1)). Causal estricto, O(N^2) flops/punto, memoria constante.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.detectors.delay_embedding import DelayCoordinateEmbedding
from sixeyes.detectors import echo_state as _es


class ReservoirState(Detector):
    """Huella de reservorio h_t normalizada causalmente (sin readout)."""

    def __init__(
        self,
        embed_dim: int = _es.EMBED_DIM,
        reservoir_size: int = _es.RESERVOIR_SIZE,
        spectral_radius: float = _es.SPECTRAL_RADIUS,
        seed: int = _es.SEED,
    ) -> None:
        self._n = reservoir_size
        self._embed = DelayCoordinateEmbedding(dimension=embed_dim, stride=1)
        rng = np.random.RandomState(seed)
        in_dim = embed_dim
        self._win = rng.uniform(-1.0, 1.0, size=(reservoir_size, in_dim)) * _es.INPUT_SCALE
        w = rng.uniform(-1.0, 1.0, size=(reservoir_size, reservoir_size))
        radius = max(abs(np.linalg.eigvals(w)).max(), 1e-9)
        self._wres = w * (spectral_radius / radius)
        self._h = np.zeros(reservoir_size)
        self._norm: deque[float] = deque(maxlen=_es.NORM_WINDOW)
        self._nsum = 0.0
        self._nsumsq = 0.0

    def _standardize(self, value: float) -> float:
        n = len(self._norm)
        if n < 2:
            return 0.0
        mean = self._nsum / n
        var = max(self._nsumsq / n - mean * mean, 0.0)
        std = math.sqrt(var)
        if std <= 1e-9:
            return 0.0
        return (value - mean) / std

    def _ingest_norm(self, value: float) -> None:
        if len(self._norm) == self._norm.maxlen:
            old = self._norm[0]
            self._nsum -= old
            self._nsumsq -= old * old
        self._norm.append(value)
        self._nsum += value
        self._nsumsq += value * value

    def update(self, value: float, timestamp: Any = "") -> tuple[np.ndarray | None, bool]:
        """Avanza el reservorio. Retorna (copia de h_t, listo?)."""
        if math.isnan(value) or math.isinf(value):
            return None, False
        z = self._standardize(value)
        vec, ready = self._embed.update(z, timestamp)
        if not ready or vec is None:
            self._ingest_norm(value)
            return None, False
        h_new = np.tanh(self._win @ vec + self._wres @ self._h)
        self._h = h_new
        self._ingest_norm(value)
        return h_new.copy(), True

    def reset(self) -> None:
        self._embed.reset()
        self._h = np.zeros(self._n)
        self._norm.clear()
        self._nsum = 0.0
        self._nsumsq = 0.0

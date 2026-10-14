"""Codificador continuo de espacio de fases (sin cuantización).

Vector de retardos (d=16) → proyección densa fija (Johnson-Lindenstrauss:
gaussiana/d con filas normalizadas, semilla fija) → latente en R^m.
La geometría se preserva en coma flotante: distancias relativas del
espacio de fases sobreviven a la proyección. Sin cajas, sin símbolos.
"""
import math

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.detectors.delay_embedding import DelayCoordinateEmbedding

EMBED_DIM = 16
LATENT_DIM = 32
SEED = 42


class PhaseSpaceEncoder:
    """Proyección densa fija de vectores de retardo (determinista)."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        latent_dim: int = LATENT_DIM,
        seed: int = SEED,
    ) -> None:
        self._embed = DelayCoordinateEmbedding(dimension=embed_dim, stride=1)
        rng = np.random.RandomState(seed)
        proj = rng.normal(0.0, 1.0, size=(latent_dim, embed_dim))
        proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12
        self._proj = proj
        self.latent_dim = latent_dim

    def update(self, value: float, timestamp: str = "") -> tuple[np.ndarray | None, bool]:
        """Devuelve (latente R^m, listo). None durante el warmup del embedding."""
        if math.isnan(value) or math.isinf(value):
            return None, False
        vec, ready = self._embed.update(value, timestamp)
        if not ready or vec is None:
            return None, False
        return self._proj @ vec, True

    def reset(self) -> None:
        """Limpia el embedding de retardos (la proyección es fija)."""
        self._embed.reset()

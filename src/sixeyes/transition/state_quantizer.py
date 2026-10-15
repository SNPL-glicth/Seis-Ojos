"""Cuantizador adaptativo de estados latentes (prototipos online).

Agrupa h_t en R^d en K prototipos con mini-k-means en streaming: asigna al
mas cercano (euclidea) y luego adapta ese prototipo hacia x con lr fijo.
Init determinista: los primeros K estados validos. Sin RNG, sin creacion de
clusters (acotado). Causal: asigna ANTES de adaptar.
"""

from __future__ import annotations

import numpy as np


class OnlinePrototypeQuantizer:
    """K prototipos adaptativos sobre vectores densos."""

    def __init__(self, n_states: int = 32, dim: int = 64, lr: float = 0.02) -> None:
        if n_states < 2:
            raise ValueError("n_states debe ser >= 2.")
        if lr <= 0.0 or lr >= 1.0:
            raise ValueError("lr debe estar en (0, 1).")
        self.n_states = n_states
        self.dim = dim
        self.lr = float(lr)
        self._protos: list[np.ndarray] = []
        self._counts: list[int] = [0] * n_states

    @property
    def ready(self) -> bool:
        return len(self._protos) >= self.n_states

    def assign(self, x: np.ndarray) -> int:
        """Indice del prototipo mas cercano (sin adaptar)."""
        d2 = [float(np.sum((p - x) ** 2)) for p in self._protos]
        return int(np.argmin(d2))

    def update(self, x: np.ndarray) -> int:
        """Asigna x al prototipo mas cercano y lo adapta. Retorna el indice."""
        x = np.asarray(x, dtype=float)
        if len(self._protos) < self.n_states:
            self._protos.append(x.copy())
            idx = len(self._protos) - 1
            self._counts[idx] += 1
            return idx
        idx = self.assign(x)
        self._protos[idx] += self.lr * (x - self._protos[idx])
        self._counts[idx] += 1
        return idx

    def reset(self) -> None:
        self._protos = []
        self._counts = [0] * self.n_states

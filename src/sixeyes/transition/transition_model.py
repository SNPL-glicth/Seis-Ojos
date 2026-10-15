"""Matriz de transicion de Markov de primer orden con suavizado Laplace.

Cuenta viajes A->B entre estados discretos en streaming. Probabilidad con
suavizado: P(j|i) = (C[i,j] + eps) / (fila_i + K*eps). Actualiza DESPUES de
puntuar (causalidad). Memoria O(K^2) constante.
"""

from __future__ import annotations

import numpy as np


class TransitionModel:
    """Conteos de transicion A->B con query de probabilidad suavizada."""

    def __init__(self, n_states: int = 32, epsilon: float = 1.0) -> None:
        if n_states < 2:
            raise ValueError("n_states debe ser >= 2.")
        if epsilon <= 0:
            raise ValueError("epsilon debe ser > 0.")
        self.n_states = n_states
        self.epsilon = float(epsilon)
        self._counts = np.zeros((n_states, n_states), dtype=float)

    def prob(self, prev: int, cur: int) -> float:
        row = float(self._counts[prev].sum())
        return (self._counts[prev, cur] + self.epsilon) / (row + self.n_states * self.epsilon)

    def observe(self, prev: int, cur: int) -> None:
        self._counts[prev, cur] += 1.0

    def reset(self) -> None:
        self._counts = np.zeros((self.n_states, self.n_states), dtype=float)

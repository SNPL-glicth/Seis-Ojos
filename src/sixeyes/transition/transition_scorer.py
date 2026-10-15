"""Sorpresa de transicion: score = -log P(S_t | S_{t-1}).

Pipeline interno por paso (todo causal, en este orden):
  1. h_t = reservorio(x_t) [None si embedding no listo -> score 0.0]
  2. S_t = argmin prototipos (pre-adaptacion)
  3. si no hay S_{t-1}: prev=S_t, adapta prototipo, score 0.0
  4. p = modelo.prob(prev, S_t); score = -log(p)
  5. modelo.observe(prev, S_t); adapta prototipo S_t; prev = S_t
"""

from __future__ import annotations

import math
from typing import Any

from sixeyes.detectors.base import Detector
from sixeyes.transition.reservoir_state import ReservoirState
from sixeyes.transition.state_quantizer import OnlinePrototypeQuantizer
from sixeyes.transition.transition_model import TransitionModel


class TransitionSurpriseDetector(Detector):
    """NLL de transicion sobre estados de reservorio cuantizados."""

    def __init__(
        self,
        n_states: int = 32,
        lr: float = 0.02,
        epsilon: float = 1.0,
        seed: int = 42,
    ) -> None:
        self._res = ReservoirState(seed=seed)
        self._quant = OnlinePrototypeQuantizer(n_states=n_states, dim=self._res._n, lr=lr)
        self._model = TransitionModel(n_states=n_states, epsilon=epsilon)
        self._prev: int | None = None

    def update(self, value: float, timestamp: Any = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        h, ready = self._res.update(value, timestamp)
        if not ready or h is None:
            return 0.0
        s = self._quant.update(h)
        if self._prev is None or not self._quant.ready:
            self._prev = s
            return 0.0
        p = self._model.prob(self._prev, s)
        score = -math.log(max(p, 1e-300))
        self._model.observe(self._prev, s)
        self._prev = s
        return max(score, 0.0)

    def reset(self) -> None:
        self._res.reset()
        self._quant.reset()
        self._model.reset()
        self._prev = None

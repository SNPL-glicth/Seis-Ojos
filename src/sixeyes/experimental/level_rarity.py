"""Rareza de nivel en streaming (soft bypass): p-valor de cola sobre histograma decaido.

Generalizacion del SpatialSpikeBypass. El bypass dice "salido del rango visto";
este detector dice "en una region del nivel rara, aunque dentro del rango".

Diseno (todo a priori, un solo intento):
- Histograma de K bins iguales sobre el rango del periodo de calentamiento
  (primeros ``warmup`` puntos) mas un margen relativo, con 2 bins de desborde
  en los extremos. Nada de rango global del archivo.
- Decaimiento exponencial con semivida ``half_life`` pasos.
- Score = p-valor de cola: 1 - (masa de bins igual o mas raros que el actual)
  / masa total. Casi uniforme bajo estacionariedad, no depende del numero
  de bins, umbral interpretable directo (0.99658 ~ p=0.34%).
- El bypass queda como caso limite: bin con masa 0 -> score 1.0.

Propiedades:
- Causalidad estricta: solo usa valores hasta t.
- Memoria acotada O(K), tiempo O(K) por paso.
- Invarianza reset().
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sixeyes.detectors.base import Detector


class LevelRarity(Detector):
    """P-valor de cola del nivel actual bajo un histograma decaido."""

    def __init__(
        self,
        n_bins: int = 80,
        half_life: float = 2000.0,
        warmup: int = 100,
        margin: float = 0.05,
    ) -> None:
        if n_bins < 4:
            raise ValueError("n_bins debe ser >= 4.")
        if half_life <= 0:
            raise ValueError("half_life debe ser > 0.")
        if warmup < 2:
            raise ValueError("warmup debe ser >= 2.")
        if margin < 0:
            raise ValueError("margin debe ser >= 0.")

        self.n_bins = n_bins
        self.half_life = float(half_life)
        self.warmup = warmup
        self.margin = margin

        self._warmup_vals: list[float] = []
        self._edges: np.ndarray | None = None
        self._mass: np.ndarray | None = None
        self._last_t: np.ndarray | None = None
        self._t: float = 0.0

    def _build_bins(self, vals: list[float]) -> None:
        vmin = min(vals)
        vmax = max(vals)
        rng = vmax - vmin
        if rng < 1e-12:
            rng = 1.0
        lo = vmin - self.margin * rng
        hi = vmax + self.margin * rng
        self._edges = np.linspace(lo, hi, self.n_bins + 1)
        self._mass = np.zeros(self.n_bins + 2, dtype=float)
        self._last_t = np.zeros(self.n_bins + 2, dtype=float)
        for v in vals:
            b = self._bin_index(float(v))
            self._mass[b] += 1.0
        self._t = 0.0

    def _bin_index(self, v: float) -> int:
        if v < self._edges[0]:
            return 0
        if v >= self._edges[-1]:
            return self.n_bins + 1
        return int(np.searchsorted(self._edges, v, side="right"))  # 1..n_bins

    def _effective_mass(self) -> np.ndarray:
        dt = self._t - self._last_t
        return self._mass * np.power(0.5, dt / self.half_life)

    def update(self, value: float, timestamp: Any = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0

        if self._edges is None:
            self._warmup_vals.append(float(value))
            if len(self._warmup_vals) >= self.warmup:
                self._build_bins(self._warmup_vals)
            return 0.0

        self._t += 1.0
        b = self._bin_index(float(value))

        eff = self._effective_mass()
        total = float(eff.sum())
        if total < 1e-12:
            self._mass[b] = 1.0
            self._last_t[b] = self._t
            return 0.0
        m_j = float(eff[b])
        rare_mass = float(eff[eff <= m_j].sum())
        score = 1.0 - rare_mass / total

        self._mass[b] = m_j + 1.0
        self._last_t[b] = self._t
        return score

    def reset(self) -> None:
        self._warmup_vals = []
        self._edges = None
        self._mass = None
        self._last_t = None
        self._t = 0.0

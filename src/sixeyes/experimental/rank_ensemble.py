"""Ensemble por rangos de réplicas ESN (experimental).

Fusión de las 3 semillas (42/7/99) por rangos causales, NO por media:
cada miembro puntúa su score crudo contra su propia cola causal
(ventana trailing W=16000, la misma del calibrador) y la salida es la
media de fracciones de rango. Sin constantes nuevas (W reutilizada).

Por qué rangos y no media: las 3 réplicas comparten escala pero difieren
en picos espurios por semilla; el rango acota la influencia de cualquier
pico individual (robusto), mientras la media lo hereda (frágil). Esto NO
es la fusión media falsada: allí se promediaban señales DISTINTAS; aquí
se vota la misma señal 3 veces.

Causalidad: rango contra pasado propio; warmup 288 en 0.0 (misma
filosofía que el calibrador). O(3) detectores + O(W) rangos por paso.
Salida en [0,1] lista para UN calibrador compartido aguas abajo.
"""

import math
from collections import deque

from sixeyes.detectors.base import Detector
from sixeyes.detectors.echo_state import EchoStatePredictor

SEEDS = (42, 7, 99)
RANK_WINDOW = 16000
RANK_WARMUP = 288


class RankEnsembleESN(Detector):
    """Media o mediana de rangos causales de 3 réplicas ESN."""

    def __init__(self, seeds: tuple[int, ...] = SEEDS, rank_window: int = RANK_WINDOW,
                 statistic: str = "mean") -> None:
        if statistic not in ("mean", "median"):
            raise ValueError("statistic debe ser 'mean' o 'median'.")
        self._members = [EchoStatePredictor(seed=s) for s in seeds]
        self._hists: list[deque[float]] = [deque(maxlen=rank_window) for _ in seeds]
        self._statistic = statistic

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        # Todos los miembros SIEMPRE actualizan su estado (sin cortocircuito).
        scores = [det.update(value, timestamp) for det in self._members]
        fracs = []
        for s, hist in zip(scores, self._hists):
            if len(hist) < RANK_WARMUP:
                hist.append(s)
                fracs.append(0.0)
                continue
            rank = sum(1 for x in hist if x <= s) / len(hist)
            hist.append(s)
            fracs.append(rank)
        if any(len(h) < RANK_WARMUP for h in self._hists):
            return 0.0
        if self._statistic == "median":
            ordered = sorted(fracs)
            fused = ordered[len(ordered) // 2]  # segundo mayor de 3
        else:
            fused = sum(fracs) / len(fracs)
        return min(max(fused, 0.0), 1.0)

    def reset(self) -> None:
        """Reinicia miembros e historiales de rango."""
        for det in self._members:
            det.reset()
        for hist in self._hists:
            hist.clear()

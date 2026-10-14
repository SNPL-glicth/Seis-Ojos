"""Sorpresa de transición markoviana (experimental).

Pregunta: "¿qué tan probable es la transición S_{t-1} → S_t?" en vez de
"¿qué tan raro es x_t?". Un flatline no es un valor raro pero sí una
secuencia de transiciones S→S inhabitual en régimen dinámico normal.

Especificación (fijada a priori, carta §2):
- Cuantización en K=8 símbolos por cotas fijas sobre z robusto:
  [-inf, -2, -1, -0.5, 0.5, 1, 2, inf]. Sin afinado in-sample.
- Matriz de conteo K×K con olvido λ=0.999 (adaptación lenta a deriva).
- Score = -log2 P̂(S_t|S_{t-1}) con suavizado de Laplace ε=1.0.
- z robusto causal interno (mediana/MAD, W=64, warmup 5).

Causalidad: z_t contra ventana de pasado; símbolo y sorpresa antes de
actualizar conteos e ingerir el valor. Memoria O(K²)=64 flotantes + O(64)
ventana z. Tiempo O(64) por paso (mediana de ventana corta).
Score crudo ≥0 no acotado (bits); aguas abajo lo normaliza el calibrador.
"""

import math
import statistics
from collections import deque

from sixeyes.detectors.base import Detector

K_SYMBOLS = 8
# Cotas fijas a priori (z-score), 7 cortes → 8 símbolos exactos con colas
# simétricas. NOTA: la especificación original listaba 6 cotas, lo que deja
# al símbolo 6 inalcanzable (7 regiones); se añade el corte en -3.0.
BOUNDS = (-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
FORGET_LAMBDA = 0.999
LAPLACE_EPS = 1.0
Z_WINDOW = 64
MIN_OBSERVATIONS = 5


def quantize(z: float) -> int:
    """Mapea z a símbolo 0..7 por cotas fijas (puro, testeable)."""
    for i, b in enumerate(BOUNDS):
        if z < b:
            return i
    return K_SYMBOLS - 1


class MarkovTransitionSurprise(Detector):
    """Sorpresa en bits de la transición de símbolos causales."""

    def __init__(
        self,
        forgetting: float = FORGET_LAMBDA,
        epsilon: float = LAPLACE_EPS,
        z_window: int = Z_WINDOW,
    ) -> None:
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting debe estar en (0, 1].")
        self._lam = forgetting
        self._eps = epsilon
        self._window: deque[float] = deque(maxlen=z_window)
        self._counts = [[0.0] * K_SYMBOLS for _ in range(K_SYMBOLS)]
        self._prev: int | None = None

    def _symbol(self, value: float) -> int | None:
        """Símbolo causal; dispersión cero + desvío = símbolo extremo."""
        if len(self._window) < MIN_OBSERVATIONS:
            return None
        mediana = statistics.median(self._window)
        mad = statistics.median([abs(x - mediana) for x in self._window])
        if 1.4826 * mad <= 0.0:
            if value == mediana:
                return quantize(0.0)
            return 0 if value < mediana else K_SYMBOLS - 1
        return quantize((value - mediana) / (1.4826 * mad))

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        sym = self._symbol(value)
        self._window.append(value)
        if sym is None:
            return 0.0
        prev, self._prev = self._prev, sym
        if prev is None:
            return 0.0
        row = self._counts[prev]
        total = sum(row) + K_SYMBOLS * self._eps
        prob = (row[sym] + self._eps) / total
        score = -math.log2(max(prob, 1e-300))
        # Actualizar conteos DESPUÉS de puntuar (causalidad).
        for j in range(K_SYMBOLS):
            row[j] *= self._lam
        row[sym] += 1.0
        return max(score, 0.0)

    def reset(self) -> None:
        """Limpia ventana z, matriz de conteos y símbolo previo."""
        self._window.clear()
        self._counts = [[0.0] * K_SYMBOLS for _ in range(K_SYMBOLS)]
        self._prev = None

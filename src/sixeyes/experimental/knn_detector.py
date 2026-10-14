"""k-NN streaming sobre vectores de retardo (experimental).

Flanco ortogonal: ni densidad puntual ni predicción paso a paso, sino FORMA
de la trayectoria en espacio de fases (patrón de `knncad`, techo NAB 57.99).
Un vector de retardo `[x_t, x_{t-1}, ..., x_{t-L+1}]` captura la firma
dinámica local; la distancia al k-ésimo vecino en un buffer acotado del
pasado mide rareza estructural (bucles extraños, transiciones de régimen).

Diseño (constantes a priori, carta §2):
- L=32, stride=1, buffer=1000 vectores, k=5, distancia euclídea.
- El vector actual consulta ANTES de ingresar al buffer (sin auto-match).
- Score crudo = distancia al k-ésimo vecino (no acotado ≥0); aguas abajo lo
  normaliza `PercentileCalibrator` como en el pipeline oficial.

Causalidad: embedding causal (`DelayCoordinateEmbedding`), consulta contra
pasado, ingesta después. Coste O(B·L)=O(32000) flops/punto + O(B) memoria
de vectores; viable en muestreos de minutos, documentado como no apto para
alta frecuencia sin LSH/proyección.
"""

import math

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.detectors.delay_embedding import DelayCoordinateEmbedding

DIMENSION = 32
STRIDE = 1
BUFFER_SIZE = 1000
K_NEIGHBORS = 5


class StreamingKNNDetector(Detector):
    """Distancia al k-ésimo vecino en espacio de fases (score crudo ≥0)."""

    def __init__(
        self,
        dimension: int = DIMENSION,
        stride: int = STRIDE,
        buffer_size: int = BUFFER_SIZE,
        k: int = K_NEIGHBORS,
    ) -> None:
        if k < 1:
            raise ValueError("k debe ser >= 1.")
        self._k = k
        self._buffer_size = buffer_size
        self._embed = DelayCoordinateEmbedding(dimension=dimension, stride=stride)
        self._buf: list[np.ndarray] = []

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        vec, ready = self._embed.update(value, timestamp)
        if not ready or vec is None:
            return 0.0
        if len(self._buf) < self._k:
            self._buf.append(vec.copy())
            return 0.0
        mat = np.stack(self._buf)
        dists = np.linalg.norm(mat - vec, axis=1)
        score = float(np.partition(dists, self._k - 1)[self._k - 1])
        self._buf.append(vec.copy())
        if len(self._buf) > self._buffer_size:
            self._buf.pop(0)
        return max(score, 0.0)

    def reset(self) -> None:
        """Limpia embedding y buffer de vecinos."""
        self._embed.reset()
        self._buf.clear()

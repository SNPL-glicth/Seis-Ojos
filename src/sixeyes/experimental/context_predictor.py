"""Predictor de próximo símbolo con contexto variable (experimental).

Responde al diagnóstico markoviano: Markov-1 fracasa porque un par aislado
no contiene física del sistema; HTM gana con memoria de SECUENCIAS. Este
módulo condiciona en el sufijo relevante (longitud adaptativa 0..D, estilo
PPM): la sorpresa nace cuando la SECUENCIA completa es imposible, aunque
cada par sea vulgar.

Diseño (constantes a priori, carta §2):
- Alfabeto K=8 con `quantize` y regla MAD=0 del módulo markoviano (reutilizada).
- Trie de contextos, profundidad máxima D=6, tope MAX_NODES=4096 nodos; al
  excederse se dividen todos los conteos (olvido global + poda de hojas a
  cero). Memoria acotada siempre.
- Predicción: contexto coincidente MÁS LARGO con conteo bruto ≥4 (si no,
  retroceso a más corto; orden 0 siempre vale). Laplace ε=1.0.
- Score = -log2 P̂ crudo en bits; lo normaliza el calibrador aguas abajo.

Causalidad: símbolo y sorpresa contra el trie del pasado; el trie se
actualiza DESPUÉS. O(D) por paso en inferencia y actualización.
"""

import math
from collections import deque

from sixeyes.detectors.base import Detector
from sixeyes.experimental.markov_transition import quantize, K_SYMBOLS
import statistics

MAX_DEPTH = 6
MAX_NODES = 4096
MIN_COUNT = 4
LAPLACE_EPS = 1.0
Z_WINDOW = 64
MIN_OBSERVATIONS = 5


class _Node:
    __slots__ = ("children", "counts", "total")

    def __init__(self) -> None:
        self.children: dict[int, "_Node"] = {}
        self.counts: list[float] = [0.0] * K_SYMBOLS
        self.total = 0.0


class VariableOrderPredictor(Detector):
    """Sorpresa del próximo símbolo dado el contexto variable."""

    def __init__(
        self,
        max_depth: int = MAX_DEPTH,
        max_nodes: int = MAX_NODES,
        min_count: int = MIN_COUNT,
        epsilon: float = LAPLACE_EPS,
    ) -> None:
        self._depth = max_depth
        self._max_nodes = max_nodes
        self._min_count = min_count
        self._eps = epsilon
        self._root = _Node()
        self._n_nodes = 1
        self._history: deque[int] = deque(maxlen=max_depth)
        self._z_window: deque[float] = deque(maxlen=Z_WINDOW)

    # -- simbolización causal (misma regla que markov_transition) --
    def _symbol(self, value: float) -> int | None:
        if len(self._z_window) < MIN_OBSERVATIONS:
            return None
        mediana = statistics.median(self._z_window)
        mad = statistics.median([abs(x - mediana) for x in self._z_window])
        if 1.4826 * mad <= 0.0:
            if value == mediana:
                return quantize(0.0)
            return 0 if value < mediana else K_SYMBOLS - 1
        return quantize((value - mediana) / (1.4826 * mad))

    # -- trie --
    def _find(self, context: tuple[int, ...]) -> _Node | None:
        node = self._root
        for s in context:
            node = node.children.get(s)
            if node is None:
                return None
        return node

    def _halve_and_prune(self) -> None:
        """Olvido global: divide conteos y poda hojas a cero (acota nodos)."""
        stack = [self._root]
        while stack:
            node = stack.pop()
            node.total = 0.0
            for s in list(node.children.keys()):
                child = node.children[s]
                for j in range(K_SYMBOLS):
                    child.counts[j] *= 0.5
                child.total = sum(child.counts)
                if child.total <= 1e-12 and not child.children:
                    del node.children[s]  # poda hoja vacía
                    continue
                node.total += child.total
                stack.append(child)
        self._recount()

    def _recount(self) -> None:
        count, stack = 0, [self._root]
        while stack:
            count += 1
            stack.extend(stack.pop().children.values())
        self._n_nodes = count

    def _predict_node(self) -> _Node:
        hist = tuple(self._history)
        for length in range(min(len(hist), self._depth), -1, -1):
            node = self._find(hist[len(hist) - length:] if length else ())
            if node is not None and (length == 0 or node.total >= self._min_count):
                return node
        return self._root

    def _ingest(self, sym: int) -> None:
        hist = tuple(self._history)
        node = self._root
        for length in range(1, min(len(hist), self._depth) + 1):
            s = hist[len(hist) - length]
            child = node.children.get(s)
            if child is None:
                child = _Node()
                node.children[s] = child
                self._n_nodes += 1
            node = child
            node.counts[sym] += 1.0
            node.total += 1.0
        self._root.counts[sym] += 1.0
        self._root.total += 1.0
        self._history.append(sym)
        if self._n_nodes > self._max_nodes:
            self._halve_and_prune()
            self._recount()

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        sym = self._symbol(value)
        self._z_window.append(value)
        if sym is None:
            return 0.0
        node = self._predict_node()
        prob = (node.counts[sym] + self._eps) / (node.total + K_SYMBOLS * self._eps)
        score = -math.log2(max(prob, 1e-300))
        self._ingest(sym)
        return max(score, 0.0)

    def reset(self) -> None:
        """Limpia trie, historial, ventana z y contador de nodos."""
        self._root = _Node()
        self._n_nodes = 1
        self._history.clear()
        self._z_window.clear()

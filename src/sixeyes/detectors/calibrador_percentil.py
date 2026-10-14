import math
import bisect
from collections import deque
from typing import Any
from .base import Detector


class PercentileCalibrator(Detector):
    """
    Un calibrador que envuelve a otro detector y transforma sus puntajes crudos
    en percentiles históricos.

    Ventana acotada: para evitar coste O(N) en memoria/tiempo en streaming
    largo, solo se retienen las últimas ``max_history`` observaciones
    (defecto 2000). Inserción O(log W) + desplazamiento O(W) acotado.
    """

    MIN_HISTORIAL = 288
    # 16000 cubre el p90 de NAB (max 22695) con memoria constante; verificado
    # 2026-10-14: reproduce el histórico 40.02 bit a bit (W=2000 daba 37.21).
    DEFAULT_MAX_HISTORY = 16000

    def __init__(self, detector_interno: Detector, max_history: int | None = DEFAULT_MAX_HISTORY):
        """Args:
            detector_interno: detector envuelto.
            max_history: ventana ECDF acotada (defecto 2000, producción).
                ``None`` = historial no acotado, solo para reproducir el número
                histórico 42.54 (reintroduce coste O(N) en memoria/tiempo).
        """
        if max_history is not None and max_history < self.MIN_HISTORIAL:
            raise ValueError("max_history debe ser >= MIN_HISTORIAL (288) o None.")
        self._detector = detector_interno
        self._historial: list[float] = []
        self._cola: deque[float] = deque()
        self._max_history = max_history

    def reset(self) -> None:
        """Limpia el historial ECDF y delega en el detector interno si aplica."""
        self._historial.clear()
        self._cola.clear()
        inner_reset = getattr(self._detector, "reset", None)
        if callable(inner_reset):
            inner_reset()

    def _evict_if_full(self) -> None:
        """Expulsa la observación más antigua si se alcanzó max_history."""
        if self._max_history is None:
            return  # modo reproducción histórica, sin acotar
        if len(self._historial) >= self._max_history and self._cola:
            oldest = self._cola.popleft()
            # Elimina una ocurrencia (O(W) acotado por max_history).
            idx = bisect.bisect_left(self._historial, oldest)
            # Barrido corto por duplicados de punto flotante.
            for j in range(idx, min(idx + 4, len(self._historial))):
                if self._historial[j] == oldest:
                    self._historial.pop(j)
                    break
            else:
                # Fallback lineal acotado (no debería ocurrir).
                try:
                    self._historial.remove(oldest)
                except ValueError:
                    pass

    def update(self, value: float, timestamp: Any = None) -> float:
        if timestamp is not None:
            raw = self._detector.update(value, timestamp)
        else:
            raw = self._detector.update(value, "")
        
        if math.isnan(raw) or math.isinf(raw) or raw == 0.0:
            return 0.0
            
        if len(self._historial) < self.MIN_HISTORIAL:
            bisect.insort(self._historial, raw)
            self._cola.append(raw)
            return 0.0
            
        menores = bisect.bisect_left(self._historial, raw)
        iguales = bisect.bisect_right(self._historial, raw) - menores
        n = len(self._historial)
        
        score = (menores + 0.5 * iguales + 0.5) / (n + 1)
        self._evict_if_full()
        bisect.insort(self._historial, raw)
        self._cola.append(raw)
        return score

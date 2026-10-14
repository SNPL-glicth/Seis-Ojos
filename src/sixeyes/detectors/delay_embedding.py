"""Módulo de incrustación causal en espacio de retardos (Takens Delay Coordinates)."""

from collections import deque
import math
from typing import Optional, Tuple
import numpy as np


class DelayCoordinateEmbedding:
    """Transformador causal de incrustación de coordenadas de retardo (Delay Coordinates).

    Mapea un flujo escalar causal x_t a una trayectoria vectorial en R^L:
        x_t = [x_t, x_{t - tau}, x_{t - 2*tau}, ..., x_{t - (L-1)*tau}]^T in R^L

    Propiedades matemáticas:
    1. Causalidad estricta y complejidad O(1):
       Utiliza un buffer circular en memoria de tamano exacto (L-1)*tau + 1.
    2. Cero amplificación de ruido observacional:
       No aplica diferencias finitas ni filtros pasa-altas. Cada coordenada
       mantiene su varianza individual Var(x_{t-k}) = sigma^2 (a diferencia de
       Delta^2 x_t que amplifica la varianza por un factor de 6).
    3. Firma invariante de congelamiento (Flatline):
       Cuando x_t = C constante, x_t colapsa exactamente a la diagonal C * 1_L,
       cuya dispersión transversal respecto a span(1_L) es exactamente 0.0.
    4. Normalización isotrópica robusta opcional:
       Estandariza mediante mediana y MAD sobre ventana móvil para evitar que
       escalas asimétricas distorsionen las distancias en R^L.
    """

    def __init__(
        self,
        dimension: int = 4,
        stride: int = 1,
        normalize: bool = False,
        mad_window: int = 50,
    ) -> None:
        """Inicializa el contenedor de incrustación de retardos.

        Args:
            dimension: Dimensión L del espacio de fase embebido (L >= 2).
            stride: Paso de retardo temporal tau (tau >= 1).
            normalize: Si es True, normaliza los valores usando mediana y MAD local.
            mad_window: Tamaño de la ventana para estimadores de mediana y MAD.
        """
        if dimension < 2:
            raise ValueError("La dimensión de incrustación L debe ser mayor o igual a 2.")
        if stride < 1:
            raise ValueError("El paso de retardo (stride) debe ser mayor o igual a 1.")

        self.L = dimension
        self.stride = stride
        self.normalize = normalize

        self._buffer_len = (self.L - 1) * self.stride + 1
        self._buffer: deque[float] = deque(maxlen=self._buffer_len)
        self._count = 0

        self._mad_window = mad_window
        self._val_history: deque[float] = deque(maxlen=mad_window) if normalize else deque()

    @property
    def is_ready(self) -> bool:
        """Indica si el buffer ha acumulado las observaciones mínimas de calentamiento."""
        return len(self._buffer) >= self._buffer_len

    def update(self, value: float, timestamp: str = "") -> Tuple[Optional[np.ndarray], bool]:
        """Procesa una nueva observación escalar entrante en tiempo real.

        Args:
            value: Observación escalar x_t.
            timestamp: Marca temporal asociada (opcional).

        Returns:
            Tuple[Optional[np.ndarray], bool]: (vector_R_L, is_ready)
            - vector_R_L: Array de tamaño (L,) con [x_t, x_{t-tau}, ..., x_{t-(L-1)tau}].
              Es None si el contenedor está en fase de calentamiento.
            - is_ready: True si el buffer está lleno y el vector es válido.
        """
        if math.isnan(value) or math.isinf(value):
            return None, False

        self._buffer.append(value)
        self._count += 1

        if self.normalize:
            self._val_history.append(value)

        if not self.is_ready:
            return None, False

        buf_list = list(self._buffer)
        vec = np.array([buf_list[-(1 + i * self.stride)] for i in range(self.L)], dtype=np.float64)

        if self.normalize and len(self._val_history) >= 5:
            med = float(np.median(self._val_history))
            mad = float(np.median(np.abs(np.array(self._val_history) - med))) * 1.4826
            if mad > 1e-8:
                vec = (vec - med) / mad

        return vec, True

    def reset(self) -> None:
        """Reinicia el estado del buffer interno."""
        self._buffer.clear()
        self._val_history.clear()
        self._count = 0
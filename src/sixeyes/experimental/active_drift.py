"""Detector de Deriva de Centroide Activo para Régimen Intermitente.

Detecta cambios abruptos en la mediana de la componente activa (no-reposo)
de una señal intermitente, ignorando estrictamente los periodos de reposo.
"""

from collections import deque
import math
from typing import Optional
import numpy as np

from sixeyes.detectors.base import Detector


class ActiveCentroidDriftTracker(Detector):
    """Rastreador de deriva de centroide en la componente activa."""

    def __init__(
        self,
        w_active: int = 30,
        eps_base: float = 1e-4,
    ) -> None:
        """Inicializa el rastreador de deriva de centroide activo.

        Args:
            w_active: Tamaño de la ventana para la mediana móvil activa.
            eps_base: Umbral por debajo del cual una observación se considera reposo.
        """
        if w_active < 2:
            raise ValueError("w_active debe ser mayor o igual a 2.")

        self.w_active = w_active
        self.eps_base = eps_base

        self._active_window: deque[float] = deque(maxlen=w_active)
        self._prev_median: Optional[float] = None

    def update(self, value: float, timestamp: str = "") -> float:
        """Actualiza el rastreador y retorna magnitud de deriva del centroide activo.

        Args:
            value: Observación escalar x_t.
            timestamp: Marca temporal (no usada, por compatibilidad).

        Returns:
            float: Magnitud del shift del centroide activo (0.0 en reposo).
        """
        if math.isnan(value) or math.isinf(value):
            return 0.0

        # En reposo: no actualizar ventana, emitir 0.0 estricto
        if abs(value) <= self.eps_base:
            return 0.0

        # Componente activa: añadir a ventana
        self._active_window.append(value)

        # Calcular shift de mediana
        if len(self._active_window) < 2:
            return 0.0

        current_median = float(np.median(self._active_window))
        
        if self._prev_median is None:
            self._prev_median = current_median
            return 0.0

        shift = abs(current_median - self._prev_median)
        self._prev_median = current_median

        # Normalización robusta: shift / (MAD + eps)
        if len(self._active_window) >= 3:
            mad = float(np.median(np.abs(np.array(self._active_window) - current_median))) * 1.4826
            return float(shift / max(mad, 1e-8))
        
        return 0.0

    def reset(self) -> None:
        """Reinicia el estado del rastreador."""
        self._active_window.clear()
        self._prev_median = None
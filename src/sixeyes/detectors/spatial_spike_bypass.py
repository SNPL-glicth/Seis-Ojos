"""Bypass Espacial de Spikes Históricos (Spatial Outlier Override).

Mecánica inspirada en Numenta HTM (numenta_detector.py):
Mantiene causally los extremos observados [min_val, max_val] a lo largo del stream.
Si una observación x_t rompe el rango histórico por más de una tolerancia relativa
(ej. 5% de la amplitud histórica total), se declara inmediatamente como anomalía
espacial pura (score = 1.0), sorteando la latencia de asimilación del embedding
de retardos y del reservorio.

Propiedades:
- Causalidad estricta O(1).
- Cero parámetros de entrenamiento.
- Soporte completo para streaming y reset().
"""

from __future__ import annotations
import math
from typing import Any

from sixeyes.detectors.base import Detector

DEFAULT_SPATIAL_TOLERANCE = 0.05
DEFAULT_WARMUP = 100


class SpatialSpikeBypass(Detector):
    """Bypass causal de picos espaciales extremos fuera de la envolvente histórica."""

    def __init__(
        self,
        detector: Detector,
        tolerance: float = DEFAULT_SPATIAL_TOLERANCE,
        warmup: int = DEFAULT_WARMUP,
    ) -> None:
        """Inicializa el bypass espacial.

        Args:
            detector: Detector subyacente envuelto.
            tolerance: Fracción fuera del rango histórico para declarar anomalía espacial (defecto 0.05).
            warmup: Número mínimo de observaciones iniciales para fijar el rango base.
        """
        if tolerance < 0.0:
            raise ValueError("tolerance debe ser >= 0.0.")
        if warmup < 1:
            raise ValueError("warmup debe ser >= 1.")

        self._detector = detector
        self.tolerance = tolerance
        self.warmup = warmup

        self._min_val: float | None = None
        self._max_val: float | None = None
        self._count: int = 0

    def update(self, value: float, timestamp: Any = "") -> float:
        """Procesa una observación causalmente.

        Si value rompe los extremos acumulados tras el periodo de warmup,
        emite 1.0 inmediatamente; de lo contrario delega en el detector envuelto.
        """
        if math.isnan(value) or math.isinf(value):
            return self._detector.update(value, timestamp)

        spatial_anomaly = False

        if self._count >= self.warmup and self._min_val is not None and self._max_val is not None:
            span = self._max_val - self._min_val
            if span > 1e-12:
                margin = span * self.tolerance
                if value > self._max_val + margin or value < self._min_val - margin:
                    spatial_anomaly = True

        # Actualizar cotas históricas causales
        if self._min_val is None or value < self._min_val:
            self._min_val = value
        if self._max_val is None or value > self._max_val:
            self._max_val = value

        self._count += 1

        base_score = self._detector.update(value, timestamp)

        if spatial_anomaly:
            return 1.0

        return base_score

    def reset(self) -> None:
        """Reinicia los extremos históricos y el detector interno."""
        self._min_val = None
        self._max_val = None
        self._count = 0
        inner_reset = getattr(self._detector, "reset", None)
        if callable(inner_reset):
            inner_reset()

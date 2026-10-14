"""Re-export de compatibilidad: la doble escala vive en `sixeyes.detectors.dual_scale`.
Este módulo se conserva para no romper imports.
"""

from sixeyes.detectors.dual_scale import (
    DualScaleRobustZ,
    SeasonalDualScaleRobustZ,
    FAST_WINDOW,
    SLOW_WINDOW,
    MIN_OBSERVATIONS,
    MIN_SLOW,
)

__all__ = [
    "DualScaleRobustZ",
    "SeasonalDualScaleRobustZ",
    "FAST_WINDOW",
    "SLOW_WINDOW",
    "MIN_OBSERVATIONS",
    "MIN_SLOW",
]

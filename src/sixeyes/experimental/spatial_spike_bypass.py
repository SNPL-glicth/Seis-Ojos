"""Compatibilidad hacia atrás: SpatialSpikeBypass promovido a sixeyes.detectors."""

from sixeyes.detectors.spatial_spike_bypass import (
    SpatialSpikeBypass,
    DEFAULT_SPATIAL_TOLERANCE,
    DEFAULT_WARMUP,
)

__all__ = ["SpatialSpikeBypass", "DEFAULT_SPATIAL_TOLERANCE", "DEFAULT_WARMUP"]

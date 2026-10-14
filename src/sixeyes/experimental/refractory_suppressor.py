"""Compatibilidad hacia atrás: RefractorySuppressor promovido a sixeyes.detectors."""

from sixeyes.detectors.refractory_suppressor import (
    RefractorySuppressor,
    DEFAULT_W_REST,
    DEFAULT_TRIGGER_THRESHOLD,
)

__all__ = ["RefractorySuppressor", "DEFAULT_W_REST", "DEFAULT_TRIGGER_THRESHOLD"]

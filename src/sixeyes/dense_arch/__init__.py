"""Arquitectura densa (proyecto nuevo, separado del pipeline oficial).

Principios: geometría continua (cero cuantización), salida graduada
causal, guardarraíles estrictos O(1) por paso. No forma parte del
pipeline NAB estándar hasta medición con gate.
"""

from sixeyes.dense_arch.encoder import PhaseSpaceEncoder
from sixeyes.dense_arch.trajectory import TrajectoryNovelty, DualTrajectoryNovelty

__all__ = [
    "PhaseSpaceEncoder",
    "TrajectoryNovelty",
    "DualTrajectoryNovelty",
]

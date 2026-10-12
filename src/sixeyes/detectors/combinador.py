"""Módulo con detectores combinados (ensembles)."""

import math
from typing import Any, List

from .base import Detector
from .calibrador_percentil import PercentileCalibrator
from .zscore_robusto import RollingRobustZ
from .subespacio import SubspaceResidualDetector


class MaxOfDetectors(Detector):
    """Combinador que ejecuta múltiples detectores y devuelve el puntaje máximo.

    Llama a update() en TODOS los detectores miembros secuencialmente (sin
    cortocircuito) para asegurar que todos actualicen su estado interno en cada paso.

    Si un miembro devuelve NaN o inf, cuenta como 0.0.
    """

    def __init__(self, detectores: List[Detector]):
        """Inicializa el combinador con una lista de detectores.

        Args:
            detectores: Lista de detectores que cumplen la interfaz Detector.
        """
        self.detectores = list(detectores)

    def update(self, value: float, timestamp: Any = None) -> float:
        """Actualiza todos los detectores y devuelve el puntaje máximo obtenido.

        Todos los detectores se ejecutan sin importar los valores que devuelvan,
        asegurando que sus estados internos y ventanas se actualicen en cada llamada.
        Si un miembro devuelve NaN o inf, ese miembro cuenta como 0.0.

        Args:
            value: Valor actual de la serie.
            timestamp: Timestamp o índice del paso actual.

        Returns:
            El score máximo (en [0, 1] si los detectores miembros emiten en [0, 1]).
        """
        max_score = 0.0
        for det in self.detectores:
            if timestamp is not None:
                score = det.update(value, timestamp)
            else:
                score = det.update(value, "")
            if math.isnan(score) or math.isinf(score):
                score = 0.0
            if score > max_score:
                max_score = score
        return float(max_score)


def Combinado() -> MaxOfDetectors:
    """Registra y construye la instancia de 'Combinado' con parámetros por defecto.

    Combina mediante MaxOfDetectors:
    1. PercentileCalibrator(RollingRobustZ())
    2. PercentileCalibrator(SubspaceResidualDetector())
    """
    return MaxOfDetectors([
        PercentileCalibrator(RollingRobustZ()),
        PercentileCalibrator(SubspaceResidualDetector()),
    ])

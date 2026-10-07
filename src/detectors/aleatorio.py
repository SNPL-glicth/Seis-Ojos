"""Detector de anomalías aleatorio."""

import random

DEFAULT_SEED: int = 42


class RandomDetector:
    """Detector de línea base que asigna scores aleatorios uniformes en [0, 1]."""

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        # Se aísla el generador en una instancia propia para no contaminar
        # ni depender del estado global de random en la aplicación.
        self._rng = random.Random(seed)

    def update(self, value: float, timestamp: str) -> float:
        """Devuelve un score en [0, 1], mayor valor significa más anómalo;
        el detector solo puede usar el pasado y el punto actual."""
        return self._rng.uniform(0.0, 1.0)

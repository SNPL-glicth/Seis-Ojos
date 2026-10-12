from typing import Protocol, runtime_checkable


# runtime_checkable permite verificar el cumplimiento del protocolo con isinstance en pruebas.
@runtime_checkable
class Detector(Protocol):
    """Protocolo base para detectores de anomalías en streaming."""

    def update(self, value: float, timestamp: str) -> float:
        """Devuelve un score en [0, 1], mayor valor significa más anómalo; el detector solo puede usar el pasado y el punto actual."""
        ...

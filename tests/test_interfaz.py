"""Pruebas de la interfaz base para detectores."""

from sixeyes.detectors.base import Detector


class DetectorEjemplo:
    """Implementación mínima de ejemplo para validar el cumplimiento del protocolo."""

    def update(self, value: float, timestamp: str) -> float:
        return 0.0


def test_clase_ejemplo_cumple_interfaz_detector() -> None:
    detector = DetectorEjemplo()
    assert isinstance(detector, Detector)

    score = detector.update(1.23, "2026-10-05T00:00:00Z")
    assert 0.0 <= score <= 1.0

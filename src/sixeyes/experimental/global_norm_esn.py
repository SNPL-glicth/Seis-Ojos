"""ESN con normalización min/max global (VARIANTE NAB-INFORMADA, experimental).

NO causal: usa mínimo/máximo del archivo completo (información de futuro,
igual que ARTime/HTM reciben). Existe SOLO para medir cuánta brecha es
información vs arquitectura. Jamás en producción streaming real; la vía
honesta derivada sería rango estimado en probatorio + ampliación en línea.
"""
from sixeyes.detectors.echo_state import EchoStatePredictor


class GlobalNormESN(EchoStatePredictor):
    """EchoStatePredictor con escala min/max global prefijada por archivo."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._lo: float | None = None
        self._hi: float | None = None

    def set_range(self, lo: float, hi: float) -> None:
        """Fija el rango global (llamar antes de stremear el archivo)."""
        self._lo, self._hi = lo, hi

    def _standardize(self, value: float) -> float:
        if self._lo is None or self._hi is None or self._hi <= self._lo:
            return super()._standardize(value)
        return (value - self._lo) / (self._hi - self._lo) * 2.0 - 1.0

    def _ingest_norm(self, value: float) -> None:
        if self._lo is not None and self._hi is not None and self._hi > self._lo:
            return  # sin estado rodante en modo global
        super()._ingest_norm(value)

    def reset(self) -> None:
        super().reset()
        self._lo, self._hi = None, None

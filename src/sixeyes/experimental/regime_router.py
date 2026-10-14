"""Enrutado por régimen observable (experimental).

Reglas congeladas en `aprendizaje.md` ANTES del código (contrato):
- R1: dt_mediano (primeros 50 timestamps) ≥ 45 min → ruta predictor.
- R2: en otro caso → ruta base oficial intacta.

El router conmuta la FUENTE (Dual vs Predictive, ambas [0,1]) bajo UN solo
calibrador y UN PeakDecay aguas abajo: la decisión es discreta y congelada
por serie (determinista, reset-invariante). Durante la estimación se
alimenta la base (los scores tempranos los anula el warmup del calibrador).

Estatus honesto: IN-SAMPLE declarado. Una sola corrida, sin iterar
constantes (45 min / 50 pts fijados a priori).
"""

from datetime import datetime

from sixeyes.detectors.base import Detector
from sixeyes.detectors.dual_scale import SeasonalDualScaleRobustZ
from sixeyes.experimental.predictor import PredictiveDetector

DT_THRESHOLD_MIN = 45
ESTIMATION_POINTS = 50


def _parse_epoch(timestamp: str) -> float | None:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace(" ", "T")).timestamp()
    except ValueError:
        pass
    try:
        from dateutil import parser as date_parser
        return date_parser.parse(timestamp).timestamp()
    except Exception:
        return None


class RegimeRouter(Detector):
    """Conmuta base/predictor por dt mediano congelado por serie."""

    def __init__(
        self,
        dt_threshold_min: float = DT_THRESHOLD_MIN,
        estimation_points: int = ESTIMATION_POINTS,
        bucket_size_minutes: int = 15,
    ) -> None:
        self._thr = dt_threshold_min
        self._est = estimation_points
        self._base = SeasonalDualScaleRobustZ(bucket_size_minutes=bucket_size_minutes)
        self._pred = PredictiveDetector()
        self._epochs: list[float] = []
        self._n = 0
        self._use_predictor: bool | None = None

    @property
    def route(self) -> str | None:
        """'predictor' | 'base' | None (sin decidir)."""
        if self._use_predictor is None:
            return None
        return "predictor" if self._use_predictor else "base"

    def _freeze(self) -> None:
        diffs = [b - a for a, b in zip(self._epochs, self._epochs[1:]) if b > a]
        if not diffs:
            self._use_predictor = False
            return
        import statistics
        dt_min = statistics.median(diffs) / 60.0
        self._use_predictor = dt_min >= self._thr

    def update(self, value: float, timestamp: str = "") -> float:
        import math
        if math.isnan(value) or math.isinf(value):
            return 0.0
        if self._use_predictor is None:
            epoch = _parse_epoch(timestamp)
            if epoch is not None:
                self._epochs.append(epoch)
            self._n += 1
            score = self._base.update(value, timestamp)
            if self._n >= self._est:
                self._freeze()
            return score
        if self._use_predictor:
            return self._pred.update(value, timestamp)
        return self._base.update(value, timestamp)

    def reset(self) -> None:
        """Limpia estimación, decisión y ambas rutas."""
        self._epochs.clear()
        self._n = 0
        self._use_predictor = None
        self._base.reset()
        self._pred.reset()

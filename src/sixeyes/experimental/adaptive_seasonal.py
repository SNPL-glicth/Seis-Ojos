"""Estacionalidad con período adaptativo (experimental).

Motivación: ``DailySeasonalRobustZ`` fija 96 buckets de 15 min, lo que asume
muestreo ~5 min y período diario. En series horarias solo 24 de 96 buckets
reciben 1 punto/día (72 buckets muertos para siempre); en series de 1 min el
período real es 1440 pasos, no 96.

Diseño: warmup causal corto (12 puntos) estima la mediana de ``dt`` entre
timestamps consecutivos y fija ``bucket_minutes = max(15, round(dt_min))`` con
``m = max(8, round(1440 / bucket_minutes))`` buckets. Doce puntos bastan para
series regulares y minimizan la perturbación del fallback (la variante con
warmup 200 perdía −3.94 pts globales por puntuar el arranque sin
estacionalidad).

Causalidad estricta: ``m`` se fija una sola vez con los primeros
``warmup_points`` timestamps; ningún punto ve el futuro. Los buckets internos
reutilizan ``DailySeasonalRobustZ`` (código probado).
"""

from datetime import datetime
import statistics

from sixeyes.detectors.base import Detector
from sixeyes.detectors.seasonal import DailySeasonalRobustZ
from sixeyes.detectors.zscore_robusto import RollingRobustZ

WARMUP_POINTS = 12
MIN_BUCKET_MINUTES = 15
MIN_BUCKETS = 8
MAX_BUCKETS = 1440
FALLBACK_BUCKETS_15MIN = 96


def _parse_epoch(timestamp: str) -> float | None:
    """Convierte un timestamp a segundos epoch. None si no parseable."""
    if not timestamp:
        return None
    dt_str = timestamp.replace(" ", "T")
    try:
        return datetime.fromisoformat(dt_str).timestamp()
    except ValueError:
        pass
    try:
        from dateutil import parser as date_parser
        return date_parser.parse(timestamp).timestamp()
    except Exception:
        return None


class AdaptiveSeasonalRobustZ(Detector):
    """Z-score robusto estacional con período estimado causalmente."""

    def __init__(
        self,
        warmup_points: int = WARMUP_POINTS,
        window_size: int = 20,
        min_observations: int = 5,
    ) -> None:
        self._warmup_points = warmup_points
        self._window_size = window_size
        self._min_obs = min_observations
        self._epochs: list[float] = []
        self._n_updates = 0
        self._bucket_minutes: int | None = None
        self._fallback_only = False
        self._fallback = RollingRobustZ(
            window_size=window_size, min_observations=min_observations
        )
        self._seasonal: DailySeasonalRobustZ | None = None

    @property
    def bucket_minutes(self) -> int | None:
        """Minutos por bucket fijados tras el warmup (None durante warmup)."""
        return self._bucket_minutes

    @property
    def bucket_count(self) -> int:
        """Número de buckets (0 durante warmup)."""
        if self._bucket_minutes is None:
            return 0
        return max(MIN_BUCKETS, min(MAX_BUCKETS, round(1440 / self._bucket_minutes)))

    def _fix_period(self) -> None:
        """Fija el período una sola vez a partir de la mediana de dt.

        Sin epochs válidos: fallback global permanente (el bucketing de
        ``DailySeasonalRobustZ`` no acepta timestamps vacíos).
        """
        diffs = [
            b - a for a, b in zip(self._epochs, self._epochs[1:]) if b > a
        ]
        if not diffs:
            self._bucket_minutes = MIN_BUCKET_MINUTES
            if not self._epochs:
                self._fallback_only = True
                return
        else:
            dt_min = statistics.median(diffs) / 60.0
            self._bucket_minutes = max(MIN_BUCKET_MINUTES, round(dt_min))
        self._seasonal = DailySeasonalRobustZ(
            bucket_size_minutes=self._bucket_minutes,
            window_size=self._window_size,
            min_observations=self._min_obs,
        )

    def update(self, value: float, timestamp: str = "") -> float:
        # Warmup: estimar dt y puntuar con el fallback global.
        if self._seasonal is None and not self._fallback_only:
            epoch = _parse_epoch(timestamp)
            if epoch is not None:
                self._epochs.append(epoch)
            self._n_updates += 1
            score = self._fallback.update(value, timestamp)
            if self._n_updates >= self._warmup_points:
                self._fix_period()
            return score
        if self._fallback_only:
            return self._fallback.update(value, timestamp)
        return self._seasonal.update(value, timestamp)

    def reset(self) -> None:
        """Limpia estimación de período, fallback y buckets estacionales."""
        self._epochs.clear()
        self._n_updates = 0
        self._bucket_minutes = None
        self._fallback_only = False
        self._seasonal = None
        self._fallback.reset()

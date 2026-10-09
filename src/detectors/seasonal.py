"""Detector de anomalías robusto que tiene en cuenta el ciclo diario y la tendencia causal."""

from collections import deque
from datetime import datetime
import math
import statistics

from detectors.zscore_robusto import RollingRobustZ


class DailySeasonalRobustZ:
    """Aplica el detector z-score robusto particionando el día en ventanas diurnas
    y sustrayendo la tendencia causal de fondo T_t (Eje A: Desacople T_t + S_t + R_t).

    Cada momento del día pertenece a un 'bucket' o canasta (por defecto de 15 minutos).
    Se mantiene una instancia independiente de RollingRobustZ para cada canasta.
    La mediana móvil de 24 horas (T_t) neutraliza derivas y rampas lentas de nivel
    antes de alimentar los buckets diurnos, aislando el verdadero residuo estacional.
    """

    def __init__(
        self,
        bucket_size_minutes: int = 15,
        window_size: int = 20,
        min_observations: int = 5,
        trend_window_size: int = 0,
    ) -> None:
        if bucket_size_minutes < 1 or bucket_size_minutes > 1440:
            raise ValueError("bucket_size_minutes debe estar entre 1 y 1440.")

        self._bucket_size = bucket_size_minutes
        self._window_size = window_size
        self._min_obs = min_observations
        self._trend_window = trend_window_size
        self._trend_buffer: deque[float] = deque(maxlen=trend_window_size) if trend_window_size > 0 else deque()
        self._buckets: dict[int, RollingRobustZ] = {}

    def _parse_and_bucket(self, timestamp: str) -> int:
        """Convierte el timestamp en un índice de canasta diaria."""
        dt_str = timestamp.replace(" ", "T")
        try:
            dt = datetime.fromisoformat(dt_str)
        except ValueError:
            hora = int(timestamp[11:13])
            minuto = int(timestamp[14:16])
            return (hora * 60 + minuto) // self._bucket_size

        return (dt.hour * 60 + dt.minute) // self._bucket_size

    def update(self, value: float, timestamp: str) -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0

        if self._trend_window > 0 and len(self._trend_buffer) >= 10:
            trend = statistics.median(self._trend_buffer)
            detrended_val = value - trend
        else:
            detrended_val = value

        if self._trend_window > 0:
            self._trend_buffer.append(value)

        bucket_idx = self._parse_and_bucket(timestamp)

        if bucket_idx not in self._buckets:
            self._buckets[bucket_idx] = RollingRobustZ(
                window_size=self._window_size,
                min_observations=self._min_obs,
            )

        return self._buckets[bucket_idx].update(detrended_val, timestamp)

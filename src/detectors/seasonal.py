"""Detector de anomalías robusto que tiene en cuenta el ciclo diario."""

from datetime import datetime

from detectors.zscore_robusto import RollingRobustZ


class DailySeasonalRobustZ:
    """Aplica el detector z-score robusto pero particionando el día en ventanas.

    Cada momento del día pertenece a un 'bucket' o canasta (por defecto de 15 minutos).
    Se mantiene una instancia independiente de RollingRobustZ para cada canasta.
    Esto permite que el detector aprenda qué es normal a las 9:00 AM vs las 3:00 AM.
    """

    def __init__(
        self,
        bucket_size_minutes: int = 15,
        window_size: int = 20,
        min_observations: int = 5,
    ) -> None:
        if bucket_size_minutes < 1 or bucket_size_minutes > 1440:
            raise ValueError("bucket_size_minutes debe estar entre 1 y 1440.")
            
        self._bucket_size = bucket_size_minutes
        self._window_size = window_size
        self._min_obs = min_observations
        self._buckets: dict[int, RollingRobustZ] = {}

    def _parse_and_bucket(self, timestamp: str) -> int:
        """Convierte el timestamp en un índice de canasta diaria."""
        # Se asume formato tipo "2014-04-10 07:15:00" que usa NAB.
        # Remplazamos espacio por T para fromisoformat
        dt_str = timestamp.replace(" ", "T")
        try:
            dt = datetime.fromisoformat(dt_str)
        except ValueError:
            # Fallback simple
            # Asumimos que los caracteres 11 a 16 son HH:MM
            hora = int(timestamp[11:13])
            minuto = int(timestamp[14:16])
            return (hora * 60 + minuto) // self._bucket_size
            
        return (dt.hour * 60 + dt.minute) // self._bucket_size

    def update(self, value: float, timestamp: str) -> float:
        bucket_idx = self._parse_and_bucket(timestamp)
        
        if bucket_idx not in self._buckets:
            self._buckets[bucket_idx] = RollingRobustZ(
                window_size=self._window_size,
                min_observations=self._min_obs
            )
            
        return self._buckets[bucket_idx].update(value, timestamp)

"""Z-score robusto de doble escala (base oficial del pipeline).

Ver `aprendizaje.md` (doble escala): NAB 45.19 vs 40.02 de la variante con
buckets fijos (+5.17). `SeasonalDualScaleRobustZ` es la base oficial del
pipeline; `DualScaleRobustZ` es su fallback sin timestamps y el estimador
de escala lenta global.
"""

import math
import statistics
from collections import deque
from datetime import datetime

from sixeyes.detectors.base import Detector
from sixeyes.detectors.zscore_robusto import Z_MEDIO

import math
import statistics
from collections import deque
from datetime import datetime

from sixeyes.detectors.base import Detector
from sixeyes.detectors.zscore_robusto import Z_MEDIO

FAST_WINDOW = 20
SLOW_WINDOW = 6400
MIN_OBSERVATIONS = 5
MIN_SLOW = 20


def _bucket_index(timestamp: str, bucket_minutes: int) -> int | None:
    """Índice de canasta diaria. None si el timestamp no es parseable."""
    if not timestamp:
        return None
    dt_str = timestamp.replace(" ", "T")
    try:
        dt = datetime.fromisoformat(dt_str)
    except ValueError:
        try:
            from dateutil import parser as date_parser
            dt = date_parser.parse(timestamp)
        except Exception:
            return None
    return (dt.hour * 60 + dt.minute) // bucket_minutes


class DualScaleRobustZ(Detector):
    """Z-score con centro local y escala dual conservadora."""

    def __init__(
        self,
        fast_window: int = FAST_WINDOW,
        slow_window: int = SLOW_WINDOW,
        min_observations: int = MIN_OBSERVATIONS,
    ) -> None:
        if fast_window < 1 or slow_window < 1:
            raise ValueError("Las ventanas deben ser >= 1.")
        self._fast_window = fast_window
        self._slow_window = slow_window
        self._min_obs = min_observations
        self._fast: deque[float] = deque(maxlen=fast_window)
        self._slow: deque[float] = deque(maxlen=slow_window)
        self._sum = 0.0
        self._sumsq = 0.0
        # Fallback de resolución mínima observada (paridad con RollingRobustZ).
        self._prev_val: float | None = None
        self._resolucion: float | None = None

    def _slow_std(self) -> float | None:
        n = len(self._slow)
        if n < MIN_SLOW:
            return None
        mean = self._sum / n
        var = max(self._sumsq / n - mean * mean, 0.0)
        return math.sqrt(var)

    def _ingest(self, value: float) -> None:
        if len(self._slow) == self._slow.maxlen:
            old = self._slow[0]
            self._sum -= old
            self._sumsq -= old * old
        if self._prev_val is not None:
            diff = abs(value - self._prev_val)
            if diff > 0.0 and (self._resolucion is None or diff < self._resolucion):
                self._resolucion = diff
        self._prev_val = value
        self._fast.append(value)
        self._slow.append(value)
        self._sum += value
        self._sumsq += value * value

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        if len(self._fast) < self._min_obs:
            self._ingest(value)
            return 0.0

        mediana = statistics.median(self._fast)
        mad = statistics.median([abs(x - mediana) for x in self._fast])
        escala_rapida = 1.4826 * mad

        escala_lenta = self._slow_std()
        if escala_lenta is None or escala_lenta <= 0.0:
            escala = escala_rapida
        elif escala_rapida <= 0.0:
            escala = escala_lenta
        else:
            escala = max(escala_rapida, escala_lenta)

        if escala <= 0.0:
            # Paridad con RollingRobustZ: resolución mínima observada.
            escala = self._resolucion if self._resolucion else 0.0

        if escala <= 0.0:
            score = 0.0
        else:
            z = abs(value - mediana) / escala
            score = z / (z + Z_MEDIO)

        self._ingest(value)
        return min(max(score, 0.0), 1.0)

    def reset(self) -> None:
        """Limpia ventanas rápidas/lentas y acumuladores Welford."""
        self._fast.clear()
        self._slow.clear()
        self._sum = 0.0
        self._sumsq = 0.0
        self._prev_val = None
        self._resolucion = None


class SeasonalDualScaleRobustZ(Detector):
    """Composición estacional + dual: centro estacional por bucket + escala dual conservadora.

    Cada bucket diario guarda su mediana/MAD rápidas (W=20, nivel local
    estacional); UNA escala lenta global (std rodante W=6400, O(1) por sumas)
    estabiliza el denominador frente a ráfagas:
    ``z = |x - mediana_bucket| / max(1.4826*MAD_bucket, std_global)``.
    Sin timestamps parseables: fallback global sin estacionalidad.
    """

    def __init__(
        self,
        bucket_size_minutes: int = 15,
        fast_window: int = FAST_WINDOW,
        slow_window: int = SLOW_WINDOW,
        min_observations: int = MIN_OBSERVATIONS,
    ) -> None:
        if bucket_size_minutes < 1 or bucket_size_minutes > 1440:
            raise ValueError("bucket_size_minutes debe estar entre 1 y 1440.")
        self._bucket_size = bucket_size_minutes
        self._fast_window = fast_window
        self._slow_window = slow_window
        self._min_obs = min_observations
        self._buckets: dict[int, deque[float]] = {}
        self._fallback = DualScaleRobustZ(
            fast_window=fast_window, slow_window=slow_window,
            min_observations=min_observations,
        )
        self._slow: deque[float] = deque(maxlen=slow_window)
        self._sum = 0.0
        self._sumsq = 0.0

    def _slow_std(self) -> float | None:
        n = len(self._slow)
        if n < MIN_SLOW:
            return None
        mean = self._sum / n
        return math.sqrt(max(self._sumsq / n - mean * mean, 0.0))

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        idx = _bucket_index(timestamp, self._bucket_size)
        if idx is None:
            return self._fallback.update(value, timestamp)
        buf = self._buckets.get(idx)
        if buf is None:
            buf = self._buckets[idx] = deque(maxlen=self._fast_window)
        if len(buf) < self._min_obs:
            buf.append(value)
            self._ingest_slow(value)
            return 0.0
        mediana = statistics.median(buf)
        mad = statistics.median([abs(x - mediana) for x in buf])
        escala_rapida = 1.4826 * mad
        escala_lenta = self._slow_std()
        if escala_lenta is None or escala_lenta <= 0.0:
            escala = escala_rapida
        elif escala_rapida <= 0.0:
            escala = escala_lenta
        else:
            escala = max(escala_rapida, escala_lenta)
        score = 0.0 if escala <= 0.0 else (lambda z: z / (z + Z_MEDIO))(abs(value - mediana) / escala)
        buf.append(value)
        self._ingest_slow(value)
        return min(max(score, 0.0), 1.0)

    def _ingest_slow(self, value: float) -> None:
        if len(self._slow) == self._slow.maxlen:
            old = self._slow[0]
            self._sum -= old
            self._sumsq -= old * old
        self._slow.append(value)
        self._sum += value
        self._sumsq += value * value

    def reset(self) -> None:
        """Limpia buckets, escala lenta global y fallback."""
        self._buckets.clear()
        self._slow.clear()
        self._sum = 0.0
        self._sumsq = 0.0
        self._fallback.reset()

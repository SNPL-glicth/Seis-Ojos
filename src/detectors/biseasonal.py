"""Detector de anomalías bi-estacional robusto (Laboral vs Fin de semana + Horario)."""

from datetime import datetime
import math
from typing import Any

from detectors.base import Detector
from detectors.zscore_robusto import RollingRobustZ


class BiSeasonalRobustZ(Detector):
    """Modela dos regímenes semanales (Laboral vs Fin de semana) y franjas horarias diurnas.

    - Régimen 0: Lunes a Viernes (Días hábiles / Laborales).
    - Régimen 1: Sábado y Domingo (Fin de semana).
    - Franjas horarias: 96 canastas de 15 minutos por día.
    Total de celdas bi-estacionales: 2 regímenes x 96 canastas = 192 casillas.

    Jerarquía causal de referencia:
    1. Si la casilla (régimen, hora) tiene >= min_observations_regime (defecto 3),
       se evalúa contra el modelo específico del régimen.
    2. En caso contrario, si la casilla diurna global (hora) tiene >= min_observations_diurnal
       (defecto 5), se evalúa contra el modelo diurno (fallback suave).
    3. Si ambas están en fase inicial, se devuelve 0.0 sin emitir falsas alarmas.
    """

    def __init__(
        self,
        bucket_size_minutes: int = 15,
        window_size: int = 20,
        min_observations_regime: int = 3,
        min_observations_diurnal: int = 5,
    ) -> None:
        if bucket_size_minutes < 1 or bucket_size_minutes > 1440:
            raise ValueError("bucket_size_minutes debe estar entre 1 y 1440.")

        self._bucket_size = bucket_size_minutes
        self._window_size = window_size
        self._min_regime = min_observations_regime
        self._min_diurnal = min_observations_diurnal

        # 96 casillas diurnas generales: bucket_idx -> RollingRobustZ
        self._diurnal_buckets: dict[int, RollingRobustZ] = {}
        # 192 casillas bi-estacionales: (regime, bucket_idx) -> RollingRobustZ
        self._regime_buckets: dict[tuple[int, int], RollingRobustZ] = {}

    def _parsear_tiempo(self, timestamp: str) -> tuple[int, int]:
        """Extrae el régimen semanal (0: laboral, 1: fin de semana) y el índice horario."""
        if not timestamp:
            return 0, 0

        dt_str = timestamp.replace(" ", "T")
        try:
            dt = datetime.fromisoformat(dt_str)
        except ValueError:
            try:
                from dateutil import parser as date_parser
                dt = date_parser.parse(timestamp)
            except Exception:
                # Fallback sintético si no se puede parsear
                try:
                    hora = int(timestamp[11:13])
                    minuto = int(timestamp[14:16])
                    return 0, (hora * 60 + minuto) // self._bucket_size
                except Exception:
                    return 0, 0

        # weekday(): Lunes es 0, Domingo es 6
        regimen = 1 if dt.weekday() >= 5 else 0
        bucket_idx = (dt.hour * 60 + dt.minute) // self._bucket_size
        return regimen, bucket_idx

    def update(self, value: float, timestamp: str) -> float:
        """Procesa una observación de forma causal y devuelve un score en [0, 1]."""
        if math.isnan(value) or math.isinf(value):
            return 0.0

        regimen, bucket_idx = self._parsear_tiempo(timestamp)
        regime_key = (regimen, bucket_idx)

        # Inicialización perezosa de los modelos
        if bucket_idx not in self._diurnal_buckets:
            self._diurnal_buckets[bucket_idx] = RollingRobustZ(
                window_size=self._window_size,
                min_observations=self._min_diurnal,
            )

        if regime_key not in self._regime_buckets:
            self._regime_buckets[regime_key] = RollingRobustZ(
                window_size=self._window_size,
                min_observations=self._min_regime,
            )

        diurnal_det = self._diurnal_buckets[bucket_idx]
        regime_det = self._regime_buckets[regime_key]

        regime_listo = len(regime_det._window) >= self._min_regime
        diurnal_listo = len(diurnal_det._window) >= self._min_diurnal

        if regime_listo:
            # 1. Régimen maduro: evaluación directa sobre el comportamiento de su grupo
            score = regime_det.update(value, timestamp)
            # Mantener diurno actualizado en paralelo
            diurnal_det.update(value, timestamp)
        elif diurnal_listo:
            # 2. Fallback suave: el régimen específico aún acumula datos, usamos diurno
            score = diurnal_det.update(value, timestamp)
            # El régimen específico acumula la observación para alcanzar madurez
            regime_det.update(value, timestamp)
        else:
            # 3. Calentamiento global: ambos acumulan, score 0.0 sin falsas alarmas
            diurnal_det.update(value, timestamp)
            regime_det.update(value, timestamp)
            score = 0.0

        return score

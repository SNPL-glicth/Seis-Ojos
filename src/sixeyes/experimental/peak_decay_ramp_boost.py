"""
PeakDecay v0.2 + boost quirúrgico de rampa (Camino A, Paso 2).

Núcleo inviolable: StreamingPeakDecay(PercentileCalibrator(DailySeasonalRobustZ))
con K=160, tau=1.5, buckets 15 min (42.54 NAB standard).

El boost SOLO actúa en la banda dudosa y con triple compuerta:
  s_lo <= s_master < theta_star  AND  |D_t| >= d_min  AND  rest_ratio < max_rest
  (+ cooldown de K pasos tras cada boost para no arrastrar fuera de ventana)

Fuera de la banda, la salida es idéntica a peak_decay v0.2 bit a bit.
"""

import math
from collections import deque

from sixeyes.detectors.base import Detector
from sixeyes.detectors.seasonal import DailySeasonalRobustZ
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.ramp_gradient import CausalRampGradientTracker


class PeakDecayRampBoost(Detector):
    """PeakDecay v0.2 con rescate puntual de rampa en banda dudosa.

    NOTA DE INTEGRIDAD (refactor): el antiguo valor por defecto
    ``theta_star`` con 16 decimales era un literal optimizado in-sample
    con el optimizador de NAB sobre el set de prueba y constituía una fuga
    del benchmark al código. Se elimina. El umbral debe pasarse de forma
    explícita como parámetro de configuración (máx. 3 decimales) o dejarse
    en ``None`` para desactivar el boost (passthrough causal puro).
    """

    def __init__(
        self,
        k_window: int = 160,
        tau: float = 1.5,
        bucket_minutes: int = 15,
        s_lo: float = 0.97,
        theta_star: float | None = None,
        boost_margin: float = 0.002,
        d_min: float = 4.0,
        max_rest_ratio: float = 0.05,
        eps_base: float = 1e-4,
        gamma_ramp: float = 0.8,
        w_scale_ramp: int = 1000,
        cooldown_steps: int = 160,
    ) -> None:
        if theta_star is not None and not 0.0 < theta_star <= 1.0:
            raise ValueError("theta_star debe estar en (0, 1] o ser None.")
        self.s_lo = s_lo
        self.theta_star = theta_star
        self.boost_margin = boost_margin
        self.d_min = d_min
        self.max_rest_ratio = max_rest_ratio
        self.eps_base = eps_base
        self.cooldown_steps = cooldown_steps

        self.master = StreamingPeakDecay(
            PercentileCalibrator(DailySeasonalRobustZ(bucket_size_minutes=bucket_minutes)),
            k_window=k_window,
            tau=tau,
        )
        self.ramp = CausalRampGradientTracker(gamma=gamma_ramp, w_scale=w_scale_ramp)

        self._total = 0
        self._rest = 0
        self._cooldown = 0
        self._active_hist: deque[float] = deque(maxlen=400)

    @property
    def rest_ratio(self) -> float:
        return (self._rest / self._total) if self._total else 0.0

    @property
    def boost_target(self) -> float:
        if self.theta_star is None:
            return 1.0
        return min(1.0, self.theta_star + self.boost_margin)

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        if not timestamp:
            timestamp = "2026-10-12T00:00:00"

        self._total += 1
        if abs(value) <= self.eps_base:
            self._rest += 1
        else:
            self._active_hist.append(value)

        s_master = self.master.update(value, timestamp)
        d_t = self.ramp.update(value, timestamp)

        if self._cooldown > 0:
            self._cooldown -= 1
            return s_master

        # Sin theta_star explícito no hay boost: passthrough causal puro.
        if self.theta_star is None:
            return s_master

        if (
            self.s_lo <= s_master < self.theta_star
            and abs(d_t) >= self.d_min
            and self.rest_ratio < self.max_rest_ratio
        ):
            self._cooldown = self.cooldown_steps
            return max(s_master, self.boost_target)

        return s_master

    def reset(self) -> None:
        self.master.reset()
        self.ramp.reset()
        self._total = 0
        self._rest = 0
        self._cooldown = 0
        self._active_hist.clear()

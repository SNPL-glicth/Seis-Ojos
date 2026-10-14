"""
Rastreador causal de persistencia de rampa direccional (Leaky Integrator).

Mecanismo físico:
En congestiones de tráfico o saturación progresiva de recursos, las anomalías
se manifiestan como derivas monótonas continuas (desaceleraciones progresivas
o subidas persistentes de tiempo de viaje/ocupación) en lugar de picos de 1 paso.

Para detectar estas firmas de forma agnóstica a las unidades:
1. Normalización por escala global:
   delta_t = (x_t - x_{t-1}) / (MAD_global + eps)
2. Integrador causal con fuga (Leaky Integrator / Momentum):
   D_t = gamma * D_{t-1} + delta_t  (gamma = 0.8)
   En ruido simétrico, E[D_t] ~ 0.
   En una rampa monótona sostenida, |D_t| acumula valores extremos (|D_t| >= 4.0).
"""

from collections import deque
import math
import numpy as np

from sixeyes.detectors.base import Detector


class CausalRampGradientTracker(Detector):
    """Rastreador causal de persistencia de gradiente con integrador con fuga."""

    def __init__(
        self,
        gamma: float = 0.8,
        w_scale: int = 1000,
        min_warmup: int = 20,
        eps: float = 1e-6,
    ) -> None:
        """Inicializa el rastreador de rampa.

        Args:
            gamma: Factor de retención de memoria (0.0 < gamma < 1.0).
            w_scale: Tamaño máximo del buffer para estimar MAD global causal.
            min_warmup: Mínimo de observaciones requeridas para computar escala.
            eps: Regularización para evitar división por cero.
        """
        self.gamma = gamma
        self.w_scale = w_scale
        self.min_warmup = min_warmup
        self.eps = eps

        self._prev_val: float | None = None
        self._d_t: float = 0.0
        self._val_buffer: deque[float] = deque(maxlen=w_scale)

    @property
    def d_t(self) -> float:
        """Valor actual del acumulador de persistencia direccional D_t."""
        return self._d_t

    @property
    def abs_d_t(self) -> float:
        """Magnitud absoluta de persistencia direccional |D_t|."""
        return abs(self._d_t)

    def update(self, value: float, timestamp: str = "") -> float:
        """Actualiza el integrador causal con la nueva observación.

        Args:
            value: Observación escalar x_t.
            timestamp: Marca temporal (opcional).

        Returns:
            float: Valor actual del acumulador D_t.
        """
        if math.isnan(value) or math.isinf(value):
            return 0.0

        self._val_buffer.append(value)

        if self._prev_val is not None and len(self._val_buffer) >= self.min_warmup:
            diff = value - self._prev_val
            arr = np.array(self._val_buffer)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) * 1.4826
            escala = max(mad, self.eps)
            delta = diff / escala
            self._d_t = self.gamma * self._d_t + delta
        else:
            self._d_t = 0.0

        self._prev_val = value
        return self._d_t

    def reset(self) -> None:
        """Reinicia el estado del rastreador."""
        self._prev_val = None
        self._d_t = 0.0
        self._val_buffer.clear()
"""Contextual Anomaly Likelihood sobre residuos / errores de predicción.

Inspirado en la formulación rigurosa de Numenta Anomaly Likelihood (Ahmad et al., 2017)
y ContextOSE:
- Mantiene una estimación adaptativa de largo plazo de la media y varianza de los
  errores mediante medias móviles exponenciales (EMA) en O(1) tiempo y memoria.
- Calcula la media de corto plazo para suavizar fluctuaciones de un solo paso.
- Evalúa el z-score estandarizado del error reciente relativo al contexto local.
- Convierte el z-score a probabilidad de cola / probabilidad acumulada mediante la
  función de error gaussiana:
      score_t = 1 - Q((mu_short - mu_long) / sigma_long) = 0.5 * (1 + erf(z / sqrt(2)))
- Acota numéricamente los resultados en [0.0, 1.0].

Propiedades fundamentales:
1. Causalidad estricta: solo utiliza información hasta el paso t.
2. Memoria acotada O(1) y tiempo O(1) por paso.
3. Invarianza completa ante reset().
4. En régimen de baja volatilidad histórica, pequeñas desviaciones sistemáticas se
   magnifican a scores cercanos a 1.0 (resuelve transiciones de régimen sutiles).
5. En régimen turbulento o de alta varianza, la estimación de varianza se expande,
   evitando detonar falsos positivos ante fluctuaciones normales.
"""

from __future__ import annotations

import math
from typing import Any

from sixeyes.detectors.base import Detector


class ContextualLikelihood(Detector):
    """Calcula la probabilidad de cola contextual de los errores de predicción."""

    def __init__(
        self,
        base_detector: Detector | None = None,
        short_window: int = 10,
        long_window: int = 1000,
        eps: float = 1e-6,
    ) -> None:
        """Args:
            base_detector: Detector base opcional que genera residuos/errores.
                Si es None, update() recibe directamente el error/score numérico.
            short_window: Tamaño de ventana efectiva de corto plazo (W' en Numenta).
            long_window: Tamaño de ventana efectiva de largo plazo (W en Numenta).
            eps: Regularizador numérico de varianza mínima para evitar división por cero.
        """
        if short_window < 1:
            raise ValueError("short_window debe ser >= 1.")
        if long_window <= short_window:
            raise ValueError("long_window debe ser > short_window.")
        if eps <= 0:
            raise ValueError("eps debe ser > 0.")

        self.base_detector = base_detector
        self.short_window = short_window
        self.long_window = long_window
        self.eps = eps

        # Factores de decaimiento EMA equivalentes a una ventana de N pasos:
        # alpha = 2 / (N + 1)
        self._alpha_short = 2.0 / (short_window + 1.0)
        self._alpha_long = 2.0 / (long_window + 1.0)

        # Estado interno
        self._step = 0
        self._mu_short = 0.0
        self._mu_long = 0.0
        self._var_long = 0.0

    def reset(self) -> None:
        """Reinicia el detector y su base si existe."""
        if self.base_detector is not None:
            self.base_detector.reset()
        self._step = 0
        self._mu_short = 0.0
        self._mu_long = 0.0
        self._var_long = 0.0

    def update(self, value: float, timestamp: str = "") -> float:
        """Procesa un valor escalar.
        
        Si tiene base_detector, value se pasa al detector base y se procesa su salida.
        Si no tiene base_detector, value se interpreta como el error/score a modelar.
        """
        if self.base_detector is not None:
            err = self.base_detector.update(value, timestamp)
        else:
            err = value

        if math.isnan(err) or math.isinf(err):
            return 0.0

        # Para errores cuadráticos o magnitudes, err >= 0
        err_val = float(err)
        self._step += 1

        if self._step == 1:
            self._mu_short = err_val
            self._mu_long = err_val
            self._var_long = self.eps
            return 0.5

        # Actualización EMA causal de medias
        self._mu_short += self._alpha_short * (err_val - self._mu_short)
        
        diff_long = err_val - self._mu_long
        self._mu_long += self._alpha_long * diff_long
        
        # Welford / EMA de varianza causal:
        # var_long = (1 - alpha_long) * var_long + alpha_long * (err_val - mu_long_prev)*(err_val - mu_long_new)
        self._var_long = (1.0 - self._alpha_long) * self._var_long + self._alpha_long * diff_long * (err_val - self._mu_long)
        
        # Salvaguarda de varianza positiva
        current_var = max(self._var_long, self.eps)
        sigma = math.sqrt(current_var)

        # Z-score de la media de corto plazo vs la distribución de largo plazo
        # Si la media reciente supera la histórica, z > 0
        z = (self._mu_short - self._mu_long) / sigma

        # Función de distribución acumulada normal estándar:
        # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        # Cuando z es muy positivo, score -> 1.0
        score = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        return max(0.0, min(1.0, float(score)))

"""Detector de anomalías basado en z-score robusto sobre ventana deslizante."""

from collections import deque
import math
import statistics

DEFAULT_WINDOW_SIZE: int = 50
MIN_WARMUP_OBSERVATIONS: int = 5
NORMAL_SCALE_MAD: float = 1.4826
NORMAL_SCALE_MEAN_AD: float = 1.253314  # Convención habitual en literatura para MeanAD (no verificada sin internet)
SCORE_SIN_EVIDENCIA: float = 0.0

# Convención habitual del z-score modificado (no verificada aquí).
# El score vale 0.5 cuando z = Z_MEDIO.
# La función score = z / (z + Z_MEDIO) es monótona y no satura de forma práctica.
Z_MEDIO: float = 3.5

class RollingRobustZ:
    """Detector de anomalías en streaming usando mediana y MAD sobre ventana móvil.

    Calcula el z-score robusto: |valor - mediana| / (1.4826 * MAD) sobre las
    observaciones previas y lo acota en [0, 1] mediante la función z / (z + Z_MEDIO).
    
    Comportamiento en casos borde:
    - Calentamiento: Devuelve 0.0 mientras la ventana tenga menos de
      MIN_WARMUP_OBSERVATIONS observaciones.
    - MAD igual a cero: Respaldo progresivo usando Desviación Absoluta Media (MeanAD) 
      y finalmente la resolución mínima observada empíricamente en la serie temporal.
    - Valores NaN o infinitos: Devuelve 0.0 y no se añaden a la ventana histórica.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        min_observations: int = MIN_WARMUP_OBSERVATIONS,
    ) -> None:
        if window_size < 1:
            raise ValueError("El tamaño de ventana debe ser mayor o igual a 1.")

        self._window_size = window_size
        self._min_observations = min_observations
        self._window: deque[float] = deque(maxlen=window_size)
        
        self._prev_val: float | None = None
        self._resolucion_observada: float | None = None

    def update(self, value: float, timestamp: str) -> float:
        """Devuelve un score en [0, 1], mayor valor significa más anómalo;
        el detector solo puede usar el pasado y el punto actual."""
        # Si el valor no es finito, se descarta
        if math.isnan(value) or math.isinf(value):
            return 0.0

        # Durante el calentamiento se retorna 0.0
        if len(self._window) < self._min_observations:
            self._window.append(value)
            self._actualizar_resolucion(value)
            return 0.0

        mediana = statistics.median(self._window)
        desviaciones = [abs(x - mediana) for x in self._window]
        mad = statistics.median(desviaciones)

        if mad == 0.0:
            escala = self._calcular_escala_alternativa(desviaciones)
            if escala > 0.0:
                z = abs(value - mediana) / escala
                score = z / (z + Z_MEDIO)
            else:
                # 3. Sin historial de variación, se devuelve constante documentada
                score = SCORE_SIN_EVIDENCIA
        else:
            escala = NORMAL_SCALE_MAD * mad
            z = abs(value - mediana) / escala
            score = z / (z + Z_MEDIO)

        # La observación actual entra a la ventana después de puntuada
        self._window.append(value)
        self._actualizar_resolucion(value)

        return min(max(score, 0.0), 1.0)
        
    def _calcular_escala_alternativa(self, desviaciones: list[float]) -> float:
        """En caso de MAD == 0, aplica respaldos para determinar la escala de variabilidad."""
        # 1. Desviación absoluta MEDIA respecto a la mediana
        mean_ad = sum(desviaciones) / len(desviaciones)
        escala = NORMAL_SCALE_MEAN_AD * mean_ad
        
        # 2. Si la anterior es 0: usa la RESOLUCIÓN OBSERVADA de la serie
        if escala == 0.0 and self._resolucion_observada is not None and self._resolucion_observada > 0.0:
            escala = self._resolucion_observada
            
        return escala
        
    def _actualizar_resolucion(self, value: float) -> None:
        """Mantiene un registro de la menor diferencia absoluta no nula observada."""
        if self._prev_val is not None:
            diff = abs(value - self._prev_val)
            if diff > 0.0:
                if self._resolucion_observada is None:
                    self._resolucion_observada = diff
                else:
                    self._resolucion_observada = min(self._resolucion_observada, diff)
        self._prev_val = value

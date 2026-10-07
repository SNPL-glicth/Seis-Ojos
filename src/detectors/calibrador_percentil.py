import math
import bisect
from typing import Any
from .base import Detector

class PercentileCalibrator(Detector):
    """
    Un calibrador que envuelve a otro detector y transforma sus puntajes crudos
    en percentiles históricos.
    
    Esto evita el problema de la falta de comparabilidad entre series (donde 
    los puntajes Z pueden tener escalas completamente distintas) y la saturación 
    de scores, asegurando una distribución uniforme en [0, 1].
    """
    
    # Cantidad mínima de valores no nulos en el historial para empezar a emitir
    # puntajes mayores a 0.0. 288 equivale a un día de datos a 5 minutos.
    # En producción para millones de series, se requeriría un estimador de 
    # cuantiles acotado en lugar de un historial completo.
    MIN_HISTORIAL = 288

    def __init__(self, detector_interno: Detector):
        """
        Inicializa el calibrador de percentiles.
        
        Args:
            detector_interno: Una instancia de un detector que cumpla la interfaz Detector.
        """
        self._detector = detector_interno
        self._historial = []

    def update(self, value: float, timestamp: Any) -> float:
        """
        Actualiza el detector interno y convierte el puntaje crudo en un percentil.
        
        Args:
            value: El valor de la métrica en el instante actual.
            timestamp: El timestamp asociado al valor.
            
        Returns:
            El score calibrado como un percentil en (0, 1) si el historial está
            lleno, o 0.0 si el score crudo es 0, NaN/Inf, o el historial es insuficiente.
        """
        raw = self._detector.update(value, timestamp)
        
        # 2. Si raw es NaN/inf, o raw == 0.0: devuelve 0.0 y NO lo agrega al historial.
        # (0.0 significa "sin evidencia" o "totalmente normal"; excluirlo evita que
        # el calentamiento del detector interno infle artificialmente los percentiles).
        if math.isnan(raw) or math.isinf(raw) or raw == 0.0:
            return 0.0
            
        # 3. Si el historial tiene menos de MIN_HISTORIAL valores: devuelve 0.0,
        # y agrega raw al historial.
        if len(self._historial) < self.MIN_HISTORIAL:
            bisect.insort(self._historial, raw)
            return 0.0
            
        # 4. Cálculo del percentil basado en el rango.
        # menores: cantidad de elementos estrictamente menores que raw
        # iguales: cantidad de elementos exactamente iguales a raw
        menores = bisect.bisect_left(self._historial, raw)
        iguales = bisect.bisect_right(self._historial, raw) - menores
        n = len(self._historial)
        
        # Fórmula: (menores + 0.5 * iguales + 0.5) / (n + 1)
        # Esta fórmula empírica asegura que el score resultante nunca valga 
        # exactamente 0 ni 1 (estrictamente contenido en (0, 1)). Esto previene 
        # reintroducir empates masivos en el tope (el problema de la saturación) 
        # que confundirían al optimizador de umbrales.
        score = (menores + 0.5 * iguales + 0.5) / (n + 1)
        
        # Insertar el nuevo valor crudo en el historial ordenado
        bisect.insort(self._historial, raw)
        
        return score

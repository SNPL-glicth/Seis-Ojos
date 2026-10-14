"""
Causal Spectral Anomaly Detector - STFT-based Spectral Salience Residual.

Detector independiente que detecta anomalías de fase/frecuencia mediante residuo
espectral en streaming, con su propio PercentileCalibrator.
"""

from collections import deque
import math
from typing import Optional
import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator


class CausalSTFTDetector(Detector):
    """
    Detector de anomalías espectrales basado en STFT causal y residuo log-espectral.
    
    Pipeline causal estricto:
    x_t -> Rolling STFT (W=64, hop=1) -> Log-Spectral Residual -> PercentileCalibrator -> [0,1]
    
    Compuerta de ruido: solo emite si saliencia > 3.5 sigma sobre piso de ruido.
    """

    def __init__(
        self,
        window_size: int = 64,
        hop: int = 1,
        nperseg: int = 32,
        noise_gate_sigma: float = 3.5,
        noise_floor_history: int = 500,
        calibrator_min_history: int = 288,
    ) -> None:
        """Inicializa el detector espectral causal.
        
        Args:
            window_size: Tamaño de ventana STFT (debe ser par).
            hop: Paso de avance (1 para streaming estricto).
            nperseg: Longitud de segmento FFT.
            noise_gate_sigma: Umbral en sigmas sobre piso de ruido para compuerta.
            noise_floor_history: Historial para estimar piso de ruido espectral.
            calibrator_min_history: Mínimo historial para calibrador percentil.
        """
        if window_size % 2 != 0:
            raise ValueError("window_size debe ser par para STFT real.")
        if hop <= 0:
            raise ValueError("hop debe ser positivo.")
        if nperseg > window_size:
            raise ValueError("nperseg no puede exceder window_size.")
        
        self.window_size = window_size
        self.hop = hop
        self.nperseg = nperseg
        self.noise_gate_sigma = noise_gate_sigma
        self.noise_floor_history = noise_floor_history
        
        # Buffer circular para STFT
        self._buffer = deque(maxlen=window_size)
        self._step = 0
        self._stft_ready = False
        
        # Historial de espectros de fondo para piso de ruido
        self._spectral_history = deque(maxlen=noise_floor_history)
        
        # Historial de residuos para compuerta
        self._residual_history = deque(maxlen=noise_floor_history)
        
        # Calibrador propio
        self._raw_history = []
        self._min_history = calibrator_min_history

    def _compute_spectral_residual(self, signal: np.ndarray) -> float:
        """Calcula residuo log-espectral vs fondo espectral adaptativo."""
        # STFT con scipy
        from scipy.signal import stft
        
        # Normalizar señal para estabilidad espectral
        sig_std = np.std(signal)
        if sig_std > 1e-8:
            signal = (signal - np.mean(signal)) / sig_std
        
        # STFT
        f, t, Zxx = stft(signal, nperseg=self.nperseg, noverlap=self.nperseg - self.hop,
                         return_onesided=True, padded=False, boundary=None)
        
        # Espectro de potencia promedio temporal
        power_spectrum = np.mean(np.abs(Zxx)**2, axis=1)
        
        # Log-espectro
        log_spectrum = np.log(np.maximum(power_spectrum, 1e-10))
        
        # Actualizar historial de fondo
        self._spectral_history.append(log_spectrum)
        
        # Calcular residuo vs fondo espectral
        if len(self._spectral_history) >= 10:
            background = np.mean(np.array(list(self._spectral_history))[:-1], axis=0)
            # Residuo KL aproximado: sum( p * log(p/q) ) donde p=actual, q=fondo
            p = np.exp(log_spectrum - np.max(log_spectrum))
            q = np.exp(background - np.max(background))
            p = p / (np.sum(p) + 1e-10)
            q = q / (np.sum(q) + 1e-10)
            
            # Divergencia KL simétrica aproximada
            kl = np.sum(p * np.log(np.maximum(p / np.maximum(q, 1e-10), 1e-10)))
            
            return float(max(0.0, kl))
        
        return 0.0

    def update(self, value: float, timestamp: str = "") -> float:
        """Procesa una observación y retorna score espectral calibrado [0,1]."""
        if math.isnan(value) or math.isinf(value):
            return 0.0
        
        # Actualizar buffer
        self._buffer.append(value)
        self._step += 1
        
        if len(self._buffer) < self.window_size:
            return 0.0
        
        self._stft_ready = True
        
        # Mantener buffer causal (rolling window)
        if len(self._buffer) > self.window_size:
            self._buffer.popleft()
        
        # Calcular residuo espectral
        signal = np.array(self._buffer, dtype=np.float64)
        residual = self._compute_spectral_residual(signal)
        
        # Actualizar historial de residuos
        self._residual_history.append(residual)
        
        # Compuerta de ruido: 3.5 sigma sobre piso
        if len(self._residual_history) >= 50:
            residuals_arr = np.array(self._residual_history)
            # Usar percentiles robustos en lugar de media/std
            median_r = np.median(residuals_arr)
            mad_r = np.median(np.abs(residuals_arr - median_r)) * 1.4826
            noise_floor = median_r + self.noise_gate_sigma * max(mad_r, 1e-8)
            
            if residual <= noise_floor:
                # Bajo compuerta -> 0.0 estricto
                raw_score = 0.0
            else:
                raw_score = residual - noise_floor
        else:
            raw_score = 0.0
        
        # Calibrador percentil interno
        if raw_score <= 0.0:
            return 0.0
        
        if len(self._raw_history) < self._min_history:
            import bisect
            bisect.insort(self._raw_history, raw_score)
            return 0.0
        
        import bisect
        menores = bisect.bisect_left(self._raw_history, raw_score)
        iguales = bisect.bisect_right(self._raw_history, raw_score) - menores
        n = len(self._raw_history)
        
        score = (menores + 0.5 * iguales + 0.5) / (n + 1)
        bisect.insort(self._raw_history, raw_score)
        
        return max(0.0, min(1.0, float(score)))

    def reset(self) -> None:
        """Reinicia el detector."""
        self._buffer.clear()
        self._step = 0
        self._stft_ready = False
        self._spectral_history.clear()
        self._residual_history.clear()
        self._raw_history.clear()
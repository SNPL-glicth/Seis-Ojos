"""
Ensemble: PeakDecay (v0.2 optimizado) + Causal Spectral Detector.

Combina el detector peak_decay optimizado (k=160, tau=1.5) con el detector
espectral causal independiente. La fusión es por máximo de scores calibrados:

    S_final = max(S_peak_decay, S_espectral)

Ambos detectores tienen sus propios PercentileCalibrators independientes,
lo que preserva la monotonía de la cola derecha requerida por el optimizador NAB.
"""

import math
from sixeyes.detectors.base import Detector
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.seasonal import DailySeasonalRobustZ
from sixeyes.detectors.causal_spectral import CausalSTFTDetector


class MasterSpectralPeakDecay(Detector):
    """
    Ensemble causal: PeakDecay (v0.2, 42.54 NAB) + Causal Spectral Detector.
    
    El detector espectral cubre el punto ciego de PeakDecay en anomalías
    de fase/frecuencia (tráfico, oscilaciones), mientras que PeakDecay
    mantiene la detección superior en spikes y cambios de nivel.
    
    Fusión por max() de scores calibrados independientemente -> preserva
    monotonía de cola derecha para el optimizador NAB.
    """

    def __init__(
        self,
        # PeakDecay v0.2 params
        k_window: int = 160,
        tau: float = 1.5,
        bucket_minutes: int = 15,
        # Spectral detector params
        spectral_window: int = 64,
        spectral_hop: int = 1,
        spectral_nperseg: int = 32,
        spectral_noise_gate_sigma: float = 3.5,
        spectral_calibrator_min_history: int = 288,
        # PeakDecay calibrator
        peak_calibrator_min_history: int = 288,
    ) -> None:
        """Inicializa el ensemble.
        
        Args:
            k_window: Ventana de memoria PeakDecay (default: 160 óptimo NAB).
            tau: Decaimiento exponencial PeakDecay (default: 1.5 óptimo NAB).
            bucket_minutes: Bucket estacional para DailySeasonalRobustZ.
            spectral_window: Ventana STFT (default 64).
            spectral_hop: Paso STFT (default 1).
            spectral_nperseg: Tamaño segmento FFT (default 32).
            spectral_noise_gate_sigma: Compuerta ruido espectral (default 3.5).
            spectral_calibrator_min_history: Historial calibrador espectral.
            peak_calibrator_min_history: Historial calibrador PeakDecay.
        """
        # === PeakDecay v0.2 (núcleo inviolable, 42.54 NAB) ===
        self.peak_decay = StreamingPeakDecay(
            PercentileCalibrator(DailySeasonalRobustZ(bucket_size_minutes=bucket_minutes)),
            k_window=k_window,
            tau=tau,
        )
        
        # === Detector Espectral Causal (independiente, su propio calibrador) ===
        self.spectral = CausalSTFTDetector(
            window_size=spectral_window,
            hop=spectral_hop,
            nperseg=spectral_nperseg,
            noise_gate_sigma=spectral_noise_gate_sigma,
            calibrator_min_history=spectral_calibrator_min_history,
        )

    def update(self, value: float, timestamp: str = "") -> float:
        """Procesa observación y retorna max(S_peak_decay, S_espectral)."""
        if math.isnan(value) or math.isinf(value):
            return 0.0
        
        if not timestamp:
            timestamp = "2026-10-12T00:00:00"
        
        # Ambos detectores corren independientemente con sus propios calibradores
        s_peak = self.peak_decay.update(value, timestamp)
        s_spectral = self.spectral.update(value, timestamp)
        
        # Fusión por máximo: preserva monotonía de cola de ambos calibradores
        return max(s_peak, s_spectral)

    def reset(self) -> None:
        """Reinicia ambos detectores."""
        self.peak_decay.reset()
        self.spectral.reset()
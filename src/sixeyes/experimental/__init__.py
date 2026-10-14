"""Detectores experimentales y legacy (excluidos del pipeline NAB estándar).

Estos módulos se mantienen únicamente con fines de investigación. No forman
parte del detector oficial congelado:

    DailySeasonalRobustZ(15min) + PercentileCalibrator + StreamingPeakDecay(K=12, tau=2.0)

Ningún script oficial (`run_nab.py`) los importa por defecto.
Legacy movido aquí: BiSeasonalRobustZ, MaxOfDetectors/Combinado,
DelayCoordinateEmbedding, CausalRampGradientTracker.
"""

from sixeyes.detectors.biseasonal import BiSeasonalRobustZ
from sixeyes.detectors.combinador import MaxOfDetectors, Combinado
from sixeyes.detectors.delay_embedding import DelayCoordinateEmbedding
from sixeyes.detectors.ramp_gradient import CausalRampGradientTracker

__all__ = [
    "GramSubspaceTracker",
    "ActiveCentroidDriftTracker",
    "CausalSTFTDetector",
    "CausalMultiscaleLDS",
    "MasterSpectralPeakDecay",
    "PeakDecayRampBoost",
    "BiSeasonalRobustZ",
    "MaxOfDetectors",
    "Combinado",
    "DelayCoordinateEmbedding",
    "CausalRampGradientTracker",
]

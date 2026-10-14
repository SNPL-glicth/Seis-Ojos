"""Detectores experimentales y no oficiales (excluidos del pipeline NAB estándar).

Estos módulos se mantienen únicamente con fines de investigación. No forman
parte del detector oficial congelado:

    SeasonalDualScaleRobustZ(15min) + PercentileCalibrator + StreamingPeakDecay(K=12, tau=2.0)

Ningún script oficial (`run_nab.py`) los importa por defecto.
No oficiales incluidos aquí: BiSeasonalRobustZ, MaxOfDetectors/Combinado,
DelayCoordinateEmbedding, CausalRampGradientTracker.
"""

from sixeyes.detectors.biseasonal import BiSeasonalRobustZ
from sixeyes.detectors.combinador import MaxOfDetectors, Combinado
from sixeyes.detectors.delay_embedding import DelayCoordinateEmbedding
from sixeyes.detectors.ramp_gradient import CausalRampGradientTracker
from sixeyes.experimental.adaptive_seasonal import AdaptiveSeasonalRobustZ
from sixeyes.experimental.dual_scale_z import (  # noqa: F401 (re-export de compatibilidad)
    DualScaleRobustZ,
    SeasonalDualScaleRobustZ,
)
from sixeyes.experimental.density_voter import HistogramDensityVoter
from sixeyes.experimental.consensus import CountConsensus
from sixeyes.experimental.predictor import HoltPredictor, ErrorLikelihood, PredictiveDetector
from sixeyes.experimental.modulated import ModulatedDetector
from sixeyes.experimental.noisy_or import NoisyOrFusion
from sixeyes.experimental.knn_detector import StreamingKNNDetector
from sixeyes.experimental.volatility_boost import VolatilityBoost
from sixeyes.experimental.hysteresis import HysteresisLatch
from sixeyes.experimental.regime_router import RegimeRouter
from sixeyes.experimental.rank_ensemble import RankEnsembleESN
from sixeyes.experimental.hybrid_orthogonal import OrthogonalHybrid
from sixeyes.experimental.global_norm_esn import GlobalNormESN
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.experimental.markov_transition import MarkovTransitionSurprise, quantize
from sixeyes.experimental.context_predictor import VariableOrderPredictor

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
    "AdaptiveSeasonalRobustZ",
    "DualScaleRobustZ",
    "SeasonalDualScaleRobustZ",
    "HistogramDensityVoter",
    "CountConsensus",
    "HoltPredictor",
    "ErrorLikelihood",
    "PredictiveDetector",
    "ModulatedDetector",
    "NoisyOrFusion",
    "StreamingKNNDetector",
    "VolatilityBoost",
    "HysteresisLatch",
    "RegimeRouter",
    "RankEnsembleESN",
    "OrthogonalHybrid",
    "GlobalNormESN",
    "EchoStatePredictor",
    "MarkovTransitionSurprise",
    "quantize",
    "VariableOrderPredictor",
]

"""State Transition Engine (STE): sorpresa de transicion sobre estados densos.

Difiere de los intentos falsados: Markov-orden-1 operaba sobre simbolos
escalares pobres (14.36) y Delta-h descartaba el readout aprendido (21.61).
Aqui la representacion es densa (reservorio R^64, mismas constantes que
EchoStatePredictor) y la estructura aprendida es la matriz de transicion
P(S_t|S_{t-1}) sobre prototipos adaptativos.
"""

from sixeyes.transition.reservoir_state import ReservoirState
from sixeyes.transition.state_quantizer import OnlinePrototypeQuantizer
from sixeyes.transition.transition_model import TransitionModel
from sixeyes.transition.transition_scorer import TransitionSurpriseDetector

__all__ = [
    "ReservoirState",
    "OnlinePrototypeQuantizer",
    "TransitionModel",
    "TransitionSurpriseDetector",
]

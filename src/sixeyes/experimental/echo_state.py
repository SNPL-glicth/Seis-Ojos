"""Re-export de compatibilidad: el predictor denso vive en `sixeyes.detectors.echo_state`.
Este módulo se conserva para no romper imports.
"""

from sixeyes.detectors.echo_state import EchoStatePredictor

__all__ = ["EchoStatePredictor"]

"""Reservorio denso + lectura RLS (base oficial del pipeline).

Ver diario (proyecto denso): NAB 48.62 vs 45.19 (+3.43), determinista.
`EchoStatePredictor` sustituye la rareza puntual por error de predicción
geométrico; el resto del pipeline (calibrador + PeakDecay) no cambia.

Responde al diagnóstico de las actas markovianas: la cuantización escalar
destruye la geometría antes de la memoria. Aquí no hay símbolos: el vector
de retardo (Takens, d=16) se proyecta a un reservorio recurrente fijo
(ESN, Nres=64, radio espectral 0.9) y una lectura lineal RLS predice el
siguiente punto. Anomalía = error cuadrático de predicción en espacio
latente, normalizado aguas abajo por el calibrador ECDF.

Diseño (constantes a priori):
- d=16, stride=1 (`DelayCoordinateEmbedding` reutilizado).
- Nres=64, semilla fija 42 (determinismo bit a bit), radio espectral 0.9
  (propiedad de eco), tanh, escala de entrada 1.0.
- Estandarización causal rodante (W=6400, sumas O(1)) antes del reservorio.
- RLS con olvido λ=0.999, P₀=I (delta=1.0). O(Nres²)=O(4096) flops/punto.

Causalidad: pronostica con el estado del pasado, puntúa el error y DESPUÉS
actualiza RLS y reservorio. Sin futuro en ningún punto.
"""

import math
from collections import deque

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.detectors.delay_embedding import DelayCoordinateEmbedding

EMBED_DIM = 16
RESERVOIR_SIZE = 64
SPECTRAL_RADIUS = 0.9
INPUT_SCALE = 1.0
RLS_DELTA = 1.0
RLS_FORGET = 0.999
NORM_WINDOW = 6400
SEED = 42


class EchoStatePredictor(Detector):
    """Error de predicción ESN-RLS (cuadrático ≥0 por defecto; con signo si
    ``signed_output=True``, para alimentar likelihoods gaussianas)."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        reservoir_size: int = RESERVOIR_SIZE,
        spectral_radius: float = SPECTRAL_RADIUS,
        seed: int = SEED,
        signed_output: bool = False,
        time_features: bool = False,
    ) -> None:
        """Args:
            signed_output: False (defecto, compatible) = error cuadrático;
                True = residuo con signo, para likelihoods gaussianas.
            time_features: False (defecto, compatible) = solo retardos;
                True = añade seno/coseno de hora y día semanal (d+4).
        """
        self._n = reservoir_size
        self._embed = DelayCoordinateEmbedding(dimension=embed_dim, stride=1)
        self._time_features = time_features
        in_dim = embed_dim + 4 if time_features else embed_dim
        rng = np.random.RandomState(seed)
        self._win = rng.uniform(-1.0, 1.0, size=(reservoir_size, in_dim)) * INPUT_SCALE
        w = rng.uniform(-1.0, 1.0, size=(reservoir_size, reservoir_size))
        radius = max(abs(np.linalg.eigvals(w)).max(), 1e-9)
        self._wres = w * (spectral_radius / radius)
        self._h = np.zeros(reservoir_size)
        self._wout = np.zeros(reservoir_size)
        self._P = np.eye(reservoir_size) * RLS_DELTA
        self._lam = RLS_FORGET
        self._signed_output = signed_output
        self._norm: deque[float] = deque(maxlen=NORM_WINDOW)
        self._nsum = 0.0
        self._nsumsq = 0.0

    def _standardize(self, value: float) -> float:
        n = len(self._norm)
        if n < 2:
            return 0.0
        mean = self._nsum / n
        var = max(self._nsumsq / n - mean * mean, 0.0)
        std = math.sqrt(var)
        if std <= 1e-9:
            return 0.0
        return (value - mean) / std

    def _ingest_norm(self, value: float) -> None:
        if len(self._norm) == self._norm.maxlen:
            old = self._norm[0]
            self._nsum -= old
            self._nsumsq -= old * old
        self._norm.append(value)
        self._nsum += value
        self._nsumsq += value * value

    @staticmethod
    def _time_vector(timestamp: str) -> np.ndarray:
        """[sin_h, cos_h, sin_dow, cos_dow]; ceros si no parseable."""
        import math as _m
        from datetime import datetime as _dt
        try:
            d = _dt.fromisoformat(timestamp.replace(" ", "T"))
        except ValueError:
            try:
                from dateutil import parser as _p
                d = _p.parse(timestamp)
            except Exception:
                return np.zeros(4)
        h = (d.hour * 60 + d.minute) / 1440.0
        w = d.weekday() / 7.0
        return np.array([_m.sin(2 * _m.pi * h), _m.cos(2 * _m.pi * h),
                         _m.sin(2 * _m.pi * w), _m.cos(2 * _m.pi * w)])

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        z = self._standardize(value)
        vec, ready = self._embed.update(z, timestamp)
        if not ready or vec is None:
            self._ingest_norm(value)
            return 0.0
        if self._time_features:
            vec = np.concatenate([vec, self._time_vector(timestamp)])
        h_new = np.tanh(self._win @ vec + self._wres @ self._h)
        pred_z = float(self._wout @ h_new)
        if not math.isfinite(pred_z) or not bool(np.all(np.isfinite(self._P))):
            # Anti-windup (2026-10-15, caso 550_SWaT): covarianza divergida en tramos
            # degenerados. Se re-ancla el readout (w=0, P=I) y se puntúa contra pred=0:
            # el punto se emite como error extremo, no como silencio ni excepción.
            self._wout = np.zeros(self._n)
            self._P = np.eye(self._n) * RLS_DELTA
            pred_z = 0.0
        resid = z - pred_z
        # Techo numérico honesto: el residuo se capa en vez de desbordar
        # (`resid ** 2` lanzaba OverflowError). La ECDF lo ordena como extremo.
        if math.isnan(resid):
            err = 0.0
        elif self._signed_output:
            err = max(-1e6, min(1e6, resid))
        else:
            err = min(resid * resid, 1e12) if math.isfinite(resid) else 1e12
        # RLS DESPUÉS de puntuar (causalidad).
        h = h_new.reshape(-1, 1)
        Ph = self._P @ h
        denom = self._lam + float((h.T @ Ph)[0, 0])
        K = Ph / max(denom, 1e-12)
        self._wout = (self._wout.reshape(-1, 1) + K * (z - pred_z)).ravel()
        self._P = (self._P - (K @ h.T) @ self._P) / self._lam
        self._h = h_new
        self._ingest_norm(value)
        if self._signed_output:
            return float(err)
        return max(float(err), 0.0)

    def reset(self) -> None:
        """Limpia reservorio, RLS, embedding y normalización."""
        self._embed.reset()
        self._h = np.zeros(self._n)
        self._wout = np.zeros(self._n)
        self._P = np.eye(self._n) * RLS_DELTA
        self._norm.clear()
        self._nsum = 0.0
        self._nsumsq = 0.0

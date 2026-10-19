"""Memoria episódica causal para regímenes dinámicos en ESN (Fase V, experimental).

Diseñada para comparar el régimen actual con patrones anteriores sin confundir
repetición con legitimidad ni usar etiquetas futuras.

Firma de Régimen Multimodal Causal:
    phi = (mu_x, sigma_x, psi_h)
    donde:
    - mu_x, sigma_x: nivel y dispersión observada de la serie temporal.
    - psi_h = E[|h_t|]: envolvente de amplitud de las activaciones del reservorio.

Métrica de Distancia Normalizada:
    d(t, k) = sqrt( 0.5 * ((mu_t - mu_k) / sigma_k)^2 + 0.5 * (||psi_t - psi_k|| / ||psi_k||)^2 )

Clasificación de Régimen:
    - d(t, k) <= 1.25: Régimen conocido k (ej. normalidad previa o modo operativo conocido).
    - d(t, k) > 1.50 para todo k: Estado no cartografiado (cuarentena).
    - Auto-promoción: Solo tras persistir T_confirm pasos consecutivos en un estado estable
      se auto-cristaliza como un nuevo régimen legítimo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

_EPS = 1e-9


@dataclass
class RegimeSignature:
    """Firma estadística compacta de un régimen dinámico."""
    regime_id: int
    name: str
    mu_x: float                   # Nivel medio de la serie en el régimen
    sigma_x: float                # Dispersión / volatilidad de la serie
    latent_envelope: np.ndarray   # bar{|h|} in R^N
    observed_steps: int = 0
    created_at_step: int = 0
    last_seen_step: int = 0

    def distance_to(self, mu: float, sigma: float, psi: np.ndarray) -> float:
        """Distancia adimensional normalizada respecto a esta firma."""
        # Componente de nivel
        d_level = abs(mu - self.mu_x) / max(self.sigma_x, _EPS)
        # Componente latente de envolvente
        norm_psi_k = float(np.linalg.norm(self.latent_envelope))
        d_latent = float(np.linalg.norm(psi - self.latent_envelope)) / max(norm_psi_k, _EPS)
        
        # Distancia euclídea compuesta
        return math.sqrt(0.5 * (d_level ** 2) + 0.5 * (d_latent ** 2))


class EpisodicRegimeMemory:
    """Gestor causal de memoria episódica para series temporales con ESN."""

    def __init__(
        self,
        reservoir_dim: int,
        quarantine_steps: int = 250,
        smoothing_gamma: float = 0.05,
    ) -> None:
        self.reservoir_dim = int(reservoir_dim)
        self.quarantine_steps = int(quarantine_steps)
        self.smoothing_gamma = float(smoothing_gamma)
        
        self.regimes: Dict[int, RegimeSignature] = {}
        self._step = 0
        self._quarantine_counter = 0
        self._smooth_mu: Optional[float] = None
        self._smooth_sigma: Optional[float] = None
        self._smooth_psi: Optional[np.ndarray] = None
        self._tentative_samples: List[Tuple[float, float, np.ndarray]] = []

    def reset(self) -> None:
        self.regimes.clear()
        self._step = 0
        self._quarantine_counter = 0
        self._smooth_mu = None
        self._smooth_sigma = None
        self._smooth_psi = None
        self._tentative_samples.clear()

    def register_regime(
        self,
        name: str,
        x_samples: List[float],
        h_samples: List[np.ndarray],
        regime_id: Optional[int] = None,
    ) -> int:
        """Registra un régimen a partir de una secuencia causal de observaciones."""
        if not x_samples or not h_samples:
            raise ValueError("Las muestras no pueden estar vacías")
        
        mu_x = float(np.mean(x_samples))
        sigma_x = float(np.std(x_samples))
        sigma_x = max(sigma_x, 0.05)
        
        h_arr = np.array(h_samples)
        latent_envelope = np.mean(np.abs(h_arr), axis=0)
        
        rid = len(self.regimes) if regime_id is None else int(regime_id)
        self.regimes[rid] = RegimeSignature(
            regime_id=rid,
            name=name,
            mu_x=mu_x,
            sigma_x=sigma_x,
            latent_envelope=latent_envelope,
            observed_steps=len(x_samples),
            created_at_step=self._step,
            last_seen_step=self._step,
        )
        return rid

    def update(self, x_t: float, h_t: np.ndarray) -> Tuple[Optional[int], float, str]:
        """Procesa la observación causal en t y clasifica el régimen actual.
        
        Retorna:
            (best_regime_id, normalized_distance, status)
            donde status in {"KNOWN", "UNCERTAIN_QUARANTINE", "NEW_PROMOTED"}.
        """
        self._step += 1
        x = float(x_t)
        h = np.asarray(h_t, dtype=float)
        psi = np.abs(h)
        
        # Filtro exponencial recursivo para estimar nivel y envolvente locales
        if self._smooth_mu is None:
            self._smooth_mu = x
            self._smooth_sigma = 0.5
            self._smooth_psi = psi.copy()
        else:
            self._smooth_mu = (1.0 - self.smoothing_gamma) * self._smooth_mu + self.smoothing_gamma * x
            err_mu = abs(x - self._smooth_mu)
            self._smooth_sigma = (1.0 - self.smoothing_gamma) * self._smooth_sigma + self.smoothing_gamma * (err_mu * 1.2533)
            self._smooth_psi = (1.0 - self.smoothing_gamma) * self._smooth_psi + self.smoothing_gamma * psi

        mu = self._smooth_mu
        sigma = max(self._smooth_sigma, 0.05)
        psi_smooth = self._smooth_psi

        if not self.regimes:
            return None, float("inf"), "UNCERTAIN_QUARANTINE"

        # Evaluar distancia causal a todos los regímenes conocidos
        best_id: Optional[int] = None
        min_dist = float("inf")
        for rid, sig in self.regimes.items():
            d = sig.distance_to(mu, sigma, psi_smooth)
            if d < min_dist:
                min_dist = d
                best_id = rid

        # Clasificación causal
        if min_dist <= 1.25 and best_id is not None:
            self.regimes[best_id].last_seen_step = self._step
            self.regimes[best_id].observed_steps += 1
            self._quarantine_counter = 0
            self._tentative_samples.clear()
            return best_id, min_dist, "KNOWN"

        # Estado en cuarentena
        self._quarantine_counter += 1
        self._tentative_samples.append((mu, sigma, psi_smooth.copy()))
        if len(self._tentative_samples) > self.quarantine_steps:
            self._tentative_samples.pop(0)

        # Si el nuevo estado persiste coherentemente durante quarantine_steps, auto-promocionar
        if self._quarantine_counter >= self.quarantine_steps:
            x_vals = [s[0] for s in self._tentative_samples]
            h_vals = [s[2] for s in self._tentative_samples]
            new_id = self.register_regime(
                name=f"Regime_{len(self.regimes)}",
                x_samples=x_vals,
                h_samples=h_vals,
            )
            self._quarantine_counter = 0
            self._tentative_samples.clear()
            return new_id, 0.0, "NEW_PROMOTED"

        return best_id, min_dist, "UNCERTAIN_QUARANTINE"

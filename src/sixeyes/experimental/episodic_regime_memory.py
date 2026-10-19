"""Memoria episódica causal para regímenes dinámicos en ESN (Fase V/VI, experimental).

Diseñada para comparar el régimen actual con patrones anteriores sin confundir
repetición con legitimidad ni usar etiquetas futuras.

Principios:
1. Distinción epistemológica: La persistencia y estabilidad estadística de un estado
   no demuestran su legitimidad. La promoción se clasifica formalmente como
   'STABLE_PROMOTED' (régimen nuevo estadísticamente estacionario), no como normalidad.
2. Firma de Régimen Multimodal Causal (tres momentos):
   phi = (mu_x, sigma_x, psi_h)
   - mu_x, sigma_x: nivel y dispersión observada de la serie temporal.
   - psi_h = E[|h_t|]: envolvente de amplitud de las activaciones del reservorio.
3. Métrica de Distancia Normalizada Equilibrada:
   d(t, k) = sqrt( 1/3 * ((mu - mu_k)/sigma_k)^2 + 1/3 * ((sigma - sigma_k)/sigma_k)^2 + 1/3 * (||psi - psi_k||/||psi_k||)^2 )
4. Clasificación con Histéresis Causal:
   - d <= 1.25: 'KNOWN' (reconocimiento inequívoco).
   - 1.25 < d <= 1.50: 'AMBIGUOUS_BORDER' (zona neutra: pausa contador de cuarentena).
   - d > 1.50: 'UNCERTAIN_QUARANTINE' (desviación: acumula sospecha hacia promoción).
5. Capacidad Acotada y Política LRU:
   - Max K = 5 regímenes.
   - 'Nominal_Regime_0' (id 0) está blindado contra expulsión.
   - Al insertar el sexto, se expulsa el menos recientemente utilizado (menor last_seen_step),
     con desempate determinista por menor regime_id.
"""

from __future__ import annotations

import math
from collections import deque
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
        """Distancia adimensional normalizada sobre nivel, dispersión y envolvente latente."""
        sigma_ref = max(self.sigma_x, _EPS)
        norm_psi_ref = max(float(np.linalg.norm(self.latent_envelope)), _EPS)

        # 1. Componente de nivel
        d_level = abs(mu - self.mu_x) / sigma_ref
        # 2. Componente de dispersión/escala
        d_scale = abs(sigma - self.sigma_x) / sigma_ref
        # 3. Componente latente de envolvente
        d_latent = float(np.linalg.norm(psi - self.latent_envelope)) / norm_psi_ref

        # Distancia euclídea media de tres componentes
        return math.sqrt((d_level ** 2 + d_scale ** 2 + d_latent ** 2) / 3.0)


class EpisodicRegimeMemory:
    """Gestor causal de memoria episódica con capacidad acotada y validación de estabilidad."""

    def __init__(
        self,
        reservoir_dim: int,
        max_regimes: int = 5,
        quarantine_steps: int = 250,
        smoothing_gamma: float = 0.05,
        d_known: float = 1.25,
        d_uncertain: float = 1.50,
        stability_volatility_max: float = 2.0,
        stability_drift_max: float = 0.35,
    ) -> None:
        if reservoir_dim < 1:
            raise ValueError("reservoir_dim debe ser >= 1")
        if max_regimes < 2:
            raise ValueError("max_regimes debe ser >= 2 para alojar base y alternativos")
        if quarantine_steps < 20:
            raise ValueError("quarantine_steps debe ser >= 20")
        if not (0.0 < smoothing_gamma <= 1.0):
            raise ValueError("smoothing_gamma debe estar en (0, 1]")
        if not (0.0 < d_known < d_uncertain):
            raise ValueError("Debe cumplirse 0 < d_known < d_uncertain")

        self.reservoir_dim = int(reservoir_dim)
        self.max_regimes = int(max_regimes)
        self.quarantine_steps = int(quarantine_steps)
        self.smoothing_gamma = float(smoothing_gamma)
        self.d_known = float(d_known)
        self.d_uncertain = float(d_uncertain)
        self.stability_volatility_max = float(stability_volatility_max)
        self.stability_drift_max = float(stability_drift_max)

        self.regimes: Dict[int, RegimeSignature] = {}
        self._next_regime_id = 0
        self._step = 0
        self._quarantine_counter = 0

        # Filtros recursivos para seguimiento de momento contemporáneo
        self._smooth_mu: Optional[float] = None
        self._smooth_sigma: Optional[float] = None
        self._smooth_psi: Optional[np.ndarray] = None

        # Buffers circulares de observaciones reales para evaluación de estabilidad
        self._tentative_x: deque[float] = deque(maxlen=self.quarantine_steps)
        self._tentative_h: deque[np.ndarray] = deque(maxlen=self.quarantine_steps)

    def reset(self) -> None:
        self.regimes.clear()
        self._next_regime_id = 0
        self._step = 0
        self._quarantine_counter = 0
        self._smooth_mu = None
        self._smooth_sigma = None
        self._smooth_psi = None
        self._tentative_x.clear()
        self._tentative_h.clear()

    def register_regime(
        self,
        name: str,
        x_samples: List[float],
        h_samples: List[np.ndarray],
        regime_id: Optional[int] = None,
    ) -> int:
        """Registra una firma de régimen a partir de observaciones crudas verificadas."""
        if not x_samples or not h_samples:
            raise ValueError("Las muestras no pueden estar vacías")
        if len(x_samples) != len(h_samples):
            raise ValueError("x_samples y h_samples deben tener igual longitud")

        # Control de capacidad acotada K con desalojo LRU (protegiendo id 0)
        if len(self.regimes) >= self.max_regimes:
            evict_candidates = [
                (sig.last_seen_step, rid)
                for rid, sig in self.regimes.items()
                if rid != 0  # Invariante 2: Nominal_Regime_0 nunca se desaloja
            ]
            if evict_candidates:
                # Desempate determinista: menor last_seen_step, luego menor rid
                evict_candidates.sort(key=lambda item: (item[0], item[1]))
                evict_id = evict_candidates[0][1]
                del self.regimes[evict_id]

        mu_x = float(np.mean(x_samples))
        sigma_x = float(np.std(x_samples))
        sigma_x = max(sigma_x, 0.05)

        h_arr = np.array(h_samples)
        latent_envelope = np.mean(np.abs(h_arr), axis=0)

        rid = self._next_regime_id if regime_id is None else int(regime_id)
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
        if regime_id is None or regime_id >= self._next_regime_id:
            self._next_regime_id = max(self._next_regime_id, rid + 1)
        return rid

    def _verify_stability(self) -> bool:
        """Verifica que el estado en cuarentena haya alcanzado equilibrio estacionario."""
        n_samples = len(self._tentative_x)
        if n_samples < min(50, self.quarantine_steps):
            return False

        recent_x = np.array(list(self._tentative_x)[-50:])
        sigma_recent = float(np.std(recent_x))

        # Comparar contra dispersión del régimen base (si existe)
        nominal_sigma = self.regimes[0].sigma_x if 0 in self.regimes else 0.5
        # 1. Criterio de volatilidad acotada
        if sigma_recent > self.stability_volatility_max * nominal_sigma:
            return False

        # 2. Criterio de deriva nula: comprobar que no exista tendencia lineal activa
        half = len(recent_x) // 2
        mu_first_half = float(np.mean(recent_x[:half]))
        mu_second_half = float(np.mean(recent_x[half:]))
        drift_norm = abs(mu_second_half - mu_first_half) / max(sigma_recent, _EPS)
        if drift_norm > self.stability_drift_max:
            return False

        return True

    def update(self, x_t: float, h_t: np.ndarray) -> Tuple[Optional[int], float, str]:
        """Procesa causalmente la observación x_t y el vector h_t.

        Retorna:
            (best_regime_id, actual_distance, status)
            donde status in {"KNOWN", "AMBIGUOUS_BORDER", "UNCERTAIN_QUARANTINE", "STABLE_PROMOTED"}.
        """
        # Validación estricta de entradas
        if not math.isfinite(x_t):
            return None, float("inf"), "INVALID_INPUT"
        h = np.asarray(h_t, dtype=float)
        if h.shape != (self.reservoir_dim,) or not np.all(np.isfinite(h)):
            return None, float("inf"), "INVALID_INPUT"

        self._step += 1
        x = float(x_t)
        psi = np.abs(h)

        # Actualización de estimadores recursivos para clasificación contemporánea
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

        # 1. Regla: Régimen Conocido Inequívoco (d <= d_known)
        if min_dist <= self.d_known and best_id is not None:
            self.regimes[best_id].last_seen_step = self._step
            self.regimes[best_id].observed_steps += 1
            # Resetear sospecha de nuevo régimen
            self._quarantine_counter = 0
            self._tentative_x.clear()
            self._tentative_h.clear()
            return best_id, min_dist, "KNOWN"

        # 2. Regla: Franja Ambigua (d_known < d <= d_uncertain)
        if min_dist <= self.d_uncertain:
            # Pausa el contador de cuarentena sin borrar evidencia acumulada previa
            return best_id, min_dist, "AMBIGUOUS_BORDER"

        # 3. Regla: Desviación Inequívoca (d > d_uncertain)
        self._quarantine_counter += 1
        self._tentative_x.append(x)
        self._tentative_h.append(h.copy())

        # Auto-promoción: requiere persistencia temporal Y estabilidad comprobada
        if self._quarantine_counter >= self.quarantine_steps:
            if self._verify_stability():
                new_id = self.register_regime(
                    name=f"Stable_Regime_{self._next_regime_id}",
                    x_samples=list(self._tentative_x),
                    h_samples=list(self._tentative_h),
                )
                self._quarantine_counter = 0
                self._tentative_x.clear()
                self._tentative_h.clear()
                actual_dist = self.regimes[new_id].distance_to(mu, sigma, psi_smooth)
                return new_id, actual_dist, "STABLE_PROMOTED"

        return best_id, min_dist, "UNCERTAIN_QUARANTINE"

    def get_memory_footprint_bytes(self) -> int:
        """Calcula el consumo de memoria real en bytes de los prototipos y buffers."""
        bytes_total = 0
        # Firmas de regímenes almacenados
        for sig in self.regimes.values():
            bytes_total += sig.latent_envelope.nbytes
            bytes_total += 8 * 5  # floats y enteros (mu, sigma, observed, created, last_seen)
        # Buffers tentativos circulares
        bytes_total += len(self._tentative_x) * 8
        for h_vec in self._tentative_h:
            bytes_total += h_vec.nbytes
        return bytes_total

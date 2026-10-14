"""BOCPD: Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

Mide P(cambio de distribución en t | x_{1:t}) — la probabilidad posterior
de que el proceso generativo CAMBIÓ en el instante actual. Difiere
fundamentalmente del ESN:

    ESN:    score_t = ‖x_t − ŷ_t‖²        rareza del PUNTO
    BOCPD:  score_t = P(r_t=0 | x_{1:t})   rareza de la TRANSICIÓN

Modelo local: Normal-Inverse-Gamma conjugado (NIGamma). En cada paso t:
  1. Predictive: P(x_t | r, x_{t-r:t-1}) — Student-t predictiva
  2. Growth:     P(r_t = r+1) ∝ P(x_t | r) × (1 − h)
  3. Changepoint: P(r_t = 0) ∝ Σ_r P(x_t | r) × h × π₀(r)
  4. Normalización + score = P(r_t = 0 | x_{1:t})

Parámetros (fijos a priori, sin barrer):
  - Hazard h = 1/250: prior de duración de régimen ~250 puntos
    (~21h en series de 5min; difuso y no-informativo).
  - Prior NIGamma: μ₀=0, κ₀=1, α₀=1, β₀=1 (vago, estándar).
  - Truncamiento T=500: solo los últimos 500 run-lengths (memoria O(500)).

Causalidad: P(x_t) se evalúa ANTES de actualizar los suficientes
estadísticos. Score emitido antes de que t entre en el prior.
O(T) por punto con T=500.
"""

import math
from collections import deque
from typing import NamedTuple

import numpy as np

from sixeyes.detectors.base import Detector

# Parámetros del prior Normal-Inverse-Gamma (vago, no-informativo).
# κ₀ pequeño = prior débil sobre μ → puntos extremos desplazan más el CP.
MU_0 = 0.0
KAPPA_0 = 0.5     # debilitado para que outliers impacten más la predictiva
ALPHA_0 = 1.0
BETA_0 = 1.0

# Hazard uniforme: 1/λ (λ = duración esperada del régimen en puntos).
HAZARD_LAMBDA = 250

# Truncamiento: solo mantener los últimos T run-lengths activos.
MAX_RUN_LENGTH = 500

# Estandarización causal rodante antes de BOCPD (igual que ESN).
# Sin esto, los valores crudos de NAB (rango 0–10000+) saturan la predictiva.
NORM_WINDOW = 500
NORM_MIN = 50     # mínimo de obs para estandarizar


class _NIGSufStats(NamedTuple):
    """Suficientes estadísticos del run length r."""
    n: int        # número de observaciones en este run
    mu: float     # media posterior
    kappa: float  # precisión del prior de μ
    alpha: float  # forma del prior de σ²
    beta: float   # escala del prior de σ²


def _nig_update(ss: _NIGSufStats, x: float) -> _NIGSufStats:
    """Actualiza los suficientes estadísticos NIGamma con una nueva obs."""
    n = ss.n + 1
    kappa = ss.kappa + 1.0
    alpha = ss.alpha + 0.5
    mu = (ss.kappa * ss.mu + x) / kappa
    beta = ss.beta + (ss.kappa * (x - ss.mu) ** 2) / (2.0 * kappa)
    return _NIGSufStats(n=n, mu=mu, kappa=kappa, alpha=alpha, beta=beta)


def _nig_pred_log_prob(ss: _NIGSufStats, x: float) -> float:
    """Log P(x | r) bajo la distribución predictiva Student-t de NIGamma."""
    # Student-t con 2α grados de libertad, media μ, varianza β(κ+1)/(ακ).
    alpha = ss.alpha
    kappa = ss.kappa
    beta = ss.beta
    mu = ss.mu
    df = 2.0 * alpha
    scale_sq = beta * (kappa + 1.0) / (alpha * kappa)
    scale = math.sqrt(max(scale_sq, 1e-12))
    # log Student-t(x; μ, scale, df)
    z = (x - mu) / scale
    log_p = (
        math.lgamma((df + 1.0) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * math.log(df * math.pi)
        - math.log(scale)
        - ((df + 1.0) / 2.0) * math.log(1.0 + z * z / df)
    )
    return log_p


class BOCPDDetector(Detector):
    """Detector de cambio de punto Bayesiano online.

    Score = P(changepoint en t | x_{1:t}) ∈ [0, 1].
    Mayor score = mayor probabilidad de que el proceso cambió en t.
    Estandariza causalmente la entrada (media/std rodantes W=500) para
    mantener el modelo NIGamma en escala unitaria independientemente de
    los valores crudos de NAB (rango 0–10000+).
    """

    def __init__(
        self,
        hazard_lambda: int = HAZARD_LAMBDA,
        max_run_length: int = MAX_RUN_LENGTH,
        mu_0: float = MU_0,
        kappa_0: float = KAPPA_0,
        alpha_0: float = ALPHA_0,
        beta_0: float = BETA_0,
        norm_window: int = NORM_WINDOW,
    ) -> None:
        self._h = 1.0 / hazard_lambda
        self._max_rl = max_run_length
        self._prior = _NIGSufStats(n=0, mu=mu_0, kappa=kappa_0,
                                   alpha=alpha_0, beta=beta_0)
        # log-probabilidades de cada run-length: índice i = run-length i.
        # Empezamos con P(r_0 = 0) = 1.
        self._log_probs: list[float] = [0.0]   # log P(r_t = i)
        self._suf_stats: list[_NIGSufStats] = [self._prior]
        # Estandarización causal rodante.
        from collections import deque
        self._norm: deque[float] = deque(maxlen=norm_window)
        self._nsum = 0.0
        self._nsumsq = 0.0
        self._norm_min = NORM_MIN

    def _standardize(self, value: float) -> float:
        n = len(self._norm)
        if n < self._norm_min:
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

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0

        # Estandarización causal ANTES de actualizar la norma (sin fuga).
        x = self._standardize(value)
        self._ingest_norm(value)
        # Durante el warmup de la norma, emitimos 0 pero igualmente
        # actualizamos el BOCPD con x=0 para acumular historia.
        h = self._h
        log_h = math.log(h)
        log_1mh = math.log(1.0 - h)

        # 1. Calcular log-predictiva P(x | r) para cada run-length activo.
        log_preds = [_nig_pred_log_prob(ss, x) for ss in self._suf_stats]

        # 2. Calcular log-joint P(x, r_t) antes de normalizar.
        #    P(r_t = r+1) ∝ P(x | r) × (1-h) × P(r_{t-1} = r)  [growth]
        #    P(r_t = 0)   ∝ Σ_r P(x | r) × h × P(r_{t-1} = r)  [change]

        # Acumulador del changepoint (r_t = 0): log-sum-exp de contribuciones.
        log_cp_terms = []
        log_growth = []
        for i, (lp, lpr) in enumerate(zip(log_preds, self._log_probs)):
            log_cp_terms.append(lp + log_h + lpr)
            log_growth.append(lp + log_1mh + lpr)  # run-length i+1 en t

        # Log-sum-exp para el changepoint.
        def logsumexp(vals: list[float]) -> float:
            if not vals:
                return -math.inf
            m = max(vals)
            return m + math.log(sum(math.exp(v - m) for v in vals))

        log_cp = logsumexp(log_cp_terms)

        # Log de la predictiva MARGINAL P(x_t | x_{1:t-1}) = log-sum-exp de
        # P(x_t | r) × P(r_{t-1} = r) sobre todos los run-lengths.
        # Es la sorpresa del punto bajo el mixture model de regímenes.
        log_pred_marginal = logsumexp(
            [lp + lpr for lp, lpr in zip(log_preds, self._log_probs)]
        )

        # 3. Nuevo vector de log-probs (changepoint + growths).
        new_log_probs = [log_cp] + log_growth

        # 4. Normalizar.
        log_norm = logsumexp(new_log_probs)
        new_log_probs = [lp - log_norm for lp in new_log_probs]

        # 5. Score = sorpresa predictiva = −log P(x_t | pasado).
        # Mayor valor = punto más sorprendente bajo el modelo bayesiano actual.
        # Clipeamos a [0, ∞) y dejamos que el calibrador ECDF lo normalice.

        # 6. Actualizar suficientes estadísticos:
        #    - r_t = 0: prior fresco (nuevo régimen)
        #    - r_t = i+1: actualizar con x el suficiente estadístico i
        new_suf = [self._prior] + [_nig_update(ss, x) for ss in self._suf_stats]

        # 7. Truncar si supera max_run_length.
        if len(new_log_probs) > self._max_rl + 1:
            # Mantener los max_rl+1 más probables (los más recientes son
            # más informativos; los run-lengths muy largos tienen masa ~0).
            new_log_probs = new_log_probs[:self._max_rl + 1]
            new_suf = new_suf[:self._max_rl + 1]
            # Re-normalizar tras truncar.
            log_norm2 = logsumexp(new_log_probs)
            new_log_probs = [lp - log_norm2 for lp in new_log_probs]

        self._log_probs = new_log_probs
        self._suf_stats = new_suf

        # Sorpresa = −log P(x_t | pasado). Más alto = más anómalo.
        # El calibrador ECDF downstream convierte a percentil [0,1].
        score = max(0.0, -log_pred_marginal)
        return score

    def reset(self) -> None:
        """Reinicia el detector a su estado inicial."""
        self._log_probs = [0.0]
        self._suf_stats = [self._prior]
        self._norm.clear()
        self._nsum = 0.0
        self._nsumsq = 0.0

"""RLS ponderado por observación con olvido efectivo acoplado (Fase III, experimental).

Formulación (peso de observación eta en [0, 1], olvido base lam):

    lam_eff = 1 - eta * (1 - lam)
    K*      = P h / (lam_eff + eta * h^T P h)
    w_t     = w_{t-1} + eta * K* * e_t
    P_t     = (P_{t-1} - eta * K* h^T P_{t-1}) / lam_eff

Lema (forma de información, Sherman-Morrison):

    P_t^{-1} = lam_eff * P_{t-1}^{-1} + eta * h h^T
    b_t      = lam_eff * b_{t-1}      + eta * y_t h_t,   w_t = P_t b_t

Es decir, es exactamente mínimos cuadrados ponderados exponencialmente, con peso
eta_t para la observación t y descuento lam_eff_t. Consecuencias:
  - eta = 1  -> RLS estándar (lam_eff = lam).
  - eta = 0  -> P_t = P_{t-1}, w_t = w_{t-1} (sin información ni olvido).
  - P_t es definida positiva en aritmética exacta para todo eta en [0, 1]
    (suma de PD y PSD). En coma flotante se verifica numéricamente en los tests.

Paridad: WeightedRLSHead delega en _RLSHead.update cuando eta >= 1, de modo que
con eta = 1 el comportamiento es bit a bit el de la línea base.
"""

from __future__ import annotations

import math

import numpy as np

from sixeyes.experimental.predictive_readout_esn import RLS_FORGET, _RLSHead


def effective_forgetting(lam: float, eta: float) -> float:
    """Olvido efectivo acoplado a la ganancia de plasticidad."""
    return 1.0 - eta * (1.0 - lam)


def weighted_rls_step(
    w: np.ndarray,
    P: np.ndarray,
    h: np.ndarray,
    e: float,
    lam: float,
    eta: float,
    symmetrize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Un paso de RLS ponderado. Función pura (no modifica w ni P).

    Con eta = 0 devuelve copias exactas de w y P (no se simetriza para no
    alterar una P de entrada ligeramente asimétrica).
    """
    if not (0.0 <= eta <= 1.0) or math.isnan(eta):
        raise ValueError("eta debe estar en [0, 1]")
    if not (0.0 < lam <= 1.0):
        raise ValueError("lam debe estar en (0, 1]")
    if eta == 0.0:
        return w.copy(), P.copy()
    lam_eff = effective_forgetting(lam, eta)
    h_col = h.reshape(-1, 1)
    Ph = P @ h_col
    denom = lam_eff + eta * float((h_col.T @ Ph)[0, 0])
    K = Ph / max(denom, 1e-12)
    w_new = (w.reshape(-1, 1) + eta * K * e).ravel()
    P_new = (P - eta * (K @ h_col.T) @ P) / lam_eff
    if symmetrize:
        P_new = 0.5 * (P_new + P_new.T)
    return w_new, P_new


class WeightedRLSHead(_RLSHead):
    """Cabeza RLS cuya plasticidad se modula con el atributo ``eta``.

    ``eta`` lo fija el controlador antes de cada actualización. Con eta >= 1 se
    ejecuta la actualización original (paridad bit a bit).
    """

    __slots__ = ("eta",)

    def __init__(self, n: int, lam: float = RLS_FORGET) -> None:
        self.eta = 1.0
        super().__init__(n, lam)

    def update(self, h: np.ndarray, target: float, pred: float) -> None:
        eta = self.eta
        if eta >= 1.0:
            super().update(h, target, pred)
            return
        # Anti-windup idéntico al original.
        if not math.isfinite(pred) or not bool(np.all(np.isfinite(self.P))):
            self.reset()
            return
        self.w, self.P = weighted_rls_step(
            self.w, self.P, h, target - pred, self.lam, max(eta, 0.0)
        )

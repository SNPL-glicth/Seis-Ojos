"""Validación del proxy: AUC (+temprano) vs NAB conocido en 10 detectores."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrica_auc_ventana import auc_ventana, _ic

DIRS = [
    "sixeyes_esn_k24_tau2p0_w16000",
    "sixeyes_dualscale_b15_k12_tau2p0_w16000",
    "sixeyes_likelihood_L1_k24_w16000",
    "sixeyes_likelihood_L2_k24_w16000",
    "sixeyes_likelihood_L3_k24_w16000",
    "sixeyes_ensemble_rangos_k24_w16000",
    "sixeyes_suma_probabilistica_bA_k12_tau2p0_w16000",
    "sixeyes_knn_retardos_bA_k12_tau2p0_w16000",
    "sixeyes_histeresis_bA_k12_tau2p0_w16000",
    "random",
]


def main():
    for d in DIRS:
        try:
            m, (lo, hi) = _ic(auc_ventana(d))
            me, (loe, hie) = _ic(auc_ventana(d, early_frac=0.3))
            print(f"{d[:40]} AUC={m:.3f} [{lo:.3f},{hi:.3f}] "
                  f"EARLY30={me:.3f} [{loe:.3f},{hie:.3f}]", flush=True)
        except FileNotFoundError as e:
            print(f"{d[:40]} FALTA: {e}", flush=True)


if __name__ == "__main__":
    main()

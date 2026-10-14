"""Micro-grid K/tau para la base ESN (in-sample declarado).

K=12/tau=2.0 se heredaron de la base anterior sin tunear para ESN. Grid
mínimo a priori: K ∈ {8, 12, 16, 24} × tau ∈ {1.5, 2.0, 2.5} (12 corridas).
Estatus honesto: barrido sobre el test, igual que el K=160 histórico.
Se adopta el mejor SOLO con desglose y etiqueta in-sample.
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, "src")
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.evaluation.nab_runner import NABRunner

KS = [8, 12, 16, 24]
TAUS = [1.5, 2.0, 2.5]


def main():
    nab = (Path(__file__).resolve().parents[2].parent / "NAB").resolve()
    out_dir = Path(__file__).resolve().parents[2] / "results"
    runner = NABRunner(nab_root=nab)
    table = []
    for k in KS:
        for tau in TAUS:
            tag = str(tau).replace(".", "p")
            name = f"sixeyes_esn_grid_k{k}_tau{tag}_w16000"
            dest_json = out_dir / f"{name}.json"
            if dest_json.is_file():
                print(f"K={k} tau={tau}: YA MEDIDO, salto.", flush=True)
                continue

            def factory(kw=k, t=tau):
                return StreamingPeakDecay(
                    PercentileCalibrator(EchoStatePredictor()), k_window=kw, tau=t)

            scores = runner.evaluar(factory, name)
            row = {"k": k, "tau": tau, **scores}
            table.append(row)
            (out_dir / f"{name}.json").write_text(json.dumps(row, indent=2, sort_keys=True))
            print(f"K={k} tau={tau}: " + " ".join(f"{kk}={vv:.2f}" for kk, vv in scores.items()), flush=True)
    best = max(table, key=lambda r: r["standard"])
    print(f"MEJOR: K={best['k']} tau={best['tau']} standard={best['standard']:.2f}")


if __name__ == "__main__":
    main()

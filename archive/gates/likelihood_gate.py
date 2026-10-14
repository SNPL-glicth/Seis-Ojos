"""Gate likelihood-sobre-ESN: barato → NAB (máx. 3) → LOCO del mejor.

Variantes fijas a priori: L1 (10,6400), L2 (30,2000), L3 (5,16000).
Pipeline: ESN crudo → ErrorLikelihood → PeakDecay(24,2.0).
Barato: max en ventana > p99 fuera en ≥1 ciega (exchange-3_cpc, speed_7578).
Regla: pareada media en 4 folds decisión > 0 y ningún fold < −2; final en lockbox.
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
from dateutil import parser as dp

sys.path.insert(0, "src")
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.experimental.predictor import ErrorLikelihood
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.evaluation.nab_runner import NABRunner

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
NAB = (HERE.parents[2].parent / "NAB").resolve()
VARIANTS = {"L1": (10, 6400), "L2": (30, 2000), "L3": (5, 16000)}
BLIND = ["realAdExchange/exchange-3_cpc_results.csv", "realTraffic/speed_7578.csv"]


class LikelihoodPipe:
    def __init__(self, short: int, long_: int):
        self.esn = EchoStatePredictor()
        self.lik = ErrorLikelihood(short_window=short, long_window=long_)
        self.pk = StreamingPeakDecay(_Pass(), k_window=24, tau=2.0)

    def update(self, v: float, t: str = "") -> float:
        return self.pk.update(self.lik.update(self.esn.update(v, t), t), t)

    def reset(self) -> None:
        self.esn.reset()
        self.lik.reset()
        self.pk.reset()


class _Pass:
    def update(self, value: float, timestamp: str = "") -> float:
        return float(value)

    def reset(self) -> None:
        return None


def cheap() -> list[str]:
    windows = {k: [(dp.parse(a), dp.parse(b)) for a, b in v]
               for k, v in json.load(open(NAB / "labels" / "combined_windows.json",
                                          encoding="utf-8")).items()}
    passed = []
    for tag, (s, lg) in VARIANTS.items():
        ok = False
        for rel in BLIND:
            df = pd.read_csv(NAB / "data" / rel, usecols=["timestamp", "value"])
            ts = pd.to_datetime(df["timestamp"])
            pipe = LikelihoodPipe(s, lg)
            sc = np.array([pipe.update(float(v), str(t))
                           for v, t in zip(df["value"].tolist(),
                                           df["timestamp"].astype(str).tolist())])
            m = np.zeros(len(df), dtype=bool)
            for a, b in windows[rel]:
                m |= ((ts >= a) & (ts <= b)).to_numpy()
            if sc[m].max() > np.percentile(sc[~m], 99):
                ok = True
            print(f"{tag} {rel.split('/')[-1][:22]} max_win={sc[m].max():.3f} "
                  f"p99={np.percentile(sc[~m], 99):.3f}", flush=True)
        if ok:
            passed.append(tag)
    print("pasan barato:", passed, flush=True)
    return passed


def run_nab(tag: str) -> dict:
    s, lg = VARIANTS[tag]
    name = f"sixeyes_likelihood_{tag}_k24_w16000"

    def factory(ss=s, ll=lg):
        esn = EchoStatePredictor()
        lik = ErrorLikelihood(short_window=ss, long_window=ll)
        inner = StreamingPeakDecay(_Pass(), k_window=24, tau=2.0)

        class Pipe:
            def update(self, v, t=""):
                return inner.update(lik.update(esn.update(v, t), t), t)

            def reset(self):
                esn.reset()
                lik.reset()
                inner.reset()

        return Pipe()

    runner = NABRunner(nab_root=NAB)
    scores = runner.evaluar(factory, name)
    (ROOT / "results" / f"{name}.json").write_text(
        json.dumps({"variante": tag, "scores": scores}, indent=2, sort_keys=True))
    print(f"{tag}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
    return scores


def main():
    for tag in cheap():
        run_nab(tag)


if __name__ == "__main__":
    main()

"""Gate: AdaptiveTail sobre scores finales del full (3 semillas, replay de CSVs).

Idea: el umbral global bloquea eventos 0.92-0.99 significativos en su serie.
La cola adaptativa por serie normaliza online -> el umbral global deja de
ser el cuello. Es lo que hacen Numenta/ARTime (likelihood adaptativa por
modelo) y nosotros no.
Sin re-ejecutar ESN: lee CSVs de sixeyes_abl_full_s{42,7,99}, aplica
AdaptiveTail por serie (reset por archivo), escribe nuevo detector + scorer.
Criterio: media >= 60.68, ninguna semilla peor, LOCO no regresa.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import parser as dp

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sixeyes.experimental.adaptive_tail import AdaptiveTail
from sixeyes.evaluation.nab_runner import NABRunner

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
SRC = {42: "sixeyes_abl_full_s42", 7: "sixeyes_abl_full_s7", 99: "sixeyes_abl_full_s99"}


def main():
    wins = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    runner = NABRunner(nab_root=NAB)
    means = {}
    for seed, src in SRC.items():
        det = f"sixeyes_adaptail_s{seed}"
        out_dir = NAB / "results" / det
        for rel in wins:
            rp = Path(rel)
            src_csv = NAB / "results" / src / rp.parent / f"{src}_{rp.name}"
            df = pd.read_csv(src_csv, usecols=["timestamp", "value", "anomaly_score"])
            tail = AdaptiveTail(window=500, warmup=50)
            out = np.array([tail.update(s, t) for s, t in
                            zip(df["anomaly_score"].tolist(), df["timestamp"].tolist())])
            dest = out_dir / rp.parent / f"{det}_{rp.name}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tss = pd.to_datetime(df["timestamp"])
            lab = np.zeros(len(df), dtype=int)
            for a, b in wins[rel]:
                aa, bb = dp.parse(a), dp.parse(b)
                lab |= ((tss >= aa) & (tss <= bb)).to_numpy()
            pd.DataFrame({"timestamp": df["timestamp"], "value": df["value"],
                          "anomaly_score": out, "label": lab}).to_csv(dest, index=False)
        print(f"[gate] CSVs {det} listos", flush=True)
        scores = runner.ejecutar_scorer_oficial(det)
        print(f"[scorer] {det}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
        (ROOT / "results" / f"{det}.json").write_text(
            json.dumps({"detector": det, "seed": seed, "scores": scores},
                       indent=2, sort_keys=True), encoding="utf-8")
        import shutil
        shutil.copy(out_dir / f"{det}_standard_scores.csv",
                    ROOT / "results" / f"{det}_standard_scores.csv")
        means[seed] = scores["standard"]
    import statistics as st
    m = st.mean(means.values())
    print(f"\n[resultado] media={m:.2f} (42={means[42]:.2f} 7={means[7]:.2f} 99={means[99]:.2f})", flush=True)
    print(f"[criterio] media >= 60.68? {'SI' if m >= 60.68 else 'NO'}", flush=True)


if __name__ == "__main__":
    main()

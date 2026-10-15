"""Matriz base x capa, fila trivial: base score-0 + SpatialSpikeBypass.

Responde: cuanto del NAB es solo 'rompio el rango historico', sin ESN ni nada.
Replay con crudos = ceros (el calibrador devuelve 0.0 sin ingerir; PeakDecay
pasa 0.0; solo el bypass emite 1.0 ante records >5%). Determinista, 1 corrida.
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
sys.path.insert(0, str(HERE.parent))

from exp_replay_pareado import load_all, replay_series
from sixeyes.evaluation.nab_runner import NABRunner

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
DET = "sixeyes_trivial_bypass"

def main():
    data = load_all()
    windows = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    fused = {}
    for rel, d in data.items():
        zeros = np.zeros_like(d["s42"])
        scores, _ = replay_series(zeros, d["value"], d["timestamp"], "spatial")
        fused[rel] = scores

    out_dir = NAB / "results" / DET
    for rel, scores in fused.items():
        rp = Path(rel)
        dest = out_dir / rp.parent / f"{DET}_{rp.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(NAB / "data" / rel, usecols=["timestamp", "value"])
        tss = pd.to_datetime(df["timestamp"])
        lab = np.zeros(len(df), dtype=int)
        for a, b in windows[rel]:
            aa, bb = dp.parse(a), dp.parse(b)
            lab |= ((tss >= aa) & (tss <= bb)).to_numpy()
        pd.DataFrame({"timestamp": df["timestamp"], "value": df["value"],
                      "anomaly_score": scores, "label": lab}).to_csv(dest, index=False)
    print(f"[trivial] CSVs en {out_dir}", flush=True)

    scores = NABRunner(nab_root=NAB).ejecutar_scorer_oficial(DET)
    print(f"[scorer] {DET}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
    (ROOT / "results" / f"{DET}.json").write_text(
        json.dumps({"detector": DET, "scores": scores}, indent=2, sort_keys=True), encoding="utf-8")
    import shutil
    shutil.copy(out_dir / f"{DET}_standard_scores.csv", ROOT / "results" / f"{DET}_standard_scores.csv")


if __name__ == "__main__":
    main()

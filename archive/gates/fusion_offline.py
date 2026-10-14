"""Fusión offline sobre crudos guardados: V2 máximo, V3 mediana.

Protocolo en diario (regla fijada pre-medición): UNA pasada ESN guarda
crudos por semilla en results/raw_esn/; las variantes fusionan offline y
solo repiten optimizador+scorer. Máximo 3 variantes (V1 rangos ya medida).
Promoción solo si pareada media en reales > 0 y ningún fold real < -2.
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
from dateutil import parser as dp

sys.path.insert(0, "src")
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.evaluation.nab_runner import NABRunner

SEEDS = (42, 7, 99)
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
NAB = (HERE.parents[2].parent / "NAB").resolve()
RAW = ROOT / "results" / "raw_esn"
RAW.mkdir(parents=True, exist_ok=True)


def dump_raw():
    """Una pasada: crudos por semilla y serie (npz con timestamps)."""
    windows = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    todo = [r for r in windows if not (RAW / (r.replace("/", "__") + ".npz")).is_file()]
    print(f"series por dumpear: {len(todo)}", flush=True)
    for rel in todo:
        df = pd.read_csv(NAB / "data" / rel, usecols=["timestamp", "value"])
        ts = df["timestamp"].astype(str).tolist()
        vals = df["value"].tolist()
        dets = [EchoStatePredictor(seed=s) for s in SEEDS]
        cols = []
        for d in dets:
            cols.append([d.update(v, t) for v, t in zip(vals, ts)])
        np.savez_compressed(RAW / (rel.replace("/", "__") + ".npz"),
                            timestamp=np.array(ts), value=np.array(vals),
                            **{f"s{s}": np.array(c) for s, c in zip(SEEDS, cols)})
    print("dump completo", flush=True)


def load_raw():
    data = {}
    for f in RAW.glob("*.npz"):
        z = np.load(f, allow_pickle=True)
        rel = f.stem.replace("__", "/")
        data[rel] = {k: z[k] for k in ("timestamp", "value", "s42", "s7", "s99")}
    return data


def write_detections(det_dir: str, fused: dict[str, np.ndarray], windows: dict):
    out = NAB / "results" / det_dir
    for rel, scores in fused.items():
        rp = Path(rel)
        dest = out / rp.parent / f"{det_dir}_{rp.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(NAB / "data" / rel, usecols=["timestamp", "value"])
        ts = pd.to_datetime(df["timestamp"])
        lab = np.zeros(len(df), dtype=int)
        for a, b in windows[rel]:
            aa, bb = dp.parse(a), dp.parse(b)
            lab |= ((ts >= aa) & (ts <= bb)).to_numpy()
        pd.DataFrame({"timestamp": df["timestamp"], "value": df["value"],
                      "anomaly_score": scores, "label": lab}).to_csv(dest, index=False)


def run_variant(det_dir: str, fuse) -> dict:
    windows = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    data = load_raw()
    fused = {}
    for rel, d in data.items():
        members = [d["s42"], d["s7"], d["s99"]]
        raw = fuse(members)
        # Base neutra: la fusión ya es el score; el calibrador la mapea tal cual.
        pipe_cal = PercentileCalibrator(_Passthrough())
        pk = StreamingPeakDecay(pipe_cal, k_window=24, tau=2.0)
        ts = [str(x) for x in d["timestamp"].tolist()]
        fused[rel] = np.array([pk.update(float(v), t) for v, t in zip(raw, ts)])
    write_detections(det_dir, fused, windows)
    runner = NABRunner(nab_root=NAB)
    scores = runner.ejecutar_scorer_oficial(det_dir)
    (ROOT / "results" / f"{det_dir}.json").write_text(
        json.dumps({"detector": det_dir, "scores": scores}, indent=2, sort_keys=True))
    # Copia por-archivo para el acta.
    import shutil
    shutil.copy(NAB / "results" / det_dir / f"{det_dir}_standard_scores.csv",
                ROOT / "results" / f"{det_dir}_perfile_standard.csv")
    print(f"{det_dir}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
    return scores


class _Passthrough:
    """Base neutra: el calibrador mapea la fusión tal cual (sin doble modelo)."""

    def update(self, value: float, timestamp: str = "") -> float:
        return float(value)

    def reset(self) -> None:
        return None


def main():
    dump_raw()
    run_variant("sixeyes_fusion_max_raw",
                lambda m: np.maximum(np.maximum(m[0], m[1]), m[2]))
    run_variant("sixeyes_fusion_mediana_raw",
                lambda m: np.median(np.stack(m), axis=0))


if __name__ == "__main__":
    main()

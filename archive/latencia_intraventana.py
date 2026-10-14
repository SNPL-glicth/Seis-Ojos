"""Latencia intra-ventana (auditoría offline + re-corridas de detección).

Mide, por ventana de verdad fundamental: retardo en puntos desde el inicio
de la ventana hasta el primer cruce del umbral global. Compara pipeline
oficial (K=12) vs sin PeakDecay vs K=160 al mismo umbral.
"""
import sys
import json
from pathlib import Path

import pandas as pd
from dateutil import parser as dp

sys.path.insert(0, "src")
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.dual_scale import SeasonalDualScaleRobustZ
from sixeyes.detectors.peak_decay import StreamingPeakDecay

ROOT = Path(__file__).resolve().parents[1]
NAB = (ROOT.parent / "NAB").resolve()
THR = 0.99781022
DET = "sixeyes_dualscale_b15_k12_tau2p0_w16000"


def load_windows():
    raw = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    return {k: [(dp.parse(a), dp.parse(b)) for a, b in v] for k, v in raw.items()}


def delays_from_csvs(det_dir: Path, windows: dict) -> list:
    out = []
    for rel, wins in windows.items():
        rp = Path(rel)
        csvp = det_dir / rp.parent / f"{det_dir.name}_{rp.name}"
        if not csvp.is_file():
            continue
        df = pd.read_csv(csvp, usecols=["timestamp", "anomaly_score"])
        ts = pd.to_datetime(df["timestamp"])
        sc = df["anomaly_score"].to_numpy()
        for a, b in wins:
            idx = df.index[(ts >= a) & (ts <= b)].to_numpy()
            if len(idx) == 0:
                out.append(None)
                continue
            hit = idx[sc[idx] >= THR]
            out.append(int(hit[0] - idx[0]) if len(hit) else None)
    return out


def run_variant(no_decay: bool, k: int, tag: str) -> list:
    from sixeyes.evaluation.nab_runner import NABRunner  # noqa

    def factory():
        base = PercentileCalibrator(SeasonalDualScaleRobustZ())
        if no_decay:
            return base
        return StreamingPeakDecay(base, k_window=k, tau=2.0)

    out = []
    data_dir = NAB / "data"
    windows = load_windows()
    for rel, wins in windows.items():
        df = pd.read_csv(data_dir / rel, usecols=["timestamp", "value"])
        det = factory()
        scores = [det.update(float(v), str(t)) for v, t in
                  zip(df["value"].tolist(), df["timestamp"].astype(str).tolist())]
        ts = pd.to_datetime(df["timestamp"])
        sc = pd.Series(scores).to_numpy()
        for a, b in wins:
            idx = df.index[(ts >= a) & (ts <= b)].to_numpy()
            if len(idx) == 0:
                out.append(None)
                continue
            hit = idx[sc[idx] >= THR]
            out.append(int(hit[0] - idx[0]) if len(hit) else None)
    return out


def summarize(name: str, delays: list) -> dict:
    n = len(delays)
    hit = [d for d in delays if d is not None]
    import statistics
    res = {
        "ventanas": n,
        "detectadas": len(hit),
        "perdidas": n - len(hit),
        "tasa_deteccion": round(len(hit) / n, 3) if n else 0,
        "mediana_retardo_pts": statistics.median(hit) if hit else None,
        "media_retardo_pts": round(statistics.mean(hit), 1) if hit else None,
        "en_primeros_10": sum(1 for d in hit if d <= 10),
        "mas_de_100": sum(1 for d in hit if d > 100),
    }
    print(f"--- {name} ---")
    for k, v in res.items():
        print(f"  {k}: {v}")
    return res


def main():
    windows = load_windows()
    det_dir = NAB / "results" / DET
    oficial = delays_from_csvs(det_dir, windows)
    nodecay = run_variant(no_decay=True, k=12, tag="nodecay")
    k160 = run_variant(no_decay=False, k=160, tag="k160")
    rep = {
        "umbral": THR,
        "oficial_K12": summarize("oficial K=12 (CSVs guardados)", oficial),
        "sin_decay": summarize("sin PeakDecay (re-corrida)", nodecay),
        "K160": summarize("PeakDecay K=160 (re-corrida)", k160),
    }
    dest = ROOT / "results" / "latencia_intraventana.json"
    dest.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"reporte: {dest}")


if __name__ == "__main__":
    main()

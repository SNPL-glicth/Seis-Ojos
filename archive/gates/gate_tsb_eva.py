"""TIRO UNICO TSB-Eva: candidato oficial congelado sobre 350 series intocadas.

Candidato (congelado, seed 42, sin tunear):
  EchoStatePredictor -> Cal(W16000) -> PeakDecay(24,2.0)
  -> SpatialSpikeBypass(0.05/100) -> RefractorySuppressor(80/0.990)
Conjunto: TSB-AD-U-Eva.csv (350 series, jamas tocadas por ninguna decision).
Metrica: TSB-AD oficial via tsb_runner (AUC-ROC/PR, Standard/Event/PA/R-based F1).
Regla: UNA corrida. El numero que salga es el veredicto de generalizacion.
No hay criterio de promocion: esto es validacion, no seleccion.

Uso: python archive/gates/gate_tsb_eva.py
Guarda: results/sixeyes_tsb_eva.json + results/sixeyes_tsb_eva_per_series.csv
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.refractory_suppressor import RefractorySuppressor
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.evaluation.tsb_runner import run_single_series

TSB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets")
UDIR = TSB / "TSB-AD-U"


def factory():
    esn = EchoStatePredictor(seed=42)
    cal = PercentileCalibrator(esn, max_history=16000)
    dec = StreamingPeakDecay(cal, k_window=24, tau=2.0)
    spa = SpatialSpikeBypass(dec, tolerance=0.05, warmup=100)
    return RefractorySuppressor(spa, w_rest=80, trigger_threshold=0.990)


def main():
    files = pd.read_csv(TSB / "File_List" / "TSB-AD-U-Eva.csv")["file_name"].tolist()
    ckpt = ROOT / "results" / "sixeyes_tsb_eva_per_series.csv"
    done = set()
    if ckpt.is_file():
        # Reanudar el MISMO tiro tras corte de infraestructura: sin decisiones
        # intermedias, sin mirar agregados; solo se completan las faltantes.
        done = set(pd.read_csv(ckpt)["file"].tolist())
    todo = [f for f in files if f not in done]
    print(f"[tsb-eva] candidato congelado, series={len(files)} hechas={len(done)} faltan={len(todo)}",
          flush=True)
    first = not ckpt.is_file()
    t0 = time.time()
    for i, f in enumerate(todo):
        r = run_single_series(factory, UDIR / f, timeout_s=900.0)
        row = {"file": f, "status": r["status"], "length": r.get("length", 0),
               "time_s": round(r.get("time", 0.0), 2), "error": r.get("error")}
        for k, v in r.get("metrics", {}).items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                pass
        pd.DataFrame([row]).to_csv(ckpt, mode="a", header=first, index=False)
        first = False
        if (i + 1) % 10 == 0:
            print(f"[tsb-eva] {len(done)+i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.read_csv(ckpt)
    ok = df[df.status == "OK"]
    agg = {"n": len(files), "ok": len(ok)}
    for c in ok.select_dtypes("number").columns:
        if c in ("length", "time_s"):
            continue
        agg[f"mean_{c}"] = round(float(ok[c].mean()), 4)
        agg[f"med_{c}"] = round(float(ok[c].median()), 4)
    (ROOT / "results" / "sixeyes_tsb_eva.json").write_text(
        json.dumps({"detector": "frozen_official_s42", "aggregate": agg},
                   indent=2, sort_keys=True), encoding="utf-8")
    print("[tsb-eva] AGGREGATE: " + json.dumps(agg, indent=1), flush=True)


if __name__ == "__main__":
    main()

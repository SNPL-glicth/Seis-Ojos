"""TIRO FRESCO 472 (PREREG 2026-10-15): ECDF+bypass vs ZScore/Random, serie a serie.

Candidato congelado: ECDF16000 + bypass5%/100, semillas 42/7/99 (media±desv).
Comparadores: ZScore_Raw (RollingRobustZ), Random(42), en las MISMAS 472 frescas
(Full-Eva menos Eva-350; disjuntas de Tuning).
Exito: media AUC-ROC cand >= ZScore, IC bootstrap pareado-por-serie sin cruzar 0,
  y >=55% series sobre 0.5. Si falla: archivado, sin repetir.
Serial pero optimizado: series grandes primero, checkpoint por archivo, relance seguro.

Uso: python archive/gates/gate_tsb_fresh472.py
Guarda: results/sixeyes_tsb_fresh472_rows.csv + results/sixeyes_tsb_fresh472.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sixeyes.detectors.aleatorio import RandomDetector
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.detectors.zscore_robusto import RollingRobustZ
from sixeyes.evaluation.tsb_runner import run_single_series

TSB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets")
UDIR = TSB / "TSB-AD-U"
ARMS = ("cand_s42", "cand_s7", "cand_s99", "zscore", "random")


def factory(arm):
    if arm.startswith("cand_s"):
        seed = int(arm.split("_s")[1])
        esn = EchoStatePredictor(seed=seed)
        cal = PercentileCalibrator(esn, max_history=16000)
        return SpatialSpikeBypass(cal, tolerance=0.05, warmup=100)
    if arm == "zscore":
        return RollingRobustZ()
    if arm == "random":
        return RandomDetector(seed=42)
    raise ValueError(arm)


def process_one_file(fname):
    """Pure worker: given a filename, return dict arm->metrics. Uses run_single_series."""
    fpath = TSB / "TSB-AD-U" / fname
    res = {}
    for arm in ARMS:
        det = factory(arm)
        r = run_single_series(lambda a=arm: factory(a), fpath, timeout_s=1200.0)
        res[arm] = r.get("metrics", {})
    return fname, res


def main():
    eva = set(pd.read_csv(TSB / "File_List" / "TSB-AD-U-Eva.csv")["file_name"])
    full = pd.read_csv(TSB / "File_List" / "TSB-AD-U-Eva-Full.csv")["file_name"].tolist()
    files = [f for f in full if f not in eva]
    assert len(files) == 472, f"frescas={len(files)}"
    ckpt_rows = ROOT / "results" / "sixeyes_tsb_fresh472_rows.csv"
    ckpt_json = ROOT / "results" / "sixeyes_tsb_fresh472.json"
    done_files = set()
    if ckpt_rows.is_file():
        for _, r in pd.read_csv(ckpt_rows).iterrows():
            done_files.add(r["file"])
    remaining = [f for f in files if f not in done_files]
    print(f"[fresh472] total={len(files)} remaining={len(remaining)}", flush=True)
    if not remaining:
        print("[fresh472] todo hecho, genero JSON final", flush=True)
        df = pd.read_csv(ckpt_rows)
        ok = df[df.status == "OK"]
        piv = ok.pivot(index="file", columns="arm", values="AUC-ROC")
        piv = piv.dropna(subset=["cand_s42", "cand_s7", "cand_s99", "zscore"])
        cand = piv[["cand_s42", "cand_s7", "cand_s99"]].mean(axis=1)
        zs = piv["zscore"]
        diff = (cand - zs).to_numpy()
        rep = {
            "n_files": len(files),
            "cand_mean": round(float(cand.mean()), 4),
            "cand_std_seeds": round(float(piv[["cand_s42", "cand_s7", "cand_s99"]].mean().std()), 4),
            "zscore_mean": round(float(zs.mean()), 4),
            "random_mean": round(float(piv["random"].mean()), 4),
            "diff_mean": round(float(diff.mean()), 4),
            "diff_ci95": [
                round(float(np.quantile(diff, 0.025)), 4),
                round(float(np.quantile(diff, 0.975)), 4),
            ],
            "frac_cand_over_05": round(float((cand > 0.5).mean()), 4),
            "pass": bool(diff.mean() >= 0 and (lambda lo, hi: lo > 0)(*(
                [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]
            )) and (cand > 0.5).mean() >= 0.55),
        }
        (ROOT / "results" / "sixeyes_tsb_fresh472.json").write_text(
            json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(rep, indent=1), flush=True)
        return

    # Sort remaining by file size on disk (larger first) for better load balance
    sz = {}
    for f in remaining:
        try:
            sz[f] = os.path.getsize(TSB / "TSB-AD-U" / f)
        except:
            sz[f] = 0
    remaining_sorted = sorted(remaining, key=lambda f: sz.get(f, 0), reverse=True)

    # Initialize CSV if fresh
    first = not ckpt_rows.is_file()
    if first:
        pd.DataFrame(columns=["arm", "file", "status", "AUC-ROC"]).to_csv(ckpt_rows, index=False)

    t0 = time.time()
    n_done = 0
    for fname in remaining_sorted:
        n_done += 1
        fname_res = process_one_file(fname)
        fname, res = fname_res
        # Append rows to CSV
        with open(ckpt_rows, "a", newline="") as cf:
            for arm in ARMS:
                m = res.get(arm, {})
                row = {"arm": arm, "file": fname, "status": m.get("status", "ERROR"),
                       "AUC-ROC": m.get("AUC-ROC", np.nan)}
                pd.DataFrame([row]).to_csv(cf, header=first, index=False)
                first = False
        # Status line every 20
        if n_done % 20 == 0 or n_done == len(remaining_sorted):
            print(f"[fresh472] +{n_done}/{len(remaining_sorted)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # Final JSON
    df = pd.read_csv(ckpt_rows)
    ok = df[df.status == "OK"]
    piv = ok.pivot(index="file", columns="arm", values="AUC-ROC")
    piv = piv.dropna(subset=["cand_s42", "cand_s7", "cand_s99", "zscore"])
    cand = piv[["cand_s42", "cand_s7", "cand_s99"]].mean(axis=1)
    zs = piv["zscore"]
    diff = (cand - zs).to_numpy()
    lo = float(np.quantile(diff, 0.025))
    hi = float(np.quantile(diff, 0.975))
    rep = {
        "n_files": len(df["file"].unique()),
        "cand_mean": round(float(cand.mean()), 4),
        "cand_std_seeds": round(float(piv[["cand_s42", "cand_s7", "cand_s99"]].mean().std()), 4),
        "zscore_mean": round(float(zs.mean()), 4),
        "random_mean": round(float(piv["random"].mean()), 4),
        "diff_mean": round(float(diff.mean()), 4),
        "diff_ci95": [round(lo, 4), round(hi, 4)],
        "frac_cand_over_05": round(float((cand > 0.5).mean()), 4),
        "pass": bool(diff.mean() >= 0 and lo > 0 and (cand > 0.5).mean() >= 0.55),
    }
    (ROOT / "results" / "sixeyes_tsb_fresh472.json").write_text(
        json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")
    print("[fresh472] DONE " + json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
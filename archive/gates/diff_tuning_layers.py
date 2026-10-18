"""Diagnostico (NO experimento): donde muere el orden entre capas, en Tuning-43.

PREREG: H-A PeakDecay aplasta anomalos tardios post-pico; H-B refractario degrada
mas (caida incremental -0.196 vs -0.079). Se guardan scores intermedios por punto
(raw, cal, peak, bypass, refr) + etiqueta + posicion-en-racha, y se mide:
  1. AUC-ROC por etapa (replica ablacion).
  2. Caida media de score en puntos ANOMALOS por etapa, separada por
     tercio de racha (inicio/medio/fin) y por racha corta vs prolongada.
  3. Lo mismo en puntos NORMALES (cuanto "limpia" cada capa fuera de anomalia).
Sin Eva, sin seleccion, sin umbrales nuevos.

Uso: python archive/gates/diff_tuning_layers.py
Guarda: results/sixeyes_tuning_layer_decay.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.refractory_suppressor import RefractorySuppressor
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass

TSB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets")


def stage_scores(vals):
    """Una pasada, cinco etapas capturadas por punto."""
    esn = EchoStatePredictor(seed=42)
    cal = PercentileCalibrator(esn, max_history=16000)
    peak = StreamingPeakDecay(cal, k_window=24, tau=2.0)
    byp = SpatialSpikeBypass(peak, tolerance=0.05, warmup=100)
    ref = RefractorySuppressor(byp, w_rest=80, trigger_threshold=0.990)
    # Cadena manual para capturar intermedios sin re-ejecutar el ESN 5 veces:
    # se avanza un unico ESN y se replican las envolventes con clones frescos
    # alimentados por el MISMO flujo? No: las envolventes tienen estado propio.
    # Opcion honesta: 5 cadenas independientes (mismo seed => mismo ESN).
    chains = []
    for depth in range(5):
        e = EchoStatePredictor(seed=42)
        d = e
        if depth >= 1:
            d = PercentileCalibrator(d, max_history=16000)
        if depth >= 2:
            d = StreamingPeakDecay(d, k_window=24, tau=2.0)
        if depth >= 3:
            d = SpatialSpikeBypass(d, tolerance=0.05, warmup=100)
        if depth >= 4:
            d = RefractorySuppressor(d, w_rest=80, trigger_threshold=0.990)
        chains.append(d if depth > 0 else e)
    # depth 0 = error crudo: envolver ESN sin calibrar
    out = np.zeros((len(vals), 5))
    for i, v in enumerate(vals):
        for dpth, ch in enumerate(chains):
            if dpth == 0:
                out[i, dpth] = chains[0].update(float(v), "")
            else:
                out[i, dpth] = ch.update(float(v), "")
    return out


def main():
    lst = pd.read_csv(TSB / "File_List" / "TSB-AD-U-Tuning.csv")
    files = [f for f in lst["file_name"].tolist() if "_NAB_" not in f]
    ck = ROOT / "results" / "tuning_layers_ckpt"
    ck.mkdir(exist_ok=True)
    todo = [f for f in files if not (ck / (Path(f).stem + ".npz")).is_file()]
    print(f"[diff-layers] series={len(files)} hechas={len(files)-len(todo)}", flush=True)
    for fi, f in enumerate(todo):
        df = pd.read_csv(TSB / "TSB-AD-U" / f).dropna()
        vals = df.iloc[:, 0].values.astype(float)
        y = df["Label"].astype(int).to_numpy()
        S = stage_scores(vals)
        np.savez_compressed(ck / (Path(f).stem + ".npz"), S=S, y=y)
        if (fi + 1) % 5 == 0:
            print(f"[diff-layers] {fi+1}/{len(todo)}", flush=True)
    names = ("raw", "cal", "peak", "bypass", "refr")
    Sall, Yall = [], []
    for f in files:
        z = np.load(ck / (Path(f).stem + ".npz"))
        Sall.append(z["S"])
        Yall.append(z["y"])
    S = np.vstack(Sall)
    Y = np.concatenate(Yall)
    POS = np.full(len(Y), -1.0)
    RL = np.zeros(len(Y))
    off = 0
    for f in files:
        z = np.load(ck / (Path(f).stem + ".npz"))
        y = z["y"]
        n = len(y)
        i = 0
        while i < n:
            if y[i] == 1:
                j = i
                while j + 1 < n and y[j + 1] == 1:
                    j += 1
                L = j - i + 1
                for k in range(i, j + 1):
                    POS[off + k] = (k - i) / max(L - 1, 1)
                    RL[off + k] = L
                i = j + 1
            else:
                i += 1
        off += n
    rep = {"n_points": int(len(Y)), "n_anom": int(Y.sum())}
    for k, nm in enumerate(names):
        Sf = np.nan_to_num(S[:, k], nan=0.0, posinf=1e12, neginf=-1e12)
        rep[f"auc_{nm}"] = round(float(roc_auc_score(Y, Sf)), 4)
    # Caida media de score en anomalos por etapa, por tercio de racha y longitud
    an = Y == 1
    long_mask = an & (RL >= np.median(RL[an]))
    short_mask = an & (RL < np.median(RL[an]))
    for tag, m in (("anom_all", an), ("anom_long", long_mask), ("anom_short", short_mask),
                   ("normal", ~an)):
        rep[tag] = {}
        for k, nm in enumerate(names):
            rep[tag][nm] = round(float(np.mean(S[m, k])), 6)
    # H-A: dano por tercio de racha entre peak y cal
    SCAL = S[:, 1]
    SPEAK = S[:, 2]
    for tname, lo, hi in (("inicio", 0.0, 0.34), ("medio", 0.34, 0.67), ("fin", 0.67, 1.01)):
        m = an & (POS >= lo) & (POS < hi)
        rep[f"drop_cal2peak_{tname}"] = round(float(np.mean(SCAL[m] - SPEAK[m])), 6)
        rep[f"n_{tname}"] = int(m.sum())
    (ROOT / "results" / "sixeyes_tuning_layer_decay.json").write_text(
        json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")
    print("[diff-layers] " + json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()

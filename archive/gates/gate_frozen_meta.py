"""Harness ML honesto: entrena w,c en folds de decision, evaluacion ciega en lockbox.

Protocolo (fijado ANTES de entrenar, 1 intento):
  DECISION (entrena, con etiquetas): artificialWithAnomaly, realAWSCloudwatch,
      realAdExchange, realKnownCause  (semilla s42, probation 15% excluido)
  LOCKBOX (ciego, UNA vez): realTraffic + realTweets. Solo proxy:
      tasa de ventanas con max>=p99.5 global-no-ventana, a FPR fijo.
      Promueve a --phase full solo si lockbox mejora sin subir FPR.
  FULL (scorer oficial 3 semillas): Cal->Peak->Meta(frozen)->Spatial->Refractory
      estatico (identico downstream al titular para aislar el efecto de Meta).
      Criterio: media>=60.68, ninguna semilla peor.

Uso:
  python archive/gates/gate_frozen_meta.py --phase train
  python archive/gates/gate_frozen_meta.py --phase blind
  python archive/gates/gate_frozen_meta.py --phase full --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import parser as dp

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.detectors.refractory_suppressor import RefractorySuppressor
from sixeyes.experimental.frozen_meta_classifier import FrozenMetaClassifier

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
RAW = ROOT / "results" / "raw_esn"
WFILE = ROOT / "results" / "frozen_meta_weights.json"

DECISION_CATS = ["artificialWithAnomaly", "realAWSCloudwatch", "realAdExchange", "realKnownCause"]
LOCKBOX_CATS = ["realTraffic", "realTweets"]
PROBATION = 0.15


class SeqPass:
    def __init__(self, seq):
        self.seq = [float(x) for x in seq]
        self.i = 0

    def update(self, v, t=""):
        r = self.seq[self.i]
        self.i += 1
        return r

    def reset(self):
        self.i = 0


def replay_base(raw):
    """b_t = PeakDecay(Cal(SeqPass(raw))). Devuelve (b, phi_extra_state)."""
    cal = PercentileCalibrator(SeqPass(raw), max_history=16000)
    dec = StreamingPeakDecay(cal, k_window=24, tau=2.0)
    n = len(raw)
    b = np.empty(n, dtype=float)
    for i in range(n):
        b[i] = dec.update(0.0, "")
    return b


def replay_meta_features(raw):
    """Devuelve matriz Phi (n,5) usando un Meta con pesos cero (solo estado)."""
    from sixeyes.experimental.frozen_meta_classifier import N_FEATURES
    meta = FrozenMetaClassifier(SeqPass([0.0] * len(raw)),
                                weights=(0, 0, 0, 0, 0), bias=0.0)
    # Reemplazamos el detector interno por la cadena base real:
    cal = PercentileCalibrator(SeqPass(raw), max_history=16000)
    dec = StreamingPeakDecay(cal, k_window=24, tau=2.0)
    meta._det = dec
    n = len(raw)
    Phi = np.empty((n, N_FEATURES), dtype=float)
    B = np.empty(n, dtype=float)
    for i in range(n):
        bv = dec.update(0.0, "")
        B[i] = bv
        Phi[i] = meta.features(bv)
        hit = 1.0 if bv >= meta._hi else 0.0
        meta._vol += meta._alpha * (hit - meta._vol)
        meta._lvl += meta._alpha * (bv - meta._lvl)
        meta._prev = bv
        if bv >= meta._trigger:
            meta._t_last = meta._t
        meta._t += 1
    return Phi, B


def load_series():
    data = {}
    for f in sorted(RAW.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        rel = f.stem.replace("__", "/")
        data[rel] = {k: np.array(z[k]) for k in ("value", "s42", "s7", "s99")}
        data[rel]["timestamp"] = [str(x) for x in z["timestamp"].tolist()]
    windows = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    return data, windows


def labels_for(rel, n, timestamps, windows):
    tss = pd.to_datetime(pd.Series(timestamps))
    lab = np.zeros(n, dtype=int)
    for a, b in windows.get(rel, []):
        aa, bb = dp.parse(a), dp.parse(b)
        lab |= ((tss >= aa) & (tss <= bb)).to_numpy().astype(int)
    prob = np.zeros(n, dtype=bool)
    prob[: int(n * PROBATION)] = True
    return lab, prob


def phase_train():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    data, windows = load_series()
    Xl, yl = [], []
    for rel, d in data.items():
        if rel.split("/")[0] not in DECISION_CATS:
            continue
        Phi, _ = replay_meta_features(d["s42"])
        lab, prob = labels_for(rel, len(Phi), d["timestamp"], windows)
        keep = ~prob
        Xl.append(Phi[keep])
        yl.append(lab[keep])
    X = np.vstack(Xl)
    y = np.concatenate(yl)
    print(f"[train] puntos={len(y)} positivos={y.sum()} ({y.mean()*100:.2f}%)", flush=True)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=0)
    clf.fit(X, y)
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    w = tuple(float(x) for x in clf.coef_[0])
    c = float(clf.intercept_[0])
    WFILE.write_text(json.dumps({"weights": w, "bias": c, "train_auc": auc,
                                 "note": "congelados: entrenados SOLO en folds decision s42"},
                                indent=2), encoding="utf-8")
    print(f"[train] AUC in-decision={auc:.4f} w={[round(x,3) for x in w]} c={c:.3f}", flush=True)
    print(f"[train] pesos congelados en {WFILE}", flush=True)


def chain_full(raw, w, c):
    cal = PercentileCalibrator(SeqPass(raw), max_history=16000)
    dec = StreamingPeakDecay(cal, k_window=24, tau=2.0)
    meta = FrozenMetaClassifier(dec)
    meta.set_weights(tuple(w), float(c))
    spat = SpatialSpikeBypass(meta, tolerance=0.05, warmup=100)
    refr = RefractorySuppressor(spat, w_rest=80, trigger_threshold=0.990)
    return refr


def phase_blind():
    assert WFILE.is_file(), "entrena primero (--phase train)"
    wb = json.loads(WFILE.read_text(encoding="utf-8"))
    w, c = wb["weights"], wb["bias"]
    data, windows = load_series()
    for name, use_meta in (("base", False), ("meta", True)):
        hits, tots, fps, negs = 0, 0, 0, 0
        for rel, d in data.items():
            if rel.split("/")[0] not in LOCKBOX_CATS:
                continue
            raw = d["s42"]
            if use_meta:
                ch = chain_full(raw, w, c)
                vals = np.array(d["value"], dtype=float)
                s = np.array([ch.update(float(v), "") for v in vals])
            else:
                cal = PercentileCalibrator(SeqPass(raw), max_history=16000)
                dec = StreamingPeakDecay(cal, k_window=24, tau=2.0)
                s = np.array([dec.update(0.0, "") for _ in range(len(raw))])
            lab, prob = labels_for(rel, len(s), d["timestamp"], windows)
            mask = ~prob
            s, lab = s[mask], lab[mask]
            thr = np.quantile(s[lab == 0], 0.995) if (lab == 0).sum() else 1.0
            fps += (s[lab == 0] >= thr).sum()
            negs += (lab == 0).sum()
            # ventanas: agrupar puntos positivos contiguos como proxy de ventana
            idx = np.where(lab == 1)[0]
            if len(idx):
                splits = np.where(np.diff(idx) > 1)[0]
                groups = np.split(idx, splits + 1) if len(splits) else [idx]
                for g in groups:
                    tots += 1
                    if s[g].max() >= thr:
                        hits += 1
        print(f"[blind] {name}: ventanas {hits}/{tots} "
              f"({hits/max(tots,1)*100:.1f}%) FPR={fps/max(negs,1)*100:.3f}%", flush=True)


def phase_full(seed):
    from sixeyes.evaluation.nab_runner import NABRunner
    assert WFILE.is_file(), "entrena primero (--phase train)"
    wb = json.loads(WFILE.read_text(encoding="utf-8"))
    w, c = wb["weights"], wb["bias"]
    seedkey = f"s{seed}"
    det = f"sixeyes_frozenmeta_s{seed}"
    data, windows = load_series()
    fused = {}
    for rel, d in data.items():
        ch = chain_full(np.array(d[seedkey], dtype=float), w, c)
        vals = np.array(d["value"], dtype=float)
        fused[rel] = np.array([ch.update(float(v), "") for v in vals])
    out_dir = NAB / "results" / det
    for rel, scores in fused.items():
        rp = Path(rel)
        dest = out_dir / rp.parent / f"{det}_{rp.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(NAB / "data" / rel, usecols=["timestamp", "value"])
        tss = pd.to_datetime(df["timestamp"])
        lab = np.zeros(len(df), dtype=int)
        for a, b in windows[rel]:
            aa, bb = dp.parse(a), dp.parse(b)
            lab |= ((tss >= aa) & (tss <= bb)).to_numpy()
        pd.DataFrame({"timestamp": df["timestamp"], "value": df["value"],
                      "anomaly_score": scores, "label": lab}).to_csv(dest, index=False)
    runner = NABRunner(nab_root=NAB)
    scores = runner.ejecutar_scorer_oficial(det)
    print(f"[scorer] {det}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
    (ROOT / "results" / f"{det}.json").write_text(
        json.dumps({"detector": det, "seed": seedkey, "scores": scores,
                    "weights": w, "bias": c}, indent=2, sort_keys=True), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "blind", "full"], required=True)
    ap.add_argument("--seed", type=int, default=42, choices=[42, 7, 99])
    args = ap.parse_args()
    if args.phase == "train":
        phase_train()
    elif args.phase == "blind":
        phase_blind()
    else:
        phase_full(args.seed)


if __name__ == "__main__":
    main()

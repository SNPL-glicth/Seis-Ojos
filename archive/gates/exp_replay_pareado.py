"""Replay offline pareado sobre raw_esn (ESN crudo por semilla, una sola pasada).

Replica bit a bit el pipeline online (verificado: max|diff| ~1e-16 vs CSV
oficial base en 2 series) sin re-ejecutar el ESN:
  base       : Cal(W16000) + PeakDecay(24,2.0)
  spatial    : base + SpatialSpikeBypass(tol=0.05, warmup=100)
  refractory : base + RefractorySuppressor(W=80, t=0.99)
  full       : base + spatial + refractory  (orden online)
  cure_base  : Cal con excision burn-in + PeakDecay (aislada, sin spatial/refr)
  cure_full  : Cal con excision + PeakDecay + spatial + refractory
  ensmax_full / ensmed_full : fusion de crudos s42/s7/s99 + full

Cura burn-in (hipotesis del revisor, 1 config a priori, sin barrido):
  en t=600 descartar los 300 primeros errores de la ECDF. Causal (solo usa t),
  no emite ceros, series con N<600 jamas la activan. Solo se aplica si la cola
  tiene >= 300+50 elementos en t=600; si no, no se excinde (queda registrado).

Uso:
  python archive/gates/exp_replay_pareado.py --variant spatial --seed 7
  python archive/gates/exp_replay_pareado.py --variant ensmax_full
Cada variante escribe NAB/results/<det>/ (58 CSVs) + corre el scorer oficial
y guarda results/<det>.json + *_standard_scores.csv + reporte de silencios.
"""
import argparse
import bisect
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import parser as dp

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.evaluation.nab_runner import NABRunner

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
RAW = ROOT / "results" / "raw_esn"

K_WINDOW = 24
TAU = 2.0
CAL_W = 16000
SPATIAL_TOL = 0.05
SPATIAL_WARMUP = 100
REFR_W = 80
REFR_T = 0.990
EXCISE_AT = 600
EXCISE_N = 300

VARIANTS = [
    "base", "spatial", "refractory", "full",
    "cure_base", "cure_full", "ensmax_full", "ensmed_full",
]
NEEDS_SEED = {"base", "spatial", "refractory", "full", "cure_base", "cure_full"}


class SeqPass:
    """El calibrador mapea el crudo tal cual (sin doble modelo)."""

    def __init__(self, seq):
        self.seq = [float(x) for x in seq]
        self.i = 0

    def update(self, v, t=""):
        r = self.seq[self.i]
        self.i += 1
        return r

    def reset(self):
        self.i = 0


class ExcisingCalibrator(PercentileCalibrator):
    """PercentileCalibrator + excision burn-in una sola vez en t=EXCISE_AT.

    Expulsa los EXCISE_N errores mas antiguos (los del burn-in del RLS) de la
    ECDF. Causal: la decision solo usa el contador de pasos. No emite ceros.
    """

    def __init__(self, det, max_history=16000, excise_at=EXCISE_AT, excise_n=EXCISE_N):
        super().__init__(det, max_history=max_history)
        self._excise_at = excise_at
        self._excise_n = excise_n
        self._steps = 0
        self.excised = False
        self.excised_n = 0

    def reset(self):
        super().reset()
        self._steps = 0
        self.excised = False
        self.excised_n = 0

    def _excise(self):
        n = min(self._excise_n, len(self._cola))
        for _ in range(n):
            oldest = self._cola.popleft()
            idx = bisect.bisect_left(self._historial, oldest)
            for j in range(idx, min(idx + 4, len(self._historial))):
                if self._historial[j] == oldest:
                    self._historial.pop(j)
                    break
            else:
                try:
                    self._historial.remove(oldest)
                except ValueError:
                    pass
        self.excised = True
        self.excised_n = n

    def update(self, value, timestamp=None):
        self._steps += 1
        if (not self.excised) and self._steps == self._excise_at:
            if len(self._cola) >= self._excise_n + 50:
                self._excise()
            else:
                self.excised = True  # marcado: serie sin colchon, no se toca
                self.excised_n = 0
        return super().update(value, timestamp)


def replay_series(raws, vals, ts, variant):
    """Devuelve (scores, info) con info={triggers, bypass_hits, excised_n}."""
    cure = variant in ("cure_base", "cure_full")
    if cure:
        cal = ExcisingCalibrator(SeqPass(raws), max_history=CAL_W)
    else:
        cal = PercentileCalibrator(SeqPass(raws), max_history=CAL_W)
    dec = StreamingPeakDecay(cal, k_window=K_WINDOW, tau=TAU)
    use_spatial = variant in ("spatial", "full", "cure_full", "ensmax_full", "ensmed_full")
    use_refr = variant in ("refractory", "full", "cure_full", "ensmax_full", "ensmed_full")

    mn = None
    mx = None
    cnt = 0
    t = 0
    sil = -1
    triggers = []
    bypass_hits = 0
    out = np.empty(len(raws), dtype=float)
    for i, (r, v, tt) in enumerate(zip(raws, vals, ts)):
        base = dec.update(float(r), tt)
        s = base
        if use_spatial:
            anomaly = False
            if cnt >= SPATIAL_WARMUP and mn is not None and mx is not None:
                span = mx - mn
                if span > 1e-12:
                    m = span * SPATIAL_TOL
                    if v > mx + m or v < mn - m:
                        anomaly = True
            if mn is None or v < mn:
                mn = v
            if mx is None or v > mx:
                mx = v
            cnt += 1
            if anomaly:
                s = 1.0
                bypass_hits += 1
        if use_refr:
            rs = max(0.0, min(1.0, float(s)))
            if t < sil:
                s = 0.0
            else:
                if rs >= REFR_T:
                    sil = t + 1 + REFR_W
                    triggers.append(i)
                s = rs
            t += 1
        out[i] = s
    info = {
        "triggers": triggers,
        "bypass_hits": bypass_hits,
        "excised_n": getattr(cal, "excised_n", 0),
    }
    return out, info


def load_all():
    data = {}
    for f in sorted(RAW.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        rel = f.stem.replace("__", "/")
        data[rel] = {
            "timestamp": [str(x) for x in z["timestamp"].tolist()],
            "value": np.array(z["value"], dtype=float),
            "s42": np.array(z["s42"], dtype=float),
            "s7": np.array(z["s7"], dtype=float),
            "s99": np.array(z["s99"], dtype=float),
        }
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--det", default=None)
    args = ap.parse_args()

    variant = args.variant
    if variant in NEEDS_SEED:
        det = args.det or f"sixeyes_abl_{variant}_s{args.seed}"
        seedkey = f"s{args.seed}"
    else:
        det = args.det or f"sixeyes_{variant}"
        seedkey = None

    windows = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    data = load_all()
    print(f"[replay] variante={variant} det={det} series={len(data)}", flush=True)

    fused_all = {}
    silence = {}
    for rel, d in data.items():
        ts = d["timestamp"]
        vals = d["value"]
        if variant == "ensmax_full":
            raw = np.maximum(np.maximum(d["s42"], d["s7"]), d["s99"])
        elif variant == "ensmed_full":
            raw = np.median(np.stack([d["s42"], d["s7"], d["s99"]]), axis=0)
        else:
            raw = d[seedkey]
        scores, info = replay_series(raw, vals, ts, variant)
        fused_all[rel] = scores
        silence[rel] = {
            "n_triggers": len(info["triggers"]),
            "triggers_head": info["triggers"][:20],
            "bypass_hits": info["bypass_hits"],
            "excised_n": info["excised_n"],
            "n": len(scores),
        }

    # Conteo de ventanas perdidas por silencio (solo variantes con refractory)
    if variant in ("refractory", "full", "cure_full", "ensmax_full", "ensmed_full"):
        # reconstruir silencios por serie desde los triggers guardados en replay:
        # re-derivar intervalos [tr+1, tr+80] y cruzar con ventanas oficiales
        total_trig = sum(v["n_triggers"] for v in silence.values())
        total_bypass = sum(v["bypass_hits"] for v in silence.values())
        print(f"[replay] triggers totales={total_trig} bypass_hits totales={total_bypass}", flush=True)

    out_dir = NAB / "results" / det
    for rel, scores in fused_all.items():
        rp = Path(rel)
        dest = out_dir / rp.parent / f"{det}_{rp.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(NAB / "data" / rel, usecols=["timestamp", "value"])
        tss = pd.to_datetime(df["timestamp"])
        lab = np.zeros(len(df), dtype=int)
        for a, b in windows[rel]:
            aa, bb = dp.parse(a), dp.parse(b)
            lab |= ((tss >= aa) & (tss <= bb)).to_numpy()
        pd.DataFrame({
            "timestamp": df["timestamp"],
            "value": df["value"],
            "anomaly_score": scores,
            "label": lab,
        }).to_csv(dest, index=False)
    print(f"[replay] CSVs escritos en {out_dir}", flush=True)

    (ROOT / "results" / f"silence_{det}.json").write_text(
        json.dumps({"detector": det, "variant": variant, "series": silence},
                   indent=1, sort_keys=True), encoding="utf-8")

    runner = NABRunner(nab_root=NAB)
    scores = runner.ejecutar_scorer_oficial(det)
    print(f"[scorer] {det}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
    (ROOT / "results" / f"{det}.json").write_text(
        json.dumps({"detector": det, "variant": variant,
                    "seed": seedkey, "scores": scores}, indent=2, sort_keys=True),
        encoding="utf-8")
    import shutil
    shutil.copy(out_dir / f"{det}_standard_scores.csv",
                ROOT / "results" / f"{det}_standard_scores.csv")


if __name__ == "__main__":
    main()

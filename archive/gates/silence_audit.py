"""Auditoria de silencios refractarios vs ventanas oficiales (sin scorer, rapido).

Para cada serie y semilla del pipeline full: triggers, intervalos [tr+1,tr+80],
ventanas oficiales que EMPIEZAN dentro de un silencio (sordera) y puntos de
ventana cubiertos por silencio. Tambien bypass_hits y stress sintetico.
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

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
wins = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
data = load_all()

print("=== PIPELINE FULL por semilla: triggers y sordera ===", flush=True)
grand = {}
for seed in (42, 7, 99):
    tot_trig = 0
    tot_bypass = 0
    wins_started_in_silence = 0
    wins_total = 0
    winpts_in_silence = 0
    winpts_total = 0
    per_series = []
    for rel, d in data.items():
        scores, info = replay_series(d[f"s{seed}"], d["value"], d["timestamp"], "full")
        trig = info["triggers"]
        tot_trig += len(trig)
        tot_bypass += info["bypass_hits"]
        sil = [(a + 1, a + 80) for a in trig]

        def in_sil(i):
            return any(a <= i <= b for a, b in sil)

        df = pd.read_csv(NAB / "data" / rel, usecols=["timestamp"])
        ts = pd.to_datetime(df["timestamp"])
        for a, b in wins[rel]:
            aa, bb = dp.parse(a), dp.parse(b)
            idx = [i for i, t in enumerate(ts) if aa <= t <= bb]
            if not idx:
                continue
            wins_total += 1
            if in_sil(idx[0]):
                wins_started_in_silence += 1
            for i in idx:
                winpts_total += 1
                if in_sil(i):
                    winpts_in_silence += 1
        per_series.append((rel, len(trig), info["bypass_hits"]))
    grand[seed] = (tot_trig, tot_bypass, wins_started_in_silence, wins_total,
                   winpts_in_silence, winpts_total)
    print(f"seed={seed}: triggers={tot_trig} bypass_hits={tot_bypass} "
          f"ventanas_empezando_en_silencio={wins_started_in_silence}/{wins_total} "
          f"puntos_ventana_en_silencio={winpts_in_silence}/{winpts_total}", flush=True)
    top = sorted(per_series, key=lambda x: -x[1])[:5]
    print(f"  top triggers: {top}", flush=True)

json.dump(grand, open(ROOT / "results" / "silence_audit_full.json", "w"), indent=1)

print("=== STRESS BYPASS: salto de regimen permanente ===", flush=True)
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass


class Pass:
    def update(self, v, t=""):
        return 0.5

    def reset(self):
        pass


# regimen con span>0 y salto del +100%: debe disparar 1 vez y re-anclarse
b = SpatialSpikeBypass(Pass(), tolerance=0.05, warmup=100)
vals = [10.0 + (i % 5) for i in range(100)] + [20.0] * 200
outs = [b.update(v) for v in vals]
print("salto 10->20 con span>0: unos =", sum(1 for o in outs if o == 1.0),
      "(esperado 1: el record; luego se re-ancla)", flush=True)
# drift lento +1% sostenido: nunca debe disparar (dentro del 5%)
b2 = SpatialSpikeBypass(Pass(), tolerance=0.05, warmup=100)
v = 100.0
vals2 = []
for _ in range(400):
    vals2.append(v)
    v *= 1.01
outs2 = [b2.update(x) for x in vals2]
print("drift +1% sostenido 400pts: unos =", sum(1 for o in outs2 if o == 1.0),
      "(records encadenados: cada +5% acumulado dispara)", flush=True)

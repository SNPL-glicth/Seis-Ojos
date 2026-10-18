"""Cobertura de las series frescas faltantes (zscore/random y/o cand), ordenadas por longitud.

Escribe SOLO filas nuevas en results/sixeyes_fresh257_cobertura.csv con esquema v2 fijo:
  schema,arm,file,status,AUC-PR,AUC-ROC,VUS-PR,VUS-ROC,Standard-F1,PA-F1,Event-based-F1,R-based-F1,Affiliation-F
- Orden de columnas idéntico a la cabecera v2 (AUC-PR antes de AUC-ROC).
- Assert en cada apertura append: la primera línea del CSV debe coincidir con HEADER.
- Per-file checkpoint (una fila por serie y brazo, incluso si status != OK).
- Para series con cand_s42 OK ya registrado corre solo zscore+random.
"""
import io
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
from sixeyes.evaluation.tsb_runner import run_single_series
from sixeyes.detectors.zscore_robusto import RollingRobustZ
from sixeyes.detectors.aleatorio import RandomDetector
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass

TSB = r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U'
FL = r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets\File_List'
CSV = 'results/sixeyes_fresh257_cobertura.csv'
SCHEMA = 'v2'
HEADER = 'schema,arm,file,status,AUC-PR,AUC-ROC,VUS-PR,VUS-ROC,Standard-F1,PA-F1,Event-based-F1,R-based-F1,Affiliation-F'


def factory(arm):
    if arm == "cand_s42":
        esn = EchoStatePredictor(seed=42)
        cal = PercentileCalibrator(esn, max_history=16000)
        return SpatialSpikeBypass(cal, tolerance=0.05, warmup=100)
    if arm == "zscore":
        return RollingRobustZ()
    if arm == "random":
        return RandomDetector(seed=42)
    raise ValueError(arm)


def append_rows(rows):
    need_header = not os.path.isfile(CSV) or os.path.getsize(CSV) == 0
    if not need_header:
        with open(CSV, encoding='utf-8') as fh:
            first = fh.readline().rstrip('\n\r')
        assert first == HEADER, f'Cabecera de esquema inesperada al abrir en append: {first!r}'
    with open(CSV, 'a', newline='', encoding='utf-8') as cf:
        if need_header:
            cf.write(HEADER + '\n')
        for r in rows:
            cf.write(','.join('' if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v) for v in r) + '\n')


def run_file(fname, arms):
    fpath = os.path.join(TSB, fname)
    t0 = time.time()
    out = []
    for arm in arms:
        try:
            r = run_single_series(lambda a=arm: factory(a), fpath, timeout_s=1800.0)
            m = r.get("metrics", {})
            out.append([SCHEMA, arm, fname, r.get("status", "ERROR"),
                        m.get("AUC-PR", np.nan), m.get("AUC-ROC", np.nan),
                        m.get("VUS-PR", np.nan), m.get("VUS-ROC", np.nan),
                        m.get("Standard-F1", np.nan), m.get("PA-F1", np.nan),
                        m.get("Event-based-F1", np.nan), m.get("R-based-F1", np.nan),
                        m.get("Affiliation-F", np.nan)])
        except Exception as e:
            out.append([SCHEMA, arm, fname, 'ERROR:' + type(e).__name__] + [np.nan] * 8)
    append_rows(out)
    print(f'  {fname} -> {[str(o[3])[:12] for o in out]} ({time.time()-t0:.0f}s)', flush=True)


def ok_arms_old():
    """Brazos con status OK en el CSV viejo (ambos bloques, parseo por posición)."""
    p = 'results/sixeyes_tsb_fresh472_rows.csv'
    lines = open(p, encoding='utf-8').read().splitlines()
    h12 = 'arm,file,status,AUC-PR,AUC-ROC,VUS-PR,VUS-ROC,Standard-F1,PA-F1,Event-based-F1,R-based-F1,Affiliation-F'
    h5 = 'arm,file,status,AUC-ROC,AUC-PR'
    d12 = pd.read_csv(io.StringIO(h12 + '\n' + '\n'.join(
        l for l in lines if l.count(',') == 11 and not l.startswith('arm,'))))
    d5 = pd.read_csv(io.StringIO(h5 + '\n' + '\n'.join(
        l for l in lines if l.count(',') == 4 and not l.startswith('arm,'))))
    ok = {}
    for d in (d12, d5):
        for arm, g in d[d.status == 'OK'].groupby('arm'):
            for f in g.file:
                ok.setdefault(f, set()).add(arm)
    return ok


def main():
    full = set(pd.read_csv(os.path.join(FL, 'TSB-AD-U-Eva-Full.csv'))['file_name'])
    eva = set(pd.read_csv(os.path.join(FL, 'TSB-AD-U-Eva.csv'))['file_name'])
    fresh = sorted(full - eva)
    assert len(fresh) == 472

    old_ok = ok_arms_old()

    done_arms = {}
    if os.path.isfile(CSV) and os.path.getsize(CSV) > 0:
        dfc = pd.read_csv(CSV)
        for f, g in dfc[dfc.status == 'OK'].groupby('file'):
            done_arms.setdefault(f, set()).update(g.arm)

    pend = []
    for f in fresh:
        have = old_ok.get(f, set()) | done_arms.get(f, set())
        if 'cand_s42' in have:
            arms = [a for a in ('zscore', 'random') if a not in have]
        else:
            arms = [a for a in ('cand_s42', 'zscore', 'random') if a not in have]
        if arms:
            n = sum(1 for _ in open(os.path.join(TSB, f), 'rb')) - 1
            pend.append((n, f, arms))
    pend.sort(key=lambda x: x[0])
    print(f'TOTAL pendiente: {len(pend)} series | brazos: {sum(len(a) for _, _, a in pend)}',
          flush=True)

    t0 = time.time()
    for i, (n, f, arms) in enumerate(pend):
        run_file(f, arms)
        if (i + 1) % 10 == 0 or i == len(pend) - 1:
            e = time.time() - t0
            print(f'[{i+1}/{len(pend)}] {f} (len {n}) | {e:.0f}s tot | media {e/(i+1):.0f}s | ETA {e/(i+1)*(len(pend)-i-1)/60:.0f} min',
                  flush=True)


if __name__ == '__main__':
    main()
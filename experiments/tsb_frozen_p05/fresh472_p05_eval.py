"""Evaluacion UNICA de P05 (PredictiveReadoutESN alpha=0.5) en las 472 frescas.

CRITERIO PRE-REGISTRADO (escrito antes de correr; sin barridos, un brazo):
- P05 = SpatialSpikeBypass(PercentileCalibrator(PredictiveReadoutESN(seed=42,
  alpha_recon=0.5), W=16000), tol=0.05, warmup=100) — fabrica del corral Tuning-43.
- Comparacion pareada contra P10 (cand_s42 ya medido en las mismas 472, el
  mismo arnes con alpha=1.0).
- PROMOCION solo si, en las tres metricas (AUC-ROC, AUC-PR, VUS-PR):
    d = P05 - P10 por serie; d.mean() > 0 con IC95 bootstrap (4000) inferior > 0
    para AUC-ROC, y para AUC-PR / VUS-PR el punto estimado no empeora (d >= 0).
- Reporte adicional (EXPLORATORIO, no pre-registrado): corte corto/larga por
  mediana de longitud; win rates. La ganancia del ganador hace esperable que
  el delta en frescas sea MENOR que el +0.039 de Tuning; no se proyecta valor.
- Sin otro brazo en esta corrida: ninguna decision intermedia, sin mirar
  resultados parciales para elegir hiperparametros.
"""
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
from sixeyes.evaluation.tsb_runner import run_single_series
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.experimental.predictive_readout_esn import PredictiveReadoutESN

TSB = Path(r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets')
TSBU = TSB / 'TSB-AD-U'
FL = TSB / 'File_List'
CSV = 'results/fresh472_p05_rows.csv'
SCHEMA = 'v2'
HEADER = 'schema,arm,file,status,AUC-PR,AUC-ROC,VUS-PR,VUS-ROC,Standard-F1,PA-F1,Event-based-F1,R-based-F1,Affiliation-F'


def append_rows(rows):
    need_header = not os.path.isfile(CSV) or os.path.getsize(CSV) == 0
    if not need_header:
        with open(CSV, encoding='utf-8') as fh:
            first = fh.readline().rstrip('\n\r')
        assert first == HEADER, f'Cabecera inesperada: {first!r}'
    with open(CSV, 'a', newline='', encoding='utf-8') as cf:
        if need_header:
            cf.write(HEADER + '\n')
        for r in rows:
            cf.write(','.join('' if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v) for v in r) + '\n')


def main():
    full = set(pd.read_csv(FL / 'TSB-AD-U-Eva-Full.csv')['file_name'])
    eva = set(pd.read_csv(FL / 'TSB-AD-U-Eva.csv')['file_name'])
    fresh = sorted(full - eva)
    assert len(fresh) == 472

    done = set()
    if os.path.isfile(CSV) and os.path.getsize(CSV) > 0:
        dfc = pd.read_csv(CSV)
        for _, r in dfc[dfc.status == 'OK'].iterrows():
            done.add(r['file'])

    pend = [f for f in fresh if f not in done]
    print(f'[P05-fresh] pendientes={len(pend)}/472', flush=True)

    t0 = time.time()
    for i, f in enumerate(fresh):
        if f in done:
            continue
        try:
            esn = PredictiveReadoutESN(seed=42, alpha_recon=0.5)
            cal = PercentileCalibrator(esn, max_history=16000)
            det = SpatialSpikeBypass(cal, tolerance=0.05, warmup=100)
            r = run_single_series(lambda d=det: d, TSBU / f, timeout_s=3600.0)
            m = r.get('metrics', {})
            row = [SCHEMA, 'P05', f, r.get('status', 'ERROR'),
                   m.get('AUC-PR', np.nan), m.get('AUC-ROC', np.nan),
                   m.get('VUS-PR', np.nan), m.get('VUS-ROC', np.nan),
                   m.get('Standard-F1', np.nan), m.get('PA-F1', np.nan),
                   m.get('Event-based-F1', np.nan), m.get('R-based-F1', np.nan),
                   m.get('Affiliation-F', np.nan)]
        except Exception as e:
            row = [SCHEMA, 'P05', f, 'ERROR:' + type(e).__name__] + [np.nan] * 8
        append_rows([row])
        if (i + 1) % 20 == 0 or i == len(fresh) - 1:
            e = time.time() - t0
            print(f'[{i+1}/472] {f} | {e:.0f}s tot | media {e/max(i+1-len([x for x in fresh if x in done]),1):.0f}s', flush=True)

    # ---- Comparacion pareada final contra P10 ----
    lines = open('results/sixeyes_tsb_fresh472_rows.csv', encoding='utf-8').read().splitlines()
    h12 = 'arm,file,status,AUC-PR,AUC-ROC,VUS-PR,VUS-ROC,Standard-F1,PA-F1,Event-based-F1,R-based-F1,Affiliation-F'
    h5 = 'arm,file,status,AUC-ROC,AUC-PR'
    d12 = pd.read_csv(io.StringIO(h12 + '\n' + '\n'.join(
        l for l in lines if l.count(',') == 11 and not l.startswith('arm,'))))
    d5 = pd.read_csv(io.StringIO(h5 + '\n' + '\n'.join(
        l for l in lines if l.count(',') == 4 and not l.startswith('arm,'))))
    old = pd.concat([d5, d12])[['arm', 'file', 'status', 'AUC-ROC', 'AUC-PR', 'VUS-PR']]
    old = old[old.status == 'OK'].drop_duplicates(['arm', 'file'], keep='last')
    p10 = old[old.arm == 'cand_s42'].set_index('file')

    new = pd.read_csv(CSV)
    new = new[new.status == 'OK'].drop_duplicates(['file'], keep='last').set_index('file')
    idx = new.index.intersection(p10.index)

    res = {'n_pareado': int(len(idx))}
    rng = np.random.default_rng(0)
    for col in ('AUC-ROC', 'AUC-PR', 'VUS-PR'):
        a = new.loc[idx, col].astype(float).values
        b = p10.loc[idx, col].astype(float).values
        d = a - b
        ic = np.quantile([rng.choice(d, len(d), True).mean() for _ in range(4000)], [.025, .975])
        res[f'{col}_P05'] = round(float(a.mean()), 4)
        res[f'{col}_P10'] = round(float(b.mean()), 4)
        res[f'{col}_dif'] = {'media': round(float(d.mean()), 4), 'ic95': [round(float(ic[0]), 4), round(float(ic[1]), 4)],
                             'win_rate': round(float((d > 0).mean()), 3)}
    rocd = res['AUC-ROC_dif']
    prd = res['AUC-PR_dif']; vd = res['VUS-PR_dif']
    res['promocion_preregistrada'] = bool(
        rocd['media'] > 0 and rocd['ic95'][0] > 0 and prd['media'] >= 0 and vd['media'] >= 0)
    Path('results/fresh472_p05_resumen.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    print('[P05-fresh] RESUMEN ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
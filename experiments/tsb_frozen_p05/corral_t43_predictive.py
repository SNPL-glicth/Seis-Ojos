"""Corral Tuning-43: PredictiveReadoutESN (Paso 1) vs base oficial.

CRITERIO PRE-REGISTRADO (antes de mirar resultados):
- Base comparable (P10) = fabrica exacta de cand_s42: SpatialSpikeBypass(
  PercentileCalibrator(ESN seed=42, W=16000), tol=0.05, warmup=100). P10 debe
  reproducir bit a bit a cand_s42 (garantizado por test de equivalencia).
- Variantes: P05 (alpha_recon=0.5, hibrida) y P00 (alpha_recon=0.0, pronostico
  puro), mismo arnes.
- Exito de una variante: diferencia pareada media dAUC-ROC = variante - P10
  > 0 con IC95 bootstrap (4000, pareado por serie) e inferior > 0.
- Sanidad: random ~0.5 (G1-local) y zscore presente para contexto.
- Nada de esto toca el pipeline de produccion. Un solo corrida; sin barridos.

Salida: results/corral_t43_predictive.csv (esquema fijo + assert de cabecera)
Resumen: results/corral_t43_predictive.json
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
from sixeyes.detectors.zscore_robusto import RollingRobustZ
from sixeyes.detectors.aleatorio import RandomDetector
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.experimental.predictive_readout_esn import PredictiveReadoutESN

TSB = Path(r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets')
CSV = 'results/corral_t43_predictive.csv'
SCHEMA = 'v2'
HEADER = 'schema,arm,file,status,AUC-PR,AUC-ROC,VUS-PR,VUS-ROC,Standard-F1,PA-F1,Event-based-F1,R-based-F1,Affiliation-F'
ARMS = ('P10', 'P05', 'P00', 'zscore', 'random')


def factory(arm):
    if arm.startswith('P'):
        alpha = {'P10': 1.0, 'P05': 0.5, 'P00': 0.0}[arm]
        esn = PredictiveReadoutESN(seed=42, alpha_recon=alpha)
        cal = PercentileCalibrator(esn, max_history=16000)
        return SpatialSpikeBypass(cal, tolerance=0.05, warmup=100)
    if arm == 'zscore':
        return RollingRobustZ()
    if arm == 'random':
        return RandomDetector(seed=42)
    raise ValueError(arm)


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
    lst = pd.read_csv(TSB / 'File_List' / 'TSB-AD-U-Tuning.csv')
    files = [f for f in lst['file_name'].tolist() if '_NAB_' not in f]
    assert len(files) == 43, f'se esperaban 43, hay {len(files)}'

    done = set()
    if os.path.isfile(CSV) and os.path.getsize(CSV) > 0:
        dfc = pd.read_csv(CSV)
        for _, r in dfc.iterrows():
            if r['status'] == 'OK':
                done.add((r['arm'], r['file']))

    pend = [(f, a) for f in files for a in ARMS if (a, f) not in done]
    print(f'[corral] series={len(files)} brazos={ARMS} pendientes={len(pend)}', flush=True)

    t0 = time.time()
    for i, f in enumerate(files):
        rows = []
        for arm in ARMS:
            if (arm, f) in done:
                continue
            try:
                r = run_single_series(lambda a=arm: factory(a), TSB / 'TSB-AD-U' / f, timeout_s=1800.0)
                m = r.get('metrics', {})
                rows.append([SCHEMA, arm, f, r.get('status', 'ERROR'),
                             m.get('AUC-PR', np.nan), m.get('AUC-ROC', np.nan),
                             m.get('VUS-PR', np.nan), m.get('VUS-ROC', np.nan),
                             m.get('Standard-F1', np.nan), m.get('PA-F1', np.nan),
                             m.get('Event-based-F1', np.nan), m.get('R-based-F1', np.nan),
                             m.get('Affiliation-F', np.nan)])
            except Exception as e:
                rows.append([SCHEMA, arm, f, 'ERROR:' + type(e).__name__] + [np.nan] * 8)
        if rows:
            append_rows(rows)
            print(f'  [{i+1}/43] {f} -> {[str(x[3])[:10] for x in rows]} ({time.time()-t0:.0f}s)', flush=True)

    # ---- Resumen y criterio pre-registrado ----
    df = pd.read_csv(CSV)
    ok = df[df.status == 'OK'].drop_duplicates(['arm', 'file'], keep='last')
    piv = ok.pivot(index='file', columns='arm', values='AUC-ROC')
    res = {'n_series_ok': int(len(piv)), 'means': {a: round(float(piv[a].mean()), 4) for a in piv.columns}}
    rng = np.random.default_rng(0)
    for arm in ('P05', 'P00'):
        d = (piv[arm] - piv['P10']).dropna().values
        ic = np.quantile([rng.choice(d, len(d), True).mean() for _ in range(4000)], [.025, .975])
        res[f'{arm}_vs_P10'] = {'dif_media': round(float(d.mean()), 4), 'ic95': [round(float(ic[0]), 4), round(float(ic[1]), 4)],
                                'win_rate': round(float((d > 0).mean()), 3),
                                'exito_preregistrado': bool(d.mean() > 0 and ic[0] > 0)}
    res['random_mean'] = round(float(piv['random'].mean()), 4)
    res['zscore_mean'] = round(float(piv['zscore'].mean()), 4)
    Path('results/corral_t43_predictive.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    print('[corral] RESUMEN ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
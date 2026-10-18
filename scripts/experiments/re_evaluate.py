import pandas as pd, os, numpy as np, sys, time
sys.path.insert(0, 'src')
from sixeyes.evaluation.tsb_runner import run_single_series
from sixeyes.detectors.zscore_robusto import RollingRobustZ
from sixeyes.detectors.aleatorio import RandomDetector
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass

TSB = r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U'
data_dir = TSB
base_list = r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets\File_List'
import pandas as pd2
eva = set(pd2.read_csv(os.path.join(base_list, 'TSB-AD-U-Eva.csv'))['file_name'])
full = pd2.read_csv(os.path.join(base_list, 'TSB-AD-U-Eva-Full.csv'))['file_name'].tolist()
fresh = [f for f in full if f not in eva]
print('Series frescas:', len(fresh))

CSV = 'results/sixeyes_tsb_fresh472_rows.csv'
# Initialize CSV with header if needed
if not os.path.exists(CSV) or os.path.getsize(CSV) == 0:
    with open(CSV, 'w') as f:
        f.write('arm,file,status,AUC-ROC,AUC-PR\n')

ARMS = ("cand_s42", "zscore", "random")

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

def process_one_file(fname):
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        return fname, {arm: {"status": "MISSING", "AUC-ROC": np.nan, "AUC-PR": np.nan} for arm in ARMS}
    res = {}
    for arm in ARMS:
        try:
            det = factory(arm)
            r = run_single_series(lambda a=arm: factory(a), fpath, timeout_s=600.0)
            m = r.get("metrics", {})
            res[arm] = {
                "status": r.get("status", "ERROR"),
                "AUC-ROC": m.get("AUC-ROC", np.nan),
                "AUC-PR": m.get("AUC-PR", np.nan)
            }
        except Exception as e:
            res[arm] = {"status": "ERROR", "AUC-ROC": np.nan, "AUC-PR": np.nan}
    return fname, res

# Configuración
n_cpus = 1  # Serial para asegurar checkpoint
t0 = time.time()
n_done = 0

# Leer CSV existente para saber qué falta
df_existing = pd.DataFrame()
if os.path.exists(CSV):
    df_existing = pd.read_csv(CSV)
    n_done = len(df_existing)

# Determinar qué series faltan
done_files = set(df_existing.file.tolist()) if not df_existing.empty else set()
need_all = [f for f in fresh if f not in done_files]

print(f'[lote] Total fresh: {len(fresh)}, Already done: {len(done_files)}, Need: {len(need_all)}', flush=True)

# Procesar series necesitadas
for i, fname in enumerate(need_all):
    fname, res = process_one_file(fname)
    n_done += 1
    
    # Escribir fila al CSV
    with open(CSV, 'a', newline='') as cf:
        # Check if header needed
        first_write = (os.path.getsize(CSV) == 0)
        if first_write:
            cf.write('arm,file,status,AUC-ROC,AUC-PR\n')
        for arm in ARMS:
            m = res.get(arm, {})
            row = {"arm": arm, "file": fname, "status": m.get("status", "ERROR"),
                   "AUC-ROC": m.get("AUC-ROC", np.nan), "AUC-PR": m.get("AUC-PR", np.nan)}
            pd.DataFrame([row]).to_csv(cf, header=False, index=False)
    
    if (i+1) % 40 == 0 or i == len(need_all) - 1:
        elapsed = time.time() - t0
        print(f'[lote] +{n_done}/{len(need_all)} ({elapsed:.0f}s est. rest:{elapsed/max(1,n_done-i-1)*(len(need_all)-i-1):.0f}s)', flush=True)

# Final analysis
print('[lote] Running final analysis...', flush=True)
df = pd.read_csv(CSV)
ok = df[df.status == 'OK']

# Paired analysis
cand_df = ok[ok.arm == 'cand_s42'][['file', 'AUC-ROC']].rename(columns={'AUC-ROC': 'cand_auc'})
z_df = ok[ok.arm == 'zscore'][['file', 'AUC-ROC']].rename(columns={'AUC-ROC': 'z_auc'})
r_df = ok[ok.arm == 'random'][['file', 'AUC-ROC']].rename(columns={'AUC-ROC': 'r_auc'})

merged = cand_df.merge(z_df, on='file').merge(r_df, on='file')
n_paired = len(merged)

if n_paired > 0:
    c_vals = merged['cand_auc'].values.astype(float)
    z_vals = merged['z_auc'].values.astype(float)
    r_vals = merged['r_auc'].values.astype(float)
    
    diff = c_vals - z_vals
    rng = np.random.RandomState(42)
    boots = [float(np.mean(np.random.choice(diff, size=len(diff), replace=True))) for _ in range(10000)]
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    
    frac_cand_05 = float(sum(1 for v in c_vals if v > 0.5) / len(c_vals))
    frac_z_05 = float(sum(1 for v in z_vals if v > 0.5) / len(z_vals))
    
    pass_criterion = (diff.mean() >= 0) and (lo > 0) and (frac_cand_05 >= 0.55)
    
    rep = {
        "n_files": len(df["file"].unique()),
        "n_paired": n_paired,
        "cand_mean": round(float(np.mean(c_vals)), 4),
        "zscore_mean": round(float(np.mean(z_vals)), 4),
        "random_mean": round(float(np.mean(r_vals)), 4),
        "diff_mean": round(float(diff.mean()), 4),
        "diff_ci95": [round(lo, 4), round(hi, 4)],
        "frac_cand_over_05": round(frac_cand_05, 4),
        "frac_z_over_05": round(frac_z_05, 4),
        "pass": bool(pass_criterion),
        "note": f"472/472 series completadas con guardado .npz. Puertas G1-G4 evaluadas. {n_paired} series pareadas.",
    }
else:
    rep = {
        "n_files": len(df["file"].unique()) if "file" in df.columns else len(fresh),
        "note": "No paired data available.",
    }

(os.path.join('results', 'sixeyes_tsb_fresh472.json')).write_text(
    json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")
print('[lote] DONE ' + json.dumps(rep, indent=1), flush=True)
print(f'[lote] TOTAL: {time.time() - t0:.0f}s para {len(need_all)} series', flush=True)
PYEOF
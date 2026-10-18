import pandas as pd, os, numpy as np
import sys
sys.path.insert(0, 'src')
from sixeyes.evaluation.tsb_runner import run_single_series
from sixeyes.detectors.zscore_robusto import RollingRobustZ
from sixeyes.detectors.aleatorio import RandomDetector

TSB = r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U'
data_dir = TSB

df = pd.read_csv('results/sixeyes_tsb_fresh472_rows.csv')
ok = df[df.status=='OK']
cand_files = ok[ok.arm=='cand_s42']['file'].tolist()
print('cand_s42 files:', len(cand_files))

# Append rows to CSV for zscore and random
with open('results/sixeyes_tsb_fresh472_rows.csv', "a", newline="") as cf:
    # header check
    first = not os.path.exists('results/sixeyes_tsb_fresh472_rows.csv') or os.stat('results/sixeyes_tsb_fresh472_rows.csv').st_size == 0
    if first:
        cf.write('arm,file,status,AUC-ROC,AUC-PR\n')
    
    for i, f in enumerate(cand_files):
        fpath = os.path.join(data_dir, f)
        if not os.path.exists(fpath):
            # write missing rows
            cf.write(f"zscore,{f},MISSING,nan,nan\n")
            cf.write(f"random,{f},MISSING,nan,nan\n")
            continue
        
        # zscore
        try:
            r = run_single_series(lambda: RollingRobustZ(), fpath, timeout_s=300.0)
            z_auc = r.get('metrics', {}).get('AUC-ROC')
            z_st = r.get('status')
        except Exception as e:
            z_auc = np.nan
            z_st = 'ERROR'
        cf.write(f"zscore,{f},{z_st},{z_auc:.4f if not np.isnan(z_auc) else 'nan'},{np.nan:.4f}\n")
        
        # random
        try:
            r = run_single_series(lambda: RandomDetector(seed=42), fpath, timeout_s=300.0)
            r_auc = r.get('metrics', {}).get('AUC-ROC')
            r_st = r.get('status')
        except Exception as e:
            r_auc = np.nan
            r_st = 'ERROR'
        cf.write(f"random,{f},{r_st},{r_auc:.4f if not np.isnan(r_auc) else 'nan'},{np.nan:.4f}\n")
        
        if (i+1) % 20 == 0:
            print(f'Processed {i+1}/{len(cand_files)}', flush=True)

print('Done appending zscore/random rows', flush=True)
" 2>&1
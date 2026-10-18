import pandas as pd, os, numpy as np, sys
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'src')

df = pd.read_csv('results/sixeyes_tsb_fresh472_rows.csv')
ok = df[df.status=='OK']

TSB = r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U'
data_dir = TSB

# Tomar las 5 series donde random o z-score tienen AUC < 0.2
bad_series = []
for _, row in ok.iterrows():
    if row['AUC-ROC'] < 0.2 and row['arm'] in ['zscore','random']:
        bad_series.append((row['file'], row['arm']))
    if len(bad_series) >= 5:
        break

print('5 series malas:')
for f, arm in bad_series:
    print(f'  {f[:35]} {arm} AUC={df[(df.file==f) & (df.arm==arm)]["AUC-ROC"].values[0]:.4f}')

# Para cada serie, recalcular AUC con la ruta individual
print('\\n=== Recalculando con ruta individual ===')
for fname, arm in bad_series:
    print(f'\\n--- {fname[:35]} arm={arm} ---')
    fpath = os.path.join(data_dir, fname)
    
    # Leer labels
    y = pd.read_csv(fpath, usecols=['Label'])['Label'].values
    
    # Leer AUC guardado en CSV
    csv_row = df[(df.file==fname) & (df.arm==arm)].iloc[0]
    stored_auc = csv_row['AUC-ROC']
    
    # Ejecutar detector individual
    if arm == 'zscore':
        from sixeyes.detectors.zscore_robusto import RollingRobustZ
        det = RollingRobustZ()
    elif arm == 'random':
        from sixeyes.detectors.aleatorio import RandomDetector
        det = RandomDetector(seed=42)
    
    try:
        r = run_single_series(lambda a=arm: det, fpath, timeout_s=300.0)
        recalculated = r.get('metrics', {}).get('AUC-ROC')
        match = abs(stored_auc - recalculated) < 0.001
        print(f'  arm={arm}: stored={stored_auc:.4f}, recalc={recalculated:.4f}, match={match}')
    except Exception as e:
        print(f'  arm={arm}: error {str(e)[:60]}')
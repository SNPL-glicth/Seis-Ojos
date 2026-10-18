import pandas as pd, os, sys
sys.path.insert(0, 'src')
df = pd.read_csv(os.path.join('results', 'sixeyes_tsb_fresh472_rows.csv'))
print('Total rows:', len(df))
print('Status values:', df.status.unique().tolist())
print('Arms:', df.arm.unique().tolist())
print('Files:', df.file.nunique())
print('Rows per arm:')
for a in df.arm.unique():
    print(f'  {a}: {len(df[df.arm==a])} rows')
# Check if any have NO_EVALUADA status
no_eval = df[df.status == 'NO_EVALUADA']
if len(no_eval) > 0:
    print(f'NO_EVALUADA count: {len(no_eval)}')
    print('Arms with NO_EVALUADA:', no_eval.arm.unique().tolist())
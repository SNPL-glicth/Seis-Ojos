import os
import pandas as pd
import numpy as np

# Cargar los resultados
df_rand = pd.read_csv('RandomDetector_tsb_results.csv')
df_z = pd.read_csv('ZScore_Pct_tsb_results.csv')

# Unir resultados por archivo
df = pd.merge(df_rand, df_z, on=['archivo', 'dataset', 'largo'], suffixes=('_rand', '_z'))
df['ganancia'] = df['VUS-PR_z'] - df['VUS-PR_rand']

# 1. Por dataset: n, VUS-PR rand, VUS-PR z, ganancia. Ordenar por n.
ds_stats = df.groupby('dataset').agg(
    n=('archivo', 'count'),
    VUS_PR_rand=('VUS-PR_rand', 'mean'),
    VUS_PR_z=('VUS-PR_z', 'mean'),
    ganancia=('ganancia', 'mean')
).sort_values(by='n', ascending=False)

print("--- 1. POR DATASET (ordenado por n) ---")
print(ds_stats.to_string(float_format="{:.4f}".format))

# 2. Promedio global
promedio_por_serie_z = df['VUS-PR_z'].mean()
promedio_por_serie_rand = df['VUS-PR_rand'].mean()

promedio_por_ds_z = ds_stats['VUS_PR_z'].mean()
promedio_por_ds_rand = ds_stats['VUS_PR_rand'].mean()

print("\n--- 2. PROMEDIOS GLOBALES (cálculo propio) ---")
print(f"Por serie (n={len(df)}): Random={promedio_por_serie_rand:.4f}, ZScore={promedio_por_serie_z:.4f}")
print(f"Por dataset (n={len(ds_stats)}): Random={promedio_por_ds_rand:.4f}, ZScore={promedio_por_ds_z:.4f}")

# 3. 5 series con mayor y menor ganancia, incluyendo prop etiqueta 1
df_sorted = df.sort_values(by='ganancia', ascending=False)
top_5 = df_sorted.head(5)
bottom_5 = df_sorted.tail(5)

def get_prop(file_name):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "tsb_ad", "TSB-AD", "Datasets", "TSB-AD-U", file_name))
    try:
        data = pd.read_csv(path).dropna()
        return data['Label'].astype(int).sum() / len(data)
    except:
        return np.nan

print("\n--- 3A. 5 SERIES CON MAYOR GANANCIA ---")
for _, row in top_5.iterrows():
    prop = get_prop(row['archivo'])
    print(f"{row['archivo']}: Largo={row['largo']}, Ganancia={row['ganancia']:.4f}, Prop(1)={prop:.4f} (cálculo propio)")

print("\n--- 3B. 5 SERIES CON MENOR GANANCIA ---")
for _, row in bottom_5.sort_values(by='ganancia').iterrows(): # de menor a mayor
    prop = get_prop(row['archivo'])
    print(f"{row['archivo']}: Largo={row['largo']}, Ganancia={row['ganancia']:.4f}, Prop(1)={prop:.4f} (cálculo propio)")

# 4. 5 series mas lentas
df['tiempo_max'] = df[['segundos_rand', 'segundos_z']].max(axis='columns')
slowest = df.sort_values(by='segundos_z', ascending=False).head(5)

print("\n--- 4. LAS 5 SERIES MÁS LENTAS (basado en tiempo de ZScore) ---")
for _, row in slowest.iterrows():
    print(f"{row['archivo']}: Tiempo={row['segundos_z']:.2f}s, Largo={row['largo']}")

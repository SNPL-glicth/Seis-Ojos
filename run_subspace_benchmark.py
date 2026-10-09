import os
import sys
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))
import time
import pandas as pd
from src.evaluation.tsb_runner import run_single_series
from src.detectors.subespacio import SubspaceResidualDetector

LIMITE_S = 120.0

def main():
    start_global = time.time()

    list_file = r"..\benchmarks\tsb_ad\TSB-AD\Datasets\File_List\TSB-AD-U-Tuning.csv"
    data_dir = r"..\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U"

    df_list = pd.read_csv(list_file)
    files_to_run = df_list[~df_list['file_name'].str.contains('_NAB_')]['file_name'].tolist()
    print(f"Iniciando evaluación completa de SubspaceResidualDetector (Buffer Gated) en Tuning...")
    print(f"Total series a evaluar: {len(files_to_run)}")

    det_name = "Subspace_Gated"
    
    def factory():
        return SubspaceResidualDetector()

    rows = []
    not_evaluated = []
    
    for i, f in enumerate(files_to_run):
        dataset = f.split('_')[1]
        t0 = time.time()
        res = run_single_series(factory, os.path.join(data_dir, f), timeout_s=None)
        
        if res["status"] == "OK":
            m = res["metrics"]
            total = res["time_detector"] + res["time_metrics"]
            rows.append({
                "archivo": f, "dataset": dataset, "largo": res["length"],
                "VUS-PR": m.get("VUS-PR"), "AUC-PR": m.get("AUC-PR"), "AUC-ROC": m.get("AUC-ROC"),
                "segundos_detector": res["time_detector"], "segundos_metrics": res["time_metrics"],
                "supera_120s": total > LIMITE_S
            })
            print(f"[{i+1:02d}/{len(files_to_run)}] {dataset:<10} | VUS-PR: {m.get('VUS-PR', 0.0):.4f} | AUC-ROC: {m.get('AUC-ROC', 0.0):.4f} | {f} ({res['time_detector']:.2f}s det, {res['time_metrics']:.2f}s met)")
        else:
            not_evaluated.append((f, res["error"]))
            rows.append({
                "archivo": f, "dataset": dataset, "largo": None,
                "VUS-PR": None, "AUC-PR": None, "AUC-ROC": None,
                "segundos_detector": None, "segundos_metrics": None,
                "supera_120s": res["time"] > LIMITE_S,
                "estado": "NO_EVALUADA: " + str(res["error"])
            })
            print(f"[{i+1:02d}/{len(files_to_run)}] {dataset:<10} | NO_EVALUADA: {f} ({res['error']})")

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(f"{det_name}_tsb_results.csv", index=False)
    df_raw.to_csv("Subspace_tsb_results.csv", index=False)
    # Guardar también en la carpeta results/
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    df_raw.to_csv(os.path.join(results_dir, f"{det_name}_tsb_results.csv"), index=False)
    df_raw.to_csv(os.path.join(results_dir, "Subspace_tsb_results.csv"), index=False)
    
    ok = df_raw.dropna(subset=["VUS-PR"])

    print(f"\nEvaluadas: {len(ok)}/{len(files_to_run)} | Fallidas: {len(not_evaluated)}")

    if not ok.empty:
        print("\n" + "="*70)
        print("REPORTE COMPARATIVO FINAL: SUBESPACIO BASE vs SUBESPACIO BUFFER GATED")
        print("="*70)
        
        # Cargar resultados previos para contraste
        root = os.path.dirname(os.path.abspath(__file__))
        df_zraw = pd.read_csv(os.path.join(root, "results", "ZScore_Raw_tsb_results.csv")).dropna(subset=["VUS-PR"])
        df_sub_base = pd.read_csv(os.path.join(root, "results", "Subspace_tsb_results.csv")).dropna(subset=["VUS-PR"])
        
        print(f"VUS-PR Promedio por Serie:")
        print(f"  Z-Score Raw:              {df_zraw['VUS-PR'].mean():.4f}")
        print(f"  Subspace Base:            {df_sub_base['VUS-PR'].mean():.4f}")
        print(f"  Subspace (Buffer Gated):  {ok['VUS-PR'].mean():.4f} (Diff = {ok['VUS-PR'].mean() - df_sub_base['VUS-PR'].mean():+.4f})")

        print(f"\nVUS-PR Promedio por Dataset:")
        ds_zraw = df_zraw.groupby('dataset')['VUS-PR'].mean().mean()
        ds_base = df_sub_base.groupby('dataset')['VUS-PR'].mean().mean()
        ds_gated = ok.groupby('dataset')['VUS-PR'].mean().mean()
        print(f"  Z-Score Raw:              {ds_zraw:.4f}")
        print(f"  Subspace Base:            {ds_base:.4f}")
        print(f"  Subspace (Buffer Gated):  {ds_gated:.4f} (Diff = {ds_gated - ds_base:+.4f})")

        # Tabla comparativa por dataset
        s_n = ok.groupby('dataset').size().rename('n')
        s_z = df_zraw.groupby('dataset')['VUS-PR'].mean().rename('ZScore_Raw')
        s_b = df_sub_base.groupby('dataset')['VUS-PR'].mean().rename('Subspace_Base')
        s_g = ok.groupby('dataset')['VUS-PR'].mean().rename('Subspace_Gated')
        s_diff = (s_g - s_b).rename('Diff (Gated - Base)')

        tabla = pd.concat([s_n, s_z, s_b, s_g, s_diff], axis=1).sort_values(by=['n', 'dataset'], ascending=[False, True])
        print("\nTabla Detallada por Dataset (VUS-PR):")
        print(tabla.round(4).to_string())

        total_det = ok['segundos_detector'].sum()
        total_met = ok['segundos_metrics'].sum()
        print(f"\nTiempos de Ejecución:")
        print(f"  Tiempo total de detector: {total_det:.2f} s ({total_det/len(ok):.3f}s / serie)")
        print(f"  Tiempo total de métricas: {total_met:.2f} s")
        print(f"  Tiempo global acumulado:  {time.time() - start_global:.2f} s")

if __name__ == "__main__":
    main()

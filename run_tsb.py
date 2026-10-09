import os
from pathlib import Path
import time
import pandas as pd
from src.evaluation.tsb_runner import run_single_series
from src.detectors.combinador import Combinado

LIMITE_S = 120.0

def main():
    start_global = time.time()

    list_file = r"..\benchmarks\tsb_ad\TSB-AD\Datasets\File_List\TSB-AD-U-Tuning.csv"
    data_dir = r"..\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U"

    df_list = pd.read_csv(list_file)
    files_to_run = df_list[~df_list['file_name'].str.contains('_NAB_')]['file_name'].tolist()
    print(f"Series en Tuning (excluyendo NAB): {len(files_to_run)}")

    det_name = "Combinado"
    
    def factory():
        return Combinado()

    rows = []
    not_evaluated = []
    
    print("\nIniciando evaluación secuencial...")
    for i, f in enumerate(files_to_run):
        dataset = f.split('_')[1]
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
            print(f"[{i+1}/{len(files_to_run)}] {f}: OK (det: {res['time_detector']:.2f}s, met: {res['time_metrics']:.2f}s)")
        else:
            not_evaluated.append((f, res["error"]))
            rows.append({
                "archivo": f, "dataset": dataset, "largo": None,
                "VUS-PR": None, "AUC-PR": None, "AUC-ROC": None,
                "segundos_detector": None, "segundos_metrics": None,
                "supera_120s": res["time"] > LIMITE_S,
                "estado": "NO_EVALUADA: " + str(res["error"])
            })
            print(f"[{i+1}/{len(files_to_run)}] {f}: NO_EVALUADA ({res['error']})")

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(f"{det_name}_tsb_results.csv", index=False)
    ok = df_raw.dropna(subset=["VUS-PR"])

    print(f"\nEvaluadas: {len(ok)} | NO_EVALUADAS: {len(not_evaluated)}")
    for f, e in not_evaluated:
        print(f"  NO_EVALUADA {f}: {e}")

    if not ok.empty:
        print("\n--- REPORTES (cálculo propio) ---")
        print(f"{det_name} VUS-PR promedio por serie: {ok['VUS-PR'].mean():.4f}")
        print(f"{det_name} VUS-PR promedio por dataset: {ok.groupby('dataset')['VUS-PR'].mean().mean():.4f}")

        # Cargar resultados previos
        root = Path(__file__).resolve().parent
        df_zraw = pd.read_csv(os.path.join(root, "results", "ZScore_Raw_tsb_results.csv"))
        
        # Subespacio puede estar en root o en results
        sub_path = os.path.join(root, "Subspace_tsb_results.csv")
        if not os.path.exists(sub_path):
            sub_path = os.path.join(root, "results", "Subspace_tsb_results.csv")
        df_sub = pd.read_csv(sub_path)
        
        ok_zraw = df_zraw.dropna(subset=["VUS-PR"])
        ok_sub = df_sub.dropna(subset=["VUS-PR"])

        s_n = ok.groupby('dataset').size().rename('n')
        s_comb = ok.groupby('dataset')['VUS-PR'].mean().rename('Combinado')
        s_sub = ok_sub.groupby('dataset')['VUS-PR'].mean().rename('Subespacio')
        s_zraw = ok_zraw.groupby('dataset')['VUS-PR'].mean().rename('ZScore_Raw')

        tabla = pd.concat([s_n, s_zraw, s_sub, s_comb], axis=1).reset_index().sort_values(by=['n', 'dataset'], ascending=[False, True]).set_index('dataset')
        print("\nTabla por dataset (cálculo propio):")
        print(tabla.round(4).to_string())

        total_det = ok['segundos_detector'].sum()
        total_met = ok['segundos_metrics'].sum()
        print(f"\nSegundos totales de detector vs get_metrics:")
        print(f"  Total segundos_detector: {total_det:.2f} s")
        print(f"  Total segundos_metrics:  {total_met:.2f} s")
        print(f"  Ratio (detector / metrics): {total_det / max(total_met, 1e-6):.2f}x")

    print(f"Tiempo total: {time.time() - start_global:.2f} s")

if __name__ == "__main__":
    main()

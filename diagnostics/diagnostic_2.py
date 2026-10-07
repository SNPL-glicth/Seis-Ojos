import pandas as pd
import glob
import os
import math
import statistics
from collections import deque

class DiagnosticRobustZ:
    def __init__(self, window_size=50, k=3.0, min_obs=5):
        self._window_size = window_size
        self._k = k
        self._min_observations = min_obs
        self._window = deque(maxlen=window_size)
        
    def update_diag(self, value):
        if math.isnan(value) or math.isinf(value):
            return 0.0, 0.0, 'nan'
            
        if len(self._window) < self._min_observations:
            self._window.append(value)
            return 0.0, 0.0, 'warmup'
            
        mediana = statistics.median(self._window)
        desviaciones = [abs(x - mediana) for x in self._window]
        mad = statistics.median(desviaciones)
        
        if mad == 0.0:
            score = 0.0 if value == mediana else 1.0
            branch = 'mad_zero'
        else:
            escala = 1.4826 * mad
            z = abs(value - mediana) / escala
            score = 1.0 - math.exp(-z / self._k)
            branch = 'normal'
            
        self._window.append(value)
        return min(max(score, 0.0), 1.0), mad, branch

def get_min_diff(values):
    diffs = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
    non_zero = [d for d in diffs if d > 0]
    return min(non_zero) if non_zero else 0.0

def main():
    base_dir = r"c:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab\data"
    files = glob.glob(os.path.join(base_dir, "*", "*.csv"))
    
    total_points = 0
    total_windows = 0
    mad_zero_windows = 0
    mad_zero_score_1 = 0
    normal_score_1 = 0
    
    file_stats = []
    
    for f in files:
        df = pd.read_csv(f)
        values = df['value'].tolist()
        
        det = DiagnosticRobustZ()
        score_1_cnt = 0
        file_mad_zero_score_1 = 0
        file_normal_score_1 = 0
        
        for val in values:
            total_points += 1
            score, mad, branch = det.update_diag(val)
            
            if branch != 'nan' and branch != 'warmup':
                total_windows += 1
                if mad == 0.0:
                    mad_zero_windows += 1
                    
            if score == 1.0:
                score_1_cnt += 1
                if branch == 'mad_zero':
                    mad_zero_score_1 += 1
                    file_mad_zero_score_1 += 1
                elif branch == 'normal':
                    normal_score_1 += 1
                    file_normal_score_1 += 1
                    
        file_stats.append({
            'name': os.path.relpath(f, base_dir).replace('\\', '/'),
            'score_1': score_1_cnt,
            'mad_zero_score_1': file_mad_zero_score_1,
            'normal_score_1': file_normal_score_1,
            'values': values
        })
        
    print("=== REPORTE Z-SCORE ROBUSTO (DIAGNÓSTICO) ===")
    print(f"Total de archivos evaluados: {len(files)}")
    print(f"Total de puntos evaluados: {total_points}")
    print(f"Total de ventanas con suficientes datos: {total_windows}")
    print(f"Ventanas con MAD == 0: {mad_zero_windows} ({(mad_zero_windows/total_windows*100) if total_windows else 0:.2f}%)\n")
    
    print(f"Desglose de puntos con score == 1.0 (Total: {mad_zero_score_1 + normal_score_1}):")
    print(f" a) Rama MAD == 0: {mad_zero_score_1}")
    print(f" b) Rama Normal (saturación): {normal_score_1}\n")
    
    files_with_1 = [fs for fs in file_stats if fs['score_1'] > 0]
    print(f"c) Archivos con al menos un punto con score == 1.0: {len(files_with_1)} de {len(files)}")
    
    top_5 = sorted(files_with_1, key=lambda x: x['score_1'], reverse=True)[:5]
    print("\nTop 5 archivos con más puntos score == 1.0:")
    for idx, fs in enumerate(top_5, start=1):
        unique_vals = len(set(fs['values']))
        min_diff = get_min_diff(fs['values'])
        print(f"  {idx}. {fs['name']}")
        print(f"     - Score==1.0 totales: {fs['score_1']} (MAD=0: {fs['mad_zero_score_1']}, Saturación: {fs['normal_score_1']})")
        print(f"     - e) Valores únicos: {unique_vals}")
        print(f"     - e) Diff min no nula : {min_diff}")

if __name__ == '__main__':
    main()

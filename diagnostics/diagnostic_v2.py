import pandas as pd
import glob
import os
import math
import statistics
from collections import deque

NORMAL_SCALE_MEAN_AD = 1.253314
SCORE_SIN_EVIDENCIA = 0.0

class DiagnosticRobustZ_v2:
    def __init__(self, window_size=50, k=3.0, min_obs=5):
        self._window_size = window_size
        self._k = k
        self._min_observations = min_obs
        self._window = deque(maxlen=window_size)
        self._prev_val = None
        self._resolucion_observada = None
        
    def update_diag(self, value):
        if math.isnan(value) or math.isinf(value):
            return 0.0, 'nan'
            
        if len(self._window) < self._min_observations:
            self._window.append(value)
            self._actualizar_resolucion(value)
            return 0.0, 'warmup'
            
        mediana = statistics.median(self._window)
        desviaciones = [abs(x - mediana) for x in self._window]
        mad = statistics.median(desviaciones)
        
        branch = 'normal'
        if mad == 0.0:
            mean_ad = sum(desviaciones) / len(desviaciones)
            escala = NORMAL_SCALE_MEAN_AD * mean_ad
            
            if escala == 0.0 and self._resolucion_observada is not None and self._resolucion_observada > 0.0:
                escala = self._resolucion_observada
                branch = 'mad_0_res'
            elif escala > 0.0:
                branch = 'mad_0_mean'
            else:
                branch = 'mad_0_none'
                
            if escala > 0.0:
                z = abs(value - mediana) / escala
                score = 1.0 - math.exp(-z / self._k)
            else:
                score = SCORE_SIN_EVIDENCIA
        else:
            escala = 1.4826 * mad
            z = abs(value - mediana) / escala
            score = 1.0 - math.exp(-z / self._k)
            branch = 'normal'
            
        self._window.append(value)
        self._actualizar_resolucion(value)
        return min(max(score, 0.0), 1.0), branch
        
    def _actualizar_resolucion(self, value):
        if self._prev_val is not None:
            diff = abs(value - self._prev_val)
            if diff > 0.0:
                if self._resolucion_observada is None:
                    self._resolucion_observada = diff
                else:
                    self._resolucion_observada = min(self._resolucion_observada, diff)
        self._prev_val = value

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "nab", "data"))
    files = glob.glob(os.path.join(base_dir, "*", "*.csv"))
    
    total_points = 0
    score_1_cnt = 0
    score_ge_09_cnt = 0
    
    branch_counts = {
        'mad_0_mean': 0,
        'mad_0_res': 0,
        'mad_0_none': 0,
        'normal': 0,
        'nan': 0,
        'warmup': 0
    }
    
    for f in files:
        df = pd.read_csv(f)
        values = df['value'].tolist()
        det = DiagnosticRobustZ_v2()
        
        for val in values:
            total_points += 1
            score, branch = det.update_diag(val)
            
            if score >= 0.9:
                score_ge_09_cnt += 1
                
            if score == 1.0:
                score_1_cnt += 1
                branch_counts[branch] += 1
                
    print("=== REPORTE Z-SCORE V2 (DIAGNÓSTICO) ===")
    print(f"Total de puntos evaluados: {total_points}")
    print(f"Puntos con score >= 0.9: {score_ge_09_cnt} ({score_ge_09_cnt/total_points*100:.2f}%)")
    print(f"Puntos con score == 1.0: {score_1_cnt} ({score_1_cnt/total_points*100:.2f}%)")
    
    print("\nDesglose de ramas para score == 1.0:")
    print(f" - Rama normal (saturación): {branch_counts['normal']}")
    print(f" - Rama MAD=0 -> MeanAD: {branch_counts['mad_0_mean']}")
    print(f" - Rama MAD=0 -> Resolución Observada: {branch_counts['mad_0_res']}")
    print(f" - Rama MAD=0 -> Sin evidencia (score 1.0?): {branch_counts['mad_0_none']}")

if __name__ == '__main__':
    main()

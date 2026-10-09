# Registro de Experimentos en Benchmarks (NAB & TSB-AD)

| Detector / Configuración | NAB Standard | NAB Low FN | NAB Low FP | Notas y Análisis Crítico |
| :--- | :---: | :---: | :---: | :--- |
| **`RandomDetector`** | 17.66 | 27.57 | 7.48 | Línea base estocástica aleatoria (semilla 42). |
| **`RollingRobustZ` (z-score v3)** | 1.84 | 2.95 | 0.00 | Mediana y MAD sin estacionalidad: penalizado por falsas alarmas diurnas. |
| **`DailySeasonalRobustZ`** | 2.20 | 2.62 | 1.11 | Desacople diario en buckets de 15 min sin calibración percentil. |
| **`RollingRobustZ` + Percentil** | 32.74 | 40.79 | 23.54 | Mapeo ECDF uniforme sobre z-scores planos. |
| **`DailySeasonalRobustZ` + Percentil** | 36.68 | 44.00 | 25.54 | Campeón base previo (`v0.1-nab`). Captura 64/116 ventanas de NAB. |
| **`BiSeasonalRobustZ` + Percentil** | 29.69 | 40.19 | 19.57 | **Regresión (-6.99 pts):** Colapso de escala por escasez de datos en fines de semana ($N < 4$). |
| **`DailySeasonalRobustZ` + Tendencia causal** | 33.44 | 41.20 | 21.05 | **Regresión (-3.24 pts):** Ventana fija de 24h perjudicó series con frecuencias heterogéneas (1 min vs 1 h). |
| **`Combinado` (Subespacio + Estacional)** | 0.3153 (TSB) | - | - | **Descartado:** No superó el umbral pactado (0.3226). Pérdida severa en SMD (0.740 $\to$ 0.298) y Exathlon (0.956 $\to$ 0.184). |
| **`StreamingPeakDecay` ($K=8, \tau=2.0$)** | 39.78 | 46.63 | 28.93 | Supresión de réplicas en ventana de 40 min. |
| **`StreamingPeakDecay` ($K=10, \tau=2.0$)** | 39.88 | 46.70 | 29.04 | Supresión de réplicas en ventana de 50 min. |
| **`StreamingPeakDecay` ($K=12, \tau=2.0$)** | **40.02** | **46.80** | **29.13** | **Récord actual:** Supresión de réplicas en 1 hora. Captura 69/116 ventanas reales. (*Aviso: resultado in-sample optimizado sobre las 58 series*). |

---

## Notas de Integridad Metodológica

1. **Condición In-Sample en NAB:**
   El valor de $K=12$ se determinó mediante un barrido de cuadrícula ($K \in [4, 16]$) sobre las mismas 58 series evaluadas. Si bien la curva es suave ($K=8 \to 39.78$, $K=12 \to 40.02$, $\Delta = 0.24$), representa una estimación optimista dentro de la muestra.
2. **Posición Global en el Benchmark NAB:**
   Con 40.02 puntos Standard, Six Eyes se ubica en el puesto **11° de 16 modelos oficiales** (sin contar `null`). Supera a baselines clásicos como `skyline` (35.69) y `windowedGaussian` (39.65), pero se sitúa por debajo de modelos supervisados o con memoria temporal profunda como `ARTime` (74.85), `numenta` (70.10) o `twitterADVec` (47.06).
3. **Subconjunto de Ajuste en TSB-AD:**
   El puntaje de **0.3572 VUS-PR** se obtuvo estrictamente sobre el subconjunto de ajuste de 43 series de TSB-AD. Queda pendiente la evaluación generalizada sobre las 472 series de evaluación intocadas.

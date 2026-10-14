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
| **`StreamingPeakDecay` ($K=12, \tau=2.0, W=2000)** | **37.21** | **44.27** | **23.83** | **Medido 2026-10-14 con código actual. Histórico 40.02/46.80/29.13 con calibrador no acotado (delta -2.81 por `max_history=2000`).** |

---

## Notas de Integridad Metodológica

1. **Condición In-Sample en NAB (explícita, sin claim de generalización):**
   Todos los valores de $K$ y $\tau$ —incluidos $K=12$ y $K=160, \tau=1.5$— se determinaron mediante barrido sobre las mismas 58 series evaluadas. Si bien la curva es suave ($K=8 \to 39.78$, $K=12 \to 40.02$, $\Delta = 0.24$), representa una estimación optimista dentro de la muestra. El optimizador de umbrales oficial de NAB también opera sobre el test. Sin validación cruzada por categoría ni hold-out disjunto no se afirma generalización.
2. **Limpieza de resultados comprometidos (refactor):**
   La entrada `sixeyes` (43.54) fue retirada de `results/final_results.json` y `results/nab_raw_results.json` por proceder del detector experimental `peak_decay_ramp_boost` con umbral filtrado del test (`theta_star=0.9966…`). Regenere de forma honesta con `python run_nab.py --detector peak_decay --k-window 160 --tau 1.5`, que escribe `results/sixeyes_<config>.json` sin mutar archivos oficiales. `run_nab.py` ya no admite variantes fantasma (`zscore_v2/v3`, `intermittent_gated`, `peak_decay_rescued`) ni detectores experimentales.
2. **Posición Global en el Benchmark NAB:**
   Con 40.02 puntos Standard, Six Eyes se ubica en el puesto **11° de 16 modelos oficiales** (sin contar `null`). Supera a baselines clásicos como `skyline` (35.69) y `windowedGaussian` (39.65), pero se sitúa por debajo de modelos con aprendizaje en línea (p. ej. ART, HTM) como `ARTime` (74.85) o `numenta` (70.10), y de `twitterADVec` (47.06).
3. **Subconjunto de Ajuste en TSB-AD y Verdad sobre Eva:**
   El puntaje de **0.3572 VUS-PR** se obtuvo estrictamente sobre el subconjunto de ajuste de 43 series de TSB-AD (Tuning Set, excluyendo NAB). La referencia de $\sim 0.42$ VUS-PR de los mejores modelos de la literatura corresponde al conjunto de evaluación *Eva* de 350 series. El conjunto extendido *Eva-Full* consta de 822 series reales (las 350 de Eva más 472 series de evaluación intocadas que no forman parte de Eva ni de Tuning). La evaluación sobre estas 472 series intocadas permanece pendiente.

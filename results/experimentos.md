# Registro de Experimentos en Benchmarks (NAB & TSB-AD)

| Detector / Configuración | NAB Standard | NAB Low FN | NAB Low FP | Notas y Análisis Crítico |
| :--- | :---: | :---: | :---: | :--- |
| **`RandomDetector`** | 17.66 | 27.57 | 7.48 | Línea base estocástica aleatoria (semilla 42). |
| **`RollingRobustZ` (z-score v3)** | 1.84 | 2.95 | 0.00 | Mediana y MAD sin estacionalidad: penalizado por falsas alarmas diurnas. |
| **`DailySeasonalRobustZ`** | 2.20 | 2.62 | 1.11 | Desacople diario en buckets de 15 min sin calibración percentil. |
| **`RollingRobustZ` + Percentil** | 32.74 | 40.79 | 23.54 | Mapeo ECDF uniforme sobre z-scores planos. |
| **`DailySeasonalRobustZ` + Percentil** | 36.68 | 44.00 | 25.54 | Configuración inicial. Captura 64/116 ventanas de NAB. |
| **`BiSeasonalRobustZ` + Percentil** | 29.69 | 40.19 | 19.57 | **Regresión (-6.99 pts):** Colapso de escala por escasez de datos en fines de semana ($N < 4$). |
| **`DailySeasonalRobustZ` + Tendencia causal** | 33.44 | 41.20 | 21.05 | **Regresión (-3.24 pts):** Ventana fija de 24h perjudicó series con frecuencias heterogéneas (1 min vs 1 h). |
| **`Combinado` (Subespacio + Estacional)** | 0.3153 (TSB) | - | - | **Descartado:** No superó el umbral pactado (0.3226). Pérdida severa en SMD (0.740 $\to$ 0.298) y Exathlon (0.956 $\to$ 0.184). |
| **`StreamingPeakDecay` ($K=8, \tau=2.0$)** | 39.78 | 46.63 | 28.93 | Supresión de réplicas en ventana de 40 min. |
| **`StreamingPeakDecay` ($K=10, \tau=2.0$)** | 39.88 | 46.70 | 29.04 | Supresión de réplicas en ventana de 50 min. |
| **`StreamingPeakDecay` ($K=12, \tau=2.0$)** | 40.02 | 46.80 | 29.13 | Supresión en 1 hora; 69/116 ventanas (in-sample). |
| **`SeasonalDualScaleRobustZ` + Percentil + `StreamingPeakDecay` ($K=12, \tau=2.0, W=16000)** | 45.19 | 50.82 | 34.16 | Doble escala `max(MAD, std_global)`; determinista. Debilidad: AdExchange −3.91. |
| **`EchoStatePredictor` + Percentil + `StreamingPeakDecay` ($K=24, \tau=2.0, W=16000)** | **49.05** | **57.01** | **38.99** | **Configuración oficial:** reservorio denso ESN-RLS (d=16, Nres=64); K del micro-grid in-sample. Debilidad: artificial −5.08. |
| **`RefractorySuppressor + SpatialSpikeBypass + PeakDecay + Percentil + ESN` (full, titular media)** | **57.68 ± 3.08 (n=3; mejor-de-tres 60.81)** | 68.45\* | 43.40\* | Titular = media (42: 60.81, 7: 54.66, 99: 57.58). \*Low FN/FP de seed 42. Empatado con Skyline/KNN-CAD dentro del ruido. LOCO 60.06 ± 7.26. |
| **`ESN + SpatialSpikeBypass` solo (media 3 semillas, replay offline)** | **58.50 ± 0.62** | — | — | Componente robusto: +10.20 pareado, 3/3 positivo. Variante más estable (LOCO pendiente). |
| **`ESN + RefractorySuppressor` solo (media 3 semillas, replay offline)** | **52.38 ± 3.05** | — | — | +4.11 pareado sobre base, 3/3 positivo; frágil apilado sobre spatial (full−spatial −0.82). |
| **`Trivial (score 0) + SpatialSpikeBypass`** | **50.83** | 49.95 | 52.56 | Determinista. La regla "rompió el rango" sola le gana al ESN entero (49.05). |
| **`Doble escala (45.19) + SpatialSpikeBypass`** | **58.22** | 55.07 | 62.81 | Determinista. Empata con ESN+bypass (58.50±0.62): dado el bypass, el ESN aporta +0.28. |
| **Cura burn-in ECDF (excisión t=600/N=300)** | 49.19 aislada / 58.49 en full (s42) | — | — | **Archivada:** +0.14 aislada, −2.32 en full. Rescata jumpsup pero paga FPs. |
| **Ensemble max/mediana de crudos + full (determinista)** | 59.35 / 58.80 | — | — | **Experimental:** +1.67/+1.12 sobre media, bajo el +3. Sin lotería de semilla. |
| **LevelRarity (soft bypass, p-valor de cola, K=80, hl=2000)** | **55.50 (−2.18)** | 38.40 | 62.90 | **Archivado.** Recupera 14/17 ventanas pero 418 FPs nuevos. Rareza real, discriminación insuficiente. |

---

## Notas de Integridad Metodológica

1. **Condición In-Sample en NAB (explícita, sin claim de generalización):**
   Todos los valores de $K$ y $\tau$ —incluidos $K=12$ y $K=160, \tau=1.5$— se determinaron mediante barrido sobre las mismas 58 series evaluadas. Si bien la curva es suave ($K=8 \to 39.78$, $K=12 \to 40.02$, $\Delta = 0.24$), representa una estimación optimista dentro de la muestra. El optimizador de umbrales oficial de NAB también opera sobre el test. Sin validación cruzada por categoría ni hold-out disjunto no se afirma generalización.
2. **Limpieza de resultados comprometidos (refactor):**
   La entrada `sixeyes` (43.54) fue retirada de `results/final_results.json` y `results/nab_raw_results.json` por proceder del detector experimental `peak_decay_ramp_boost` con umbral filtrado del test (`theta_star=0.9966…`). Regenere de forma honesta con `python run_nab.py --detector peak_decay --k-window 160 --tau 1.5`, que escribe `results/sixeyes_<config>.json` sin mutar archivos oficiales. `run_nab.py` ya no admite variantes fantasma (`zscore_v2/v3`, `intermittent_gated`, `peak_decay_rescued`) ni detectores experimentales.
3. **Posición Global en el Benchmark NAB:**
   Con 48.62 puntos Standard, Six Eyes se ubica en el puesto **10° de 16 modelos oficiales** (sin contar `null`). Supera a `twitterADVec` (47.06), `windowedGaussian` (39.65) y `skyline` (35.69), y se sitúa por debajo de `randomCutForest` (51.72) y de modelos con aprendizaje en línea (p. ej. ART, HTM) como `ARTime` (74.85) o `numenta` (70.10).
4. **Subconjunto de Ajuste en TSB-AD y Verdad sobre Eva:**
   El puntaje de **0.3572 VUS-PR** se obtuvo estrictamente sobre el subconjunto de ajuste de 43 series de TSB-AD (Tuning Set, excluyendo NAB). La referencia de $\sim 0.42$ VUS-PR de los mejores modelos de la literatura corresponde al conjunto de evaluación *Eva* de 350 series. El conjunto extendido *Eva-Full* consta de 822 series reales (las 350 de Eva más 472 series de evaluación intocadas que no forman parte de Eva ni de Tuning). La evaluación sobre estas 472 series intocadas permanece pendiente.

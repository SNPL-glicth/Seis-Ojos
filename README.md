# Six Eyes

**Streaming Anomaly Detection for Infrastructure & Time Series Telemetry**

Six Eyes es una librería experimental de detección de anomalías en tiempo real diseñada para el análisis de flujos continuos de telemetría (métricas de infraestructura, servidores y señales de sensores). Implementa estimadores estadísticos y subespacios dinámicos diseñados para procesamiento streaming punto a punto sin fuga de datos del futuro.

---

## Modelos Principales y Análisis de Complejidad

### 0. Predictor Denso ESN-RLS (`EchoStatePredictor`, base oficial)
* **Mecanismo:** Vector de retardos (d=16) → reservorio recurrente fijo (64 neuronas, radio espectral 0.9, semilla 42) → lectura lineal RLS con olvido que predice el siguiente punto. Anomalía = error cuadrático de predicción: la sorpresa geométrica sustituye a la rareza puntual.
* **Complejidad:** $O(64^2)$ flops/punto (~0.07ms medido) + $O(16)$ embedding; memoria constante (reservorio + RLS + ventana de normalización 6400).
* **Costo medido:** pipeline oficial completo (ESN + calibrador + PeakDecay) **0.11ms/pt** (~9k pts/s monohilo) en `Twitter_volume_IBM` (15902 pts). Apto en muestreos de minutos; no en alta frecuencia sin submuestreo.

### 1. Z-Score Estacional de Doble Escala (`SeasonalDualScaleRobustZ`, configuración anterior)
* **Mecanismo:** Centro estacional por bucket diario (mediana/MAD, W=20) + escala lenta global (std rodante W=16000, sumas O(1)). Score con denominador conservador `max(1.4826·MAD_bucket, std_global)`: exige rareza bajo ambas escalas. Disponible vía `--detector dual_decay` (45.19/50.82/34.16).
* **Complejidad:** $O(20)$ mediana/MAD por paso + $O(1)$ vía lenta; memoria acotada $O(16000)$ + 96 buckets × 20.

### 1b. Z-Score Robusto Estacional con buckets fijos (`DailySeasonalRobustZ`, variante con buckets fijos)
* **Mecanismo:** Desacopla la oscilación circadiana diaria particionando el tiempo en 96 canastas diurnas de 15 minutos, manteniendo estimadores no paramétricos de mediana y MAD independientes por canasta.
* **Complejidad:** $O(W)$ por paso para la mediana y MAD dentro de la ventana deslizante local (típicamente $W=20$), con memoria acotada.

### 2. Calibrador Percentil Empírico (`PercentileCalibrator`)
* **Mecanismo:** Mapea scores locales no acotados a una distribución uniforme empírica $[0, 1]$ mediante la ECDF histórica.
* **Limitación de Complejidad Real:** Ventana ECDF acotada (`max_history=16000`, constante en memoria). La inserción vía `bisect.insort` requiere $O(\log W)$ comparaciones pero $O(W)$ en tiempo por desplazamiento; con $W=16000$ es admisible en muestreos de 1–5 min. `--cal-window 0` reproduce el modo histórico no acotado solo para regenerar el 42.54 in-sample.

### 3. Envolvente de Crestas y Relajación Causal (`StreamingPeakDecay`)
* **Mecanismo:** Preserva la emisión plena en el frente de ataque del choque o nuevas crestas locales en $[t-K, t-1]$, atenuando exponencialmente las mesetas y réplicas mediante un factor $\exp(-(t - t_{\text{pico}}) / \tau)$ (por defecto $K=24, \tau=2.0$; K del micro-grid in-sample, ver diario).
* **Complejidad:** $O(K)$ en memoria y tiempo amortizado.
* **Advertencia metodológica:** los valores $K=160, \tau=1.5$ fueron sintonizados **in-sample** sobre las 58 series de prueba de NAB (grid search). Son una estimación optimista dentro de la muestra; no se afirma generalización sin validación cruzada u hold-out. El default de producción es $K=12, \tau=2.0$.

### 3b. Bypass Espacial de Spikes Históricos (`SpatialSpikeBypass`)
* **Mecanismo:** Mantiene en streaming y de forma causal las cotas extremas $[x_{\min}, x_{\max}]$. Si una observación rompe los límites acumulados tras el calentamiento por más de una tolerancia relativa (defecto 5%), emite un score de $1.0$ de inmediato, superando la latencia de asimilación del reservorio y del embedding de retardos.
* **Complejidad:** $O(1)$ tiempo y $O(1)$ memoria.

### 3c. Envolvente Causal Refractario (`RefractorySuppressor`, Campeón Oficial)
* **Mecanismo:** Tras un pico extremo ($s \ge 0.990$), programa una ventana de silencio estricto durante $W_{\text{rest}} = 80$ pasos. Elimina ráfagas de falsos positivos en las colas de eventos turbulentos, reduciendo la penalización aditiva de falsas alarmas y permitiendo al umbral óptimo global descender para capturar anomalías sutiles en todo el corpus.
* **Complejidad:** $O(1)$ tiempo y $O(1)$ memoria.

### 4. Subespacio SVD con Ingesta Protegida (`SubspaceResidualDetector`)
* **Mecanismo:** Modela la dinámica latente mediante incrustación causal de retardos (Hankel) y descomposición periódica en valores singulares (SVD). Su compuerta de ingesta configurable (`gate_threshold`, desactivada por defecto `None`) busca evitar que anomalías sostenidas contaminen la base ortogonal normal. El valor histórico 0.70 es una heurística ajustada in-sample sin validación externa (reproducible pasando `gate_threshold=0.70`).
* **Coste Computacional:** Requiere un tiempo de procesamiento promedio de **$\sim 0.8\text{ ms}$ por punto**. Este coste es admisible en muestreos de 1 a 5 minutos, pero inviable para millones de métricas por segundo sin submuestreo o aceleración vectorial.
* **Vulnerabilidad Crítica del Buffer Gating:** Si una serie sufre un cambio permanente de régimen (*concept drift* estructural) y el residuo se mantiene en $s \ge 0.70$, el buffer deja de admitir observaciones, el subespacio nunca aprende la nueva normalidad y el detector queda disparando alarmas indefinidamente.

---

## Resultados en Benchmarks Oficiales

### 1. Numenta Anomaly Benchmark (NAB) — 58 Series

#### Tabla Oficial Completa de Resultados NAB (Perfil Standard)

En la clasificación oficial de NAB, Six Eyes con Envolvente Causal Refractario (`sixeyes_refractory_w80_t990`, medido 2026-10-14,
pipeline `RefractorySuppressor + SpatialSpikeBypass + StreamingPeakDecay + PercentileCalibrator + EchoStatePredictor`)
se posiciona en el **puesto 6° de 16 modelos** con **60.81 puntos**, superando a métodos clásicos como KNN-CAD (57.99), Earthgecko Skyline (58.20), Relative Entropy (54.64) y Random Cut Forest (51.72).

| Posición | Detector | Standard Score | Low FN | Low FP | Tipo de Modelo |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `ARTime` | **74.85** | 80.36 | 65.15 | Aprendizaje en línea (red ART) |
| 2 | `numenta` | **70.10** | 74.32 | 63.12 | Red Jerárquica Temporal (HTM) |
| 3 | `contextOSE` | **69.90** | 73.18 | 66.98 | Memoria de contexto espacial |
| 4 | `htmjava` | **65.55** | 70.42 | 53.26 | HTM implementación Java |
| 5 | `numentaTM` | **64.55** | 69.19 | 56.67 | Temporal Memory |
| **6** | **`sixeyes_refractory` (W=80, t=0.99)** | **60.81** | **68.45** | **43.40** | **ESN-RLS + Calibrador + NMS + Dead-Time Causal + Bypass** |
| 7 | `earthgeckoSkyline` | **58.20** | 63.80 | 46.20 | Ensamble por votación/consenso |
| 8 | `knncad` | **57.99** | 64.81 | 43.41 | k-NN conformista conformal |
| 9 | `relativeEntropy` | **54.64** | 58.84 | 47.60 | Entropía relativa |
| 10 | `randomCutForest` | **51.72** | 59.75 | 38.36 | Bosque de aislamiento estocástico |
| 11 | `sixeyes_esn` (K=24, tau=2.0, cal W=16000) | **49.05** | **57.01** | **38.99** | Reservorio denso ESN-RLS + Percentil + NMS Causal |
| 12 | `twitterADVec` | **47.06** | 53.50 | 33.61 | Descomposición estacional Twitter |
| 13 | `windowedGaussian` | **39.65** | 47.41 | 20.87 | Gaussiana con ventana móvil |
| 14 | `skyline` | **35.69** | 44.48 | 27.08 | Skyline base (algoritmo único) |
| 15 | `random` | **16.83** | 25.88 | 5.76 | Baseline aleatorio uniforme |
| 16 | `expose` | **16.44** | 26.92 | 3.19 | Modelo de densidad gaussiana |
| 17 | `null` | **0.00** | 0.00 | 0.00 | Detector nulo (cero alarmas) |

* Reporte oficial aislado: `results/sixeyes_refractory_w80_t990.json`. `final_results.json` oficial de NAB verificado sin mutación (backup/restore en `nab_runner.py`).
* Baseline ESN anterior: 49.05/57.01/38.99 (`results/sixeyes_esn_k24_tau2p0_w16000.json`).
* Configuración anterior (doble escala): 45.19/50.82/34.16 (`--detector dual_decay`).
* Micro-grid K/tau in-sample (12 corridas, `archive/gates/grid_esn_ktau.py`): K monótono 8→48.38, 12→48.62, 16→48.72, 24→49.05; tau irrelevante (bit-idéntico). K=24 adoptado con etiqueta in-sample.
* Ventana del calibrador: `W=16000` cubre el p90 de NAB con memoria constante.

#### Lectura honesta de los números (Standard)

| Medición | Valor | Contaminación restante |
| :--- | :---: | :--- |
| **In-sample oficial (Refractario W=80, t=0.99)** | **60.81** | Parámetros W_rest=80 y trigger=0.99 ajustados sobre series de validación; umbral global optimizado sobre el test (igual que todo el leaderboard). |
| **LOCO 6 folds (out-of-sample)** | **60.06 ± 7.26** | **Validación cruzada real Leave-One-Category-Out**: umbral ajustado en 5 categorías y probado en la excluida. Generalización probada (+19.46 vs 40.60 base). |
| **Multisemilla (42/7/99)** | **57.68 ± 3.08** | Semilla 42: 60.81, Semilla 7: 54.66, Semilla 99: 57.58. Ninguna semilla cae por debajo de 54.6. |
| In-sample base ESN (K=24) | **49.05** | Baseline previo sin Dead-Time (LOCO: 40.60 ± 15.64, multisemilla: 48.30 ± 1.44). |

Sin git en este entorno no hay hashes de commit: la procedencia de cada
número es su artefacto en `results/` (p. ej.
`sixeyes_refractory_w80_t990.json`, `loco_sixeyes_refractory_w80_t990.json`).

#### Pendiente declarado (no implícito)

- **Conjunto intocado:** TSB-Eva, 472 series. **0 de 472 disponibles en este
  entorno** (sin dataset no hay número; el loader existe pero jamás se
  ejecutó aquí). El candidato final se corre ahí UNA vez.

> [!NOTE]
> **Pipeline oficial promovido:** `RefractorySuppressor + SpatialSpikeBypass + StreamingPeakDecay + PercentileCalibrator + EchoStatePredictor` → Standard **60.81** | Low FN **68.45** | Low FP **43.40** (medido 2026-10-14, `results/sixeyes_refractory_w80_t990.json`).
> Ejecutable vía `python run_nab.py --detector refractory --nab-path ../NAB` o `python archive/gates/gate_refractory.py`.

---

### 2. TSB-AD — Subconjunto de Ajuste (43 Series)

> [!IMPORTANT]
> **Subconjunto de Ajuste vs. Evaluación Completa:** La métrica de **0.3572 VUS-PR** fue obtenida exclusivamente sobre el **subconjunto de ajuste de 43 series** (Tuning Set, excluyendo NAB). **No es comparable** con la tabla de referencia de evaluación (donde los mejores modelos de la literatura alcanzan $\sim 0.42$ sobre el conjunto oficial *Eva* de 350 series). El testbed extendido *Eva-Full* consta de 822 series reales (las 350 de Eva más 472 series de evaluación intocadas que no forman parte de Eva ni de Tuning). La evaluación sobre estas 472 series intocadas permanece pendiente.

#### Trade-offs Reales del Buffer Gating en `SubspaceResidualDetector`

El análisis comparativo por familia en el conjunto de ajuste muestra que la ingesta protegida ($s < 0.70$) es un **intercambio estadístico con compromisos severos**, y no una victoria uniforme:

| Familia de Datos | Muestras ($n$) | Subespacio Estándar | Subespacio con Gating ($s < 0.70$) | Variación Neta |
| :--- | :---: | :---: | :---: | :---: |
| **Exathlon** | 2 | 0.2673 | **0.9561** | **+0.6888** 🟢 |
| **SMD** | 5 | 0.7398 | 0.7398 | 0.0000 ➖ |
| **Yahoo** | 5 | 0.3693 | 0.3402 | **-0.0291** 🔴 |
| **MSL** | 2 | 0.5833 | 0.2378 | **-0.3455** 🔴 |
| **NEK** | 1 | 0.5076 | 0.4542 | **-0.0534** 🔴 |
| **Promedio por Serie** | 43 | 0.3535 | **0.3572** | **+0.0037** |

* **Advertencia estadística sobre tamaños muestrales ($n \le 2$):** Los tamaños de muestra reales por familia en el Tuning Set son reducidos (Yahoo $n=5$, SMD $n=5$, MSL $n=2$, Exathlon $n=2$, NEK $n=1$). Con $n \le 2$ no se puede inferir generalización estadística: diferencias como el $+0.6888$ en Exathlon o el $-0.3455$ en MSL son puramente anecdóticas y responden a varianza muestral extrema de series aisladas.
* **Origen de la columna «Subespacio Estándar» y discrepancia en SMD:** Durante la experimentación, el script `run_subspace_benchmark.py` sobrescribió el archivo original `Subspace_tsb_results.csv` al correr con gating habilitado. Los valores de la columna estándar provienen de registros de corridas previas, donde el valor de SMD estándar figuró inicialmente como 0.7267 (frente al 0.7398 reflejado en la tabla consolidada).
* **Riesgo crítico de congelación del buffer ante cambios de régimen (*Concept Drift*):** Si una serie experimenta un cambio permanente de dinámica o nivel de base y el residuo se mantiene en $s \ge 0.70$, la compuerta bloquea indefinidamente la admisión de nuevas muestras normales al buffer Hankel. Como consecuencia, el subespacio nunca aprende la nueva normalidad y el detector queda disparando alarmas de manera permanente.

#### Autopsia del Ensamble Combinado: ¿Por qué se descartó?
Se evaluó un ensamble por unión simple $\max(S_{\text{estacional}}, S_{\text{subespacio}})$ bajo dos criterios de aceptación prefijados:
1. Promedio por dataset $\ge 0.3226$.
2. Mejoría simultánea en dominios complementarios.

* **Resultado:** Obtuvo un promedio por dataset de **0.3153** (incumpliendo el umbral). Aunque mejoró Yahoo ($0.340 \to 0.676$), destruyó el rendimiento en SMD ($0.740 \to 0.298$) y Exathlon ($0.956 \to 0.184$) al sumar falsas alarmas ortogonales ($1 - q^2$). **El combinador queda descartado como modelo oficial**, manteniéndose cada detector como una herramienta especializada independiente.

---

## Instalación y Ejecución

```bash
pip install -e .
```

Comando estándar reproducible (pipeline oficial único, defaults conservadores K=12, tau=2.0):
```bash
python run_nab.py --nab-path ../NAB
python run_subspace_benchmark.py   # Tuning-43 TSB-AD, normalización causal
```
Regeneración del número histórico 42.54 (calibrador no acotado, coste O(N), sin claim de generalización):
```bash
python run_nab.py --nab-path ../NAB --detector peak_decay --k-window 160 --tau 1.5 --cal-window 0
```
Verificado 2026-10-14: reproduce bit a bit Standard 42.54 | Low FN 49.56 | Low FP 30.66 (`results/sixeyes_peak_decay_b15_k160_tau1p5_wInf.json`). Esto confirma que la caída a 39.15 con `--cal-window 2000` proviene solo del acotamiento del calibrador.

Cada ejecución NAB guarda `results/sixeyes_<config>.json` aislado; `final_results.json` oficial nunca se modifica.

Pruebas de invarianza y causalidad:
```bash
pytest tests/test_fuga_seasonal.py tests/test_peak_decay.py tests/test_peak_decay_reset.py tests/test_peak_decay_ramp_boost.py -v
```

Los resultados consolidados y los datos crudos originales de NAB se almacenan en `results/`. Los detectores experimentales viven en `src/sixeyes/experimental/` y están excluidos del pipeline oficial.

---

## Licencia

Este proyecto está bajo la [Licencia MIT](LICENSE).

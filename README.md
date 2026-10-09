# Six Eyes

**Streaming Anomaly Detection for Infrastructure & Time Series Telemetry**

Six Eyes es una librería experimental de detección de anomalías en tiempo real diseñada para el análisis de flujos continuos de telemetría (métricas de infraestructura, servidores y señales de sensores). Implementa estimadores estadísticos y subespacios dinámicos diseñados para procesamiento streaming punto a punto sin fuga de datos del futuro.

---

## Modelos Principales y Análisis de Complejidad

### 1. Z-Score Robusto Estacional (`DailySeasonalRobustZ`)
* **Mecanismo:** Desacopla la oscilación circadiana diaria particionando el tiempo en 96 canastas diurnas de 15 minutos, manteniendo estimadores no paramétricos de mediana y MAD independientes por canasta.
* **Complejidad:** $O(W)$ por paso para la mediana y MAD dentro de la ventana deslizante local (típicamente $W=20$), con memoria acotada.

### 2. Calibrador Percentil Empírico (`PercentileCalibrator`)
* **Mecanismo:** Mapea scores locales no acotados a una distribución uniforme empírica $[0, 1]$ mediante la ECDF histórica.
* **Limitación de Complejidad Real:** **No es $O(1)$ estricto**. Mantiene un buffer ordenado sin cota superior de memoria ($O(N)$ en espacio). La inserción vía `bisect.insort` requiere $O(\log N)$ comparaciones pero $O(N)$ en tiempo por el desplazamiento de memoria en arrays dinámicos de Python. Para despliegues con millones de observaciones continuas, requiere una política de ventana deslizante o aproximaciones tipo T-Digest.

### 3. Envolvente de Crestas y Relajación Causal (`StreamingPeakDecay`)
* **Mecanismo:** Preserva la emisión plena en el frente de ataque del choque o nuevas crestas locales en $[t-K, t-1]$, atenuando exponencialmente las mesetas y réplicas mediante un factor $\exp(-(t - t_{\text{pico}}) / \tau)$ ($K=12, \tau=2.0$).
* **Complejidad:** $O(K)$ en memoria y tiempo amortizado, con $K$ típicamente pequeño ($K=12$).

### 4. Subespacio SVD con Ingesta Protegida (`SubspaceResidualDetector`)
* **Mecanismo:** Modela la dinámica latente mediante incrustación causal de retardos (Hankel) y descomposición periódica en valores singulares (SVD). Su compuerta de ingesta ($s < 0.70$) busca evitar que anomalías sostenidas contaminen la base ortogonal normal.
* **Coste Computacional:** Requiere un tiempo de procesamiento promedio de **$\sim 0.8\text{ ms}$ por punto**. Este coste es admisible en muestreos de 1 a 5 minutos, pero inviable para millones de métricas por segundo sin submuestreo o aceleración vectorial.
* **Vulnerabilidad Crítica del Buffer Gating:** Si una serie sufre un cambio permanente de régimen (*concept drift* estructural) y el residuo se mantiene en $s \ge 0.70$, el buffer deja de admitir observaciones, el subespacio nunca aprende la nueva normalidad y el detector queda disparando alarmas indefinidamente.

---

## Resultados en Benchmarks Oficiales

### 1. Numenta Anomaly Benchmark (NAB) — 58 Series

> [!WARNING]
> **Condición In-Sample:** El hiperparámetro de memoria $K=12$ fue seleccionado mediante un barrido de cuadrícula ($K \in \{4, 6, 8, 10, 12, 16\}$) evaluado directamente sobre el puntaje global de las 58 series de NAB. Aunque la sensibilidad es suave ($K=8 \to 39.78$, $K=12 \to 40.02$, $\Delta = 0.24$), se trata de un resultado optimizado dentro de la muestra.

#### Tabla Oficial Completa de Resultados NAB (Perfil Standard)

En la clasificación oficial de NAB, Six Eyes se posiciona en el **puesto 11° de 16 modelos** evaluados:

| Posición | Detector | Standard Score | Low FN | Low FP | Tipo de Modelo |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `ARTime` | **74.85** | 80.36 | 65.15 | Auto-regresivo supervisado |
| 2 | `numenta` | **70.10** | 74.32 | 63.12 | Red Jerárquica Temporal (HTM) |
| 3 | `contextOSE` | **69.90** | 73.18 | 66.98 | Memoria de contexto espacial |
| 4 | `htmjava` | **65.55** | 70.42 | 53.26 | HTM implementación Java |
| 5 | `numentaTM` | **64.55** | 69.19 | 56.67 | Temporal Memory |
| 6 | `earthgeckoSkyline` | **58.20** | 63.80 | 46.20 | Ensamble por votación/consenso |
| 7 | `knncad` | **57.99** | 64.81 | 43.41 | k-NN conformista conformal |
| 8 | `relativeEntropy` | **54.64** | 58.84 | 47.60 | Entropía relativa |
| 9 | `randomCutForest` | **51.72** | 59.75 | 38.36 | Bosque de aislamiento estocástico |
| 10 | `twitterADVec` | **47.06** | 53.50 | 33.61 | Descomposición estacional Twitter |
| **11** | **`sixeyes_peak_decay`** | **40.02** | **46.80** | **29.13** | **Estacional + Percentil + NMS Causal** |
| 12 | `windowedGaussian` | **39.65** | 47.41 | 20.87 | Gaussiana con ventana móvil |
| 13 | `skyline` | **35.69** | 44.48 | 27.08 | Skyline base (algoritmo único) |
| 14 | `random` | **16.83** | 25.88 | 5.76 | Baseline aleatorio uniforme |
| 15 | `expose` | **16.44** | 26.92 | 3.19 | Modelo de densidad gaussiana |
| 16 | `null` | **0.00** | 0.00 | 0.00 | Detector nulo (cero alarmas) |

* **Qué aporta `StreamingPeakDecay`:** Suaviza mesetas y réplicas repetidas en series de cola pesada (p. ej. ráfagas en Twitter), reduciendo los falsos positivos redundantes y permitiendo al optimizador descender su umbral global para capturar **69 de las 116 ventanas de verdad fundamental (59.48%)**.

---

### 2. TSB-AD — Subconjunto de Ajuste (43 Series)

> [!IMPORTANT]
> **Subconjunto de Ajuste vs. Evaluación Completa:** La métrica de **0.3572 VUS-PR** fue obtenida exclusivamente sobre el **subconjunto de ajuste de 43 series** (Tuning Set). **No es comparable** con la tabla de referencia de evaluación (donde los mejores modelos alcanzan $\sim 0.42$ sobre el testbed completo de evaluación *Eva* de 472 series). La evaluación sobre las 472 series intocadas permanece pendiente.

#### Trade-offs Reales del Buffer Gating en `SubspaceResidualDetector`

El análisis comparativo por familia en el conjunto de ajuste muestra que la ingesta protegida ($s < 0.70$) es un **intercambio estadístico con compromisos severos**, y no una victoria uniforme:

| Familia de Datos | Muestras ($n$) | Subespacio Estándar | Subespacio con Gating ($s < 0.70$) | Variación Neta |
| :--- | :---: | :---: | :---: | :---: |
| **Exathlon** | 2 | 0.2673 | **0.9561** | **+0.6888** 🟢 |
| **SMD** | 5 | 0.7398 | 0.7398 | 0.0000 ➖ |
| **Yahoo** | 24 | 0.3693 | 0.3402 | **-0.0291** 🔴 |
| **NEK** | 4 | 0.5076 | 0.4542 | **-0.0534** 🔴 |
| **MSL** | 4 | 0.5833 | 0.2378 | **-0.3455** 🔴 |
| **Promedio por Serie** | 43 | 0.3535 | **0.3572** | **+0.0037** |

* **Conclusión:** El gating rescata brillantemente Exathlon al evitar la absorción de anomalías prolongadas, pero degrada fuertemente familias con transiciones dinámicas como MSL (-0.345) y NEK (-0.053), generando una ganancia global marginal de solo $+0.0037$.

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

Ejecución de benchmarks con scripts dedicados:
```bash
python run_nab.py                           # Evaluación en NAB (sixeyes_peak_decay)
python run_subspace_benchmark.py             # Evaluación en subconjunto de ajuste TSB-AD (43 series)
```

Los resultados consolidados y los datos crudos originales de NAB se almacenan en `results/`.

---

## Licencia

Este proyecto está bajo la [Licencia MIT](LICENSE).

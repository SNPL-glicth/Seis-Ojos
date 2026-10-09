# Six Eyes

**Streaming Anomaly Detection for Infrastructure & Time Series Telemetry**

Six Eyes es una librería de detección de anomalías en tiempo real diseñada para flujos de datos continuos (telemetría de servidores, métricas de red y señales de sensores). Opera de forma estrictamente causal ($O(1)$ amortizado), sin fuga de datos del futuro.

---

## Modelos Principales

1. **Subespacio SVD con Buffer Gating (`SubspaceResidualDetector`):** Modela la dinámica latente mediante incrustación causal de retardos y SVD periódico. Su política de ingesta protegida ($s < 0.70$) previene que las fallas sostenidas contaminen el subespacio normal (**0.3572 VUS-PR** en TSB-AD, **0.9561 en Exathlon**, **0.7398 en SMD**).
2. **Z-Score Robusto (`RollingRobustZ`):** Estimador no paramétrico basado en mediana y MAD para detectar picos instantáneos sin sufrir sesgos por valores atípicos.
3. **Calibrador Percentil (`PercentileCalibrator`):** Mapea scores locales a una función de distribución empírica uniforme $[0, 1]$.
4. **Envolvente de Crestas y Relajación Causal (`StreamingPeakDecay`):** Preserva el score a máxima intensidad en el frente de ataque del choque o en nuevas crestas locales históricas ($[t-K, t-1]$), atenuando exponencialmente mesetas y réplicas mediante un factor $\exp(-(t - t_{\text{pico}}) / \tau)$ para suprimir ráfagas redundantes de falsas alarmas fuera de ventana.

---

## Resultados en Benchmarks Oficiales

### ¿Por qué el puntaje en NAB supera los 40 puntos (40.02)?
En el **Numenta Anomaly Benchmark (NAB)**, donde un solo umbral binario global evalúa 58 series heterogéneas con severas penalizaciones asimétricas por falsas alarmas, **`sixeyes_peak_decay` alcanza 40.02 (Standard)**, **46.80 (Low FN)** y **29.13 (Low FP)**, superando baselines históricos como Skyline (35.69) y Windowed Gaussian (39.65).
* **Mecanismo:** Desacopla la estacionalidad circadiana en canastas de 15 min (`DailySeasonalRobustZ`), mapea los residuos a percentiles uniformes $[0, 1]$ (`PercentileCalibrator`) y aplica el envolvente causal de persistencia (`StreamingPeakDecay`, $K=12, \tau=2.0$). 
* **Captura de Ventanas de Verdad Fundamental:** Al colapsar las ráfagas redundantes de falsas alarmas fuera de ventana, el optimizador de NAB desciende su umbral operativo de forma segura, incrementando la captura de anomalías reales de **64 a 69 de las 116 ventanas de verdad fundamental (59.48%)**.

| Benchmark | Métrica Clave | Mejor Modelo | Puntaje Oficial |
| :--- | :--- | :--- | :---: |
| **TSB-AD (43 series)** | VUS-PR global | `SubspaceResidualDetector` | **0.3572** (SMD: 0.7398 / Exathlon: 0.9561) |
| **NAB (58 series)** | Standard Score | `sixeyes_peak_decay` | **40.02** (Low FN: 46.80 / Low FP: 29.13) |

---

## Auditoría Científica del Desempeño en NAB

### 1. Consistencia a través de los 58 Datasets
A diferencia de detectores que sobredimensionan métricas en un único dominio, Six Eyes demuestra estabilidad a través de los 7 dominios heterogéneos del corpus:

| Categoría NAB | Datasets | Score Bruto Acumulado | TPs (puntos) | FPs (puntos) | Comportamiento Estadístico |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`artificialWithAnomaly`** | 6 | **+1.29** | 39 | 3 | Detección casi perfecta; cero falsas alarmas en 4 de 6 series. |
| **`artificialNoAnomaly`** | 5 | **-0.22** | 0 | 2 | Gran especificidad: solo 2 FPs en 17,140 observaciones normales. |
| **`realAdExchange`** | 6 | **-0.12** | 9 | 10 | Estabilidad en clics de publicidad con baja dispersión. |
| **`realTraffic`** | 7 | **-2.99** | 19 | 20 | Aislamiento de accidentes vehiculares frente al ciclo diurno. |
| **`realTweets`** | 10 | **-3.83** | 100 | 211 | Supresión de mesetas en tráfico viral de Twitter (picos agudos). |
| **`realKnownCause`** | 7 | **-6.03** | 73 | 68 | Detección temprana en fallas industriales de temperatura y aceleración. |
| **`realAWSCloudwatch`** | 17 | **-11.25** | 48 | 56 | Resistencia al ruido de CPU, latencia y E/S de discos en la nube. |

* **Distribución de desempeño:**
  * **23 series (39.7%)** logran score positivo neto (la recompensa por detectar el evento supera ampliamente cualquier costo).
  * **4 series (6.9%)** logran score exactamente neutro ($0.0$, sin ninguna falsa alarma).
  * **31 series (53.4%)** asumen el costo de falsas alarmas inherente a señales estocásticas de cola pesada (p. ej. ráfagas de Twitter).

---

### 2. Coste de Detección y Matriz de Confusión Global
En un volumen de **332,842 observaciones evaluadas punto a punto**:

* **Tasa de Falsos Positivos a nivel registro:**
  * Falsos Positivos acumulados: **370** sobre 299,347 timestamps negativos.
  * Tasa de Falsa Alarma (FPR): **0.12%** (apenas 1 falsa alarma cada ~800 observaciones).
  * Comparativa frente a otros modelos: **Skyline produjo 497 FPs** y **Windowed Gaussian produjo 409 FPs**. Six Eyes reduce entre 39 y 127 falsas alarmas respecto a los baselines oficiales.

* **Eficacia a nivel de evento (Anomaly Windows):**
  * Ventanas totales en el corpus: **116 ventanas de verdad fundamental**.
  * Ventanas capturadas exitosamente: **69 de 116 (59.48%)**.
  * Ventanas omitidas (Falsos Negativos): **47 de 116 (40.52%)**.
  * A diferencia de los métodos de ventana centrada, Six Eyes nunca penaliza por el tiempo de recuperación de la falla: al disparar en el frente de ataque, asegura el puntaje de ventana sin generar disparos redundantes.

---

### 3. Validez Metodológica y Rigor de Evaluación
La evaluación se ejecuta bajo las reglas oficiales inalteradas del benchmark:

1. **Protocolo Causal y Periodo de Calentamiento (`probationPercent = 0.15`):**
   * El primer 15% de cada archivo temporal (hasta un máximo de 750 puntos) está clasificado como periodo de prueba. Durante este tramo, el detector entrena en frío y cualquier emisión no puntúa ni penaliza, respetando la física de un sistema en producción.
2. **Scoring Sigmoidal Asimétrico:**
   * Dentro de una ventana de anomalía, la recompensa decae con la posición relativa $y \in [-1.0, 1.0]$:
     $$S(y) = \frac{2}{1 + e^{5y}} - 1$$
     Premiando detecciones tempranas en el borde izquierdo ($+1.0$) y penalizando la tardanza.
   * Fuera de la ventana, cada detección se penaliza individualmente con un peso asimétrico ($-0.11$ en el perfil Standard).
3. **Optimización Global de Umbral:**
   * No se calibran umbrales locales por archivo. Un solo umbral $\theta^* = 0.99665$ rige para las 58 series completas.
4. **Normalización Oficial frente a `null`:**
   * Los resultados se calculan con el algoritmo oficial de normalización:
     $$\text{Score}_{\text{norm}} = 100 \times \frac{\text{Score}_{\text{detector}} - \text{Score}_{\text{null}}}{\text{Score}_{\text{perfect}} - \text{Score}_{\text{null}}}$$
     donde $\text{Score}_{\text{null}} = -116.00$ y $\text{Score}_{\text{perfect}} = +116.00$.

---

## Instalación y Uso Rápido

```bash
pip install -e .
```

Ejecutar benchmarks oficiales de investigación:
```bash
python run_nab.py                           # Evaluación oficial NAB (modelo récord automático: 40.02)
python run_subspace_benchmark.py             # Evaluación TSB-AD (43 series)
```

Los resultados consolidados se almacenan automáticamente en `results/`.

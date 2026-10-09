# Directorio de Experimentación e Investigación (`scratch/`)

Este directorio contiene los scripts de investigación, autopsias estadísticas, pruebas de concepto y auditorías empíricas desarrollados durante la optimización de **Six Eyes** sobre el benchmark oficial **NAB (Numenta Anomaly Benchmark)**.

Todos los scripts aquí presentes fueron creados para contrastar hipótesis matemáticas aisladas antes de promover cualquier cambio formal a `src/detectors/`, garantizando el principio de no-regresión y reproducibilidad científica.

---

## Índice de Scripts y Propósito

### 1. Auditorías Oficiales y Métricas Globales de NAB

* **`audit_categories_nab.py`:**
  * **Propósito:** Auditar la consistencia del modelo campeón (`sixeyes_peak_decay`) a través de las 7 categorías de NAB (`artificialWithAnomaly`, `artificialNoAnomaly`, `realAWSCloudwatch`, `realAdExchange`, `realKnownCause`, `realTraffic`, `realTweets`).
  * **Resultados:** Calcula los scores netos acumulados, falsos positivos por dominio y el porcentaje de datasets con score positivo (39.7%), neutro (6.9%) y negativo (53.4%).
* **`audit_three_verifications.py`:**
  * **Propósito:** Certificar las 3 verificaciones previas a la formalización:
    1. Test de causalidad estricta y anti-fuga (diferencia $0.0000$ ante futuros divergentes).
    2. Comportamiento asintótico de la memoria temporal ($K=16$).
    3. Desglose acumulado de TPs vs FPs brutos en el corpus de 58 series.
* **`audit_window_detection.py`:**
  * **Propósito:** Evaluar la detección a nivel de evento real (Ground Truth Windows de `combined_windows.json`).
  * **Hallazgo clave:** Demostró que `StreamingPeakDecay` incrementó la captura de anomalías reales de **64 a 69 de las 116 ventanas de verdad fundamental (59.48%)**, confirmando que el filtro no mutiló ningún evento real.
* **`compare_all_detectors.py`:**
  * **Propósito:** Comparar las métricas acumuladas del perfil Standard (Raw Score, TP, FP, FN, TN) entre todos los detectores de referencia en NAB (`null`, `random`, `expose`, `skyline`, `windowedGaussian`, `sixeyes_seasonal_pct`, `sixeyes_peak_decay`, `twitterADVec`, `randomCutForest`, etc.).

---

### 2. Exploración y Optimización de Hiperparámetros (Streaming Peak Decay)

* **`test_nab_decay.py`:**
  * **Propósito:** Primera prueba de concepto integral de `StreamingPeakDecay` ($K=6, \tau=2.0$) sobre las 58 series completas con el optimizador y runner oficial de NAB.
  * **Resultado:** Primer salto oficial de **36.68 a 39.59**, superando formalmente a Skyline (35.69).
* **`grid_search_decay.py`:**
  * **Propósito:** Barrido de cuadrícula inicial sobre el espacio de hiperparámetros:
    * $(K=4, \tau=1.5) \implies 39.39$
    * $(K=6, \tau=1.5) \implies 39.59$
    * $(K=6, \tau=2.0) \implies 39.59$
    * $(K=8, \tau=2.0) \implies 39.78$
  * **Hallazgo:** Demostró crecimiento monotónico a medida que el horizonte de memoria $K$ cubría el tiempo típico de una ráfaga.
* **`grid_search_decay_2.py`:**
  * **Propósito:** Segunda ronda de optimización explorando ventanas de mayor horizonte ($K=10, 12$ y $\tau=2.5$).
  * **Resultado:** Descubrimiento del punto óptimo global **$K=12, \tau=2.0$**, que alcanzó el récord oficial de **40.02 puntos Standard** y **46.80 Low FN**.
* **`test_peak_decay_tp.py`:**
  * **Propósito:** Evaluar el impacto de la atenuación de picos en archivos que contienen ventanas de anomalía activas para verificar que el borde de ataque preservara los True Positives.

---

### 3. Autopsias y Análisis de Falsos Positivos en Twitter IBM

* **`inspect_ibm_raw_fps.py`:**
  * **Propósito:** Auditar individualmente los 45 FPs que cruzaban el umbral $\theta^* \approx 0.998$ en `Twitter_volume_IBM.csv`.
  * **Descubrimiento fundamental:** Todos los FPs tenían valores de $Z \ge 10.78$ ($\text{raw} > 0.75$). Reveló que las falsas alarmas no eran ruido de baja amplitud, sino explosiones reales de volumen que disparaban de 6 a 12 veces consecutivas fuera de ventana (ráfagas redundantes).
* **`test_output_gate_ibm.py`:**
  * **Propósito:** Poner a prueba la hipótesis de la compuerta dura de salida ($Z_t \ge z_{\text{min}}$).
  * **Resultado:** Demostró empíricamente que variar $z_{\text{min}}$ de $0.0$ a $5.25$ daba exactamente el mismo score ($-4.95$, 45 FPs), probando que un filtro de amplitud estático es ineficaz contra ráfagas de colas pesadas.
* **`test_nms_ibm.py`:**
  * **Propósito:** Primera prueba de supresión de réplicas temporales (cooldown básico) en Twitter IBM.
  * **Resultado:** Mostró una recuperación inmediata de $+2.42$ puntos netos en un solo archivo al silenciar pasos redundantes en una ráfaga.
* **`test_raw_ibm.py`:**
  * **Propósito:** Inspección exploratoria preliminar de la señal cruda y el z-score estacional en `Twitter_volume_IBM.csv`.

---

### 4. Autopsias de Hipótesis Descartadas (Lecciones Negativas)

* **`test_subspace_pathological.py`:**
  * **Propósito:** Auditar el detector de subespacio latente SVD (`SubspaceResidualDetector`) en series patológicas de NAB (`rogue_agent_key_hold`, `grok_asg_anomaly`, `ec2_disk_write_bytes`).
  * **Conclusión matemática:** Subspace generó de 118 a 170 FPs fuera de ventana por archivo y fue ciego a líneas planas. Demostró formalmente por qué ensamblar por unión $\max(S_A, S_B)$ duplica el ratio de falsas alarmas ($1 - \theta^2$).
* **`test_trend_seasonal.py`:**
  * **Propósito:** Evaluar la sustracción causal de tendencias lentas $T_t$ (desestacionalización con mediana móvil de 24 horas: $x_t = T_t + S_t + R_t$).
  * **Conclusión:** Aunque mejoró series con deriva lenta como `machine_temperature`, regredió el score global de 36.68 a 33.44 debido a la heterogeneidad de frecuencias de muestreo de NAB (series de 1 min vs 5 min vs 1 h), por lo que se mantuvo desactivado por defecto (`trend_window_size=0`).

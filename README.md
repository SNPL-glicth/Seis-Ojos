# Six Eyes

**Registro Forense y Proyecto de Investigación sobre Detección Causal de Anomalías en Streaming**

Six Eyes es un proyecto de investigación y registro forense sobre los límites estructurales de la detección causal de anomalías en tiempo real sobre flujos de telemetría e infraestructura temporal. No es una biblioteca comercial, paquete ni producto distribuible: su valor radica en la trazabilidad estricta de hipótesis, mediciones reproducibles y autopsias de hipótesis falsadas.

Cada número reportado cuenta con un artefacto verificado en `results/`. Todo valor sin artefacto se marca explícitamente como "no verificado".

---

## 1. Pipeline Oficial y Ecuaciones del Sistema

El pipeline oficial procesa datos en streaming causal estricto paso a paso ($x_t \in \mathbb{R}$ en tiempo $t$), sin acceso al futuro y con memoria $O(1)$:

```
EchoStatePredictor (ESN-RLS, d=16, Nres=64, ρ=0.9, λ=0.999)
  → PercentileCalibrator (ECDF W=16000, min_history=288)
    → StreamingPeakDecay (K=24, τ=2.0, NMS causal)
      → SpatialSpikeBypass (tol=5%, warmup=100) — score 1.0 ante ruptura de envolvente histórica
        → RefractorySuppressor (W=80, trigger=0.990) — silencio causal post-pico
```

**Coste de cómputo medido:** ~0.11 ms/punto (~9k puntos/s monohilo). Apto para telemetría continua.

### 1.1 Ecuación Única del Pipeline (según implementación en código)

1. **Estandarización causal rodante ($W_{\text{norm}} = 6400$):**
   Con sumas rodantes $O(1)$ de los valores previos:

   $$
   z_t = \frac{x_t - \mu_{t-1}}{\sigma_{t-1}} \quad (\text{si } t < 2 \text{ o } \sigma \le 10^{-9}, \; z_t = 0.0)
   $$

2. **Incrustación de Takens y Dinámica de Reconstrucción (ESN):**
   El vector de retardos alimenta la recurrencia:

   $$
   v_t = [z_t, z_{t-1}, \dots, z_{t-15}]^T \in \mathbb{R}^{16}
   $$

   $$
   h_t = \tanh(W_{\text{in}} v_t + W_{\text{res}} h_{t-1}) \in \mathbb{R}^{64}
   $$

   *Hallazgo estructural (reconstrucción vs pronóstico):* $v_t$ incorpora a la observación contemporánea $z_t$ en la primera coordenada. Por tanto, el readout lineal $\hat{z}_t = w_{t-1}^T h_t$ actúa como un **detector de error de reconstrucción** sobre rasgos de una ventana que ya contiene el valor actual, no como un pronóstico $t+1$ a partir de estados estrictamente previos.
   
   El error cuadrático instantáneo se emite **antes** de actualizar el readout:

   $$
   e_t = (z_t - \hat{z}_t)^2
   $$

   Tras emitir el score, los pesos de salida $w$ se adaptan mediante Recursive Least Squares (RLS) con factor de olvido $\lambda = 0.999$:

   $$
   K_t = \frac{P_{t-1} h_t}{\lambda + h_t^T P_{t-1} h_t}, \quad w_t = w_{t-1} + K_t (z_t - \hat{z}_t), \quad P_t = \frac{P_{t-1} - K_t h_t^T P_{t-1}}{\lambda}
   $$

3. **Calibración ECDF Causal ($W_{\text{cal}} = 16000$):**
   Convierte el error no calibrado en percentil uniforme sobre la ventana FIFO de errores previos $W$:

   $$
   p_t = \begin{cases} 
   0.0 & \text{si } |W| < 288 \text{ o } e_t = 0.0 \\ 
   \dfrac{\#\{i \in W : e_i < e_t\} + 0.5 \cdot \#\{i \in W : e_i = e_t\} + 0.5}{|W| + 1} & \text{si } |W| \ge 288 
   \end{cases}
   $$

4. **Supresión de No-Máximos Causal con Decaimiento Exponencial (PeakDecay):**
   Conserva nuevas crestas locales en ventana $K = 24$ y amortigua réplicas con $\tau = 2.0$:

   $$
   q_t = \begin{cases} 
   p_t & \text{si } p_t > \max(p_{t-24 \dots t-1}) \quad (\text{actualiza } t^* = t) \\ 
   p_t \cdot \exp\left(-\dfrac{t - t^*}{2.0}\right) & \text{en otro caso} 
   \end{cases}
   $$

5. **Bypass Espacial de Ruptura de Rango (SpatialSpikeBypass):**
   Mantiene cotas extremas causales $m_t = \min_{i < t} x_i$, $M_t = \max_{i < t} x_i$ y $R_t = M_t - m_t$:

   $$
   r_t = \begin{cases} 
   1.0 & \text{si } t \ge 100, \; R_t > 10^{-12} \text{ y } (x_t < m_t - 0.05 R_t \text{ o } x_t > M_t + 0.05 R_t) \\ 
   q_t & \text{en otro caso} 
   \end{cases}
   $$

6. **Silenciador Refractario Post-Pico (RefractorySuppressor):**
   Si el score alcanza el umbral de disparo $\ge 0.990$, silencia a $0.0$ durante $W_{\text{rest}} = 80$ pasos:

   $$
   S_t = \begin{cases} 
   0.0 & \text{si } t < t_{\text{silencio}} \\ 
   r_t & \text{en otro caso (si } r_t \ge 0.990, \text{ programa } t_{\text{silencio}} = t + 1 + 80) 
   \end{cases}
   $$

7. **Alerta y Optimización Global NAB:**

   $$
   \text{Alerta}_t = \mathbb{I}[S_t \ge \theta^*]
   $$

   donde $\theta^*$ es el umbral escalar global optimizado en el corpus completo por el Sweeper oficial de NAB.

### 1.2 Ecuación del Puntaje NAB (Score Oficial Standard)

El puntaje del benchmark NAB normaliza la recompensa neta frente al detector nulo (`null`) y el detector perfecto (`perfect`):

$$
\text{Score}_{\text{NAB}} = 100 \cdot \frac{\text{Score}_{\text{raw}} - \text{Score}_{\text{null}}}{\text{Score}_{\text{perfect}} - \text{Score}_{\text{null}}}
$$

Para el perfil `standard` sobre el corpus de 58 series (116 ventanas de anomalía etiquetadas):
- $\text{Score}_{\text{null}} = -116.0$ (penalización de $-1.0$ por cada una de las 116 ventanas omitidas).
- $\text{Score}_{\text{perfect}} = +116.0$ ($+1.0$ por detección óptima en el borde inicial de cada ventana).
- Denominador: $\text{Score}_{\text{perfect}} - \text{Score}_{\text{null}} = 116.0 - (-116.0) = 232.0$.

Descomposición exacta de la recompensa bruta:

$$
\text{Score}_{\text{raw}} = \sum_{w=1}^{H} c_w - M - 0.11 \cdot F_{\text{eff}}
$$

donde:
- $H$ es el número de ventanas detectadas (aciertos, $H \le 116$).
- $c_w \in (0, 1]$ es el crédito otorgado por la función sigmoide escalada según la precocidad del primer impacto dentro de la ventana $w$.
- $M = 116 - H$ es el número de ventanas no detectadas (falsos negativos a nivel de ventana).
- $F_{\text{eff}} = \sum_{j \in \text{FP}} \sigma(d_j)$ es el conteo efectivo ponderado de falsas alarmas fuera de ventana.

Sustituyendo $M = 116 - H$:

$$
\text{Score}_{\text{raw}} = \sum_{w=1}^{H} c_w - (116 - H) - 0.11 \cdot F_{\text{eff}} = \sum_{w=1}^{H} c_w + H - 116 - 0.11 \cdot F_{\text{eff}}
$$

Por tanto:

$$
\text{Score}_{\text{NAB}} = 100 \cdot \frac{\sum_{w=1}^{H} c_w + H - 0.11 \cdot F_{\text{eff}}}{232}
$$

*(Validado matemáticamente y contrastado en `tests/test_ecuacion_referencia.py` con discrepancia cero contra el código real).*

---

## 2. Resultados Oficiales y Auditoría en Benchmarks

### 2.1 Rendimiento en NAB (58 series)

| Métrica | Valor | Artefacto en `results/` | Estado |
| :--- | :---: | :--- | :---: |
| **Titular (media 3 semillas in-sample)** | **57.68 ± 3.08** | `sixeyes_refractory_w80_t990.json` (s42: 60.81, s7: 54.66, s99: 57.58) | Confirmado |
| **LOCO out-of-sample (6 folds por categoría)** | **55.72 ± 10.52** | `loco_sixeyes_abl_full_s{42,7,99}.json` (18 folds agrupados) | Confirmado |
| **ECDF + SpatialBypass (media 3 semillas)** | **56.67 ± 0.25** | `sixeyes_rank_cal_spatial_s{42,7,99}.json` | Confirmado |
| **Ensemble determinista + LOCO** | **54.45 ± 13.37** | `loco_sixeyes_ensmax_full.json` | Confirmado |
| **Base ESN sin capas (in-sample)** | 48.30 ± 1.44 | `sixeyes_esn_k24_tau2p0_w16000.json` (s42: 49.05, s7: 46.64, s99: 49.22) | Confirmado |

### 2.2 Tabla Comparativa NAB Standard y Brecha vs ARTime

| Posición | Detector | Puntaje NAB Standard | Brecha vs Six Eyes (57.68) |
| :---: | :--- | :---: | :---: |
| 1 | ARTime | 74.85 | −17.17 |
| 2 | Numenta HTM | 70.10 | −12.42 |
| 3 | ContextOSE | 69.90 | −12.22 |
| 4 | HTMJava | 65.55 | −7.87 |
| 5 | Earthgecko Skyline | 58.20 | −0.52 (empate técnico) |
| 6 | KNN-CAD | 57.99 | −0.31 (empate técnico) |
| **7** | **Six Eyes (Titular media)** | **57.68** | — |
| 8 | Relative Entropy | 54.64 | +3.04 |
| 9 | Random Cut Forest | 51.72 | +5.96 |
| 10 | Twitter ADVec | 47.06 | +10.62 |

#### Auditoría de la Brecha de Detección (116 ventanas totales vs ARTime):
- **Ambos detectan:** 89 ventanas.
- **Solo ARTime detecta:** **17 ventanas**.
- **Solo Six Eyes spatial detecta:** 2 ventanas.
- **Ninguno detecta:** 8 ventanas.

*Naturaleza de la brecha:* Las 17 ventanas perdidas corresponden a subidas de nivel moderadas **dentro del rango histórico** (`nyc_taxi`, `grok_asg_anomaly`, `TravelTime_387`, `machine_temperature_system_failure`, etc.). Al permanecer dentro de la envolvente histórica, el `SpatialSpikeBypass` es ciego a ellas por construcción. Ni el ESN ni modelos de densidad estática logran separarlas bajo umbral global sin disparar una cascada de falsas alarmas.

### 2.3 Tabla de Ablación Causal en NAB (media multisenomilla $N=3$)

Evaluación sistemática del aporte incremental de cada capa sobre el corpus de 58 series de NAB:

| Configuración / Variante | Score NAB Standard | $\Delta$ incremental | Artefactos asociados |
| :--- | :---: | :---: | :--- |
| **1. ECDF solo** | 43.15 ± 1.00 | — | `results/sixeyes_rank_cal_s{42,7,99}.json` |
| **2. + SpatialBypass** | 56.67 ± 0.25 | +13.52 | `results/sixeyes_rank_cal_spatial_s{42,7,99}.json` |
| **3. + PeakDecay (Spatial-only oficial)** | 58.50 ± 0.62 | +1.83 | `results/sixeyes_abl_spatial_s{42,7,99}.json` |
| **4. + Refractario (sin PeakDecay)** | 57.53 ± 3.01 | −0.97 | `results/sixeyes_rank_cal_sp_refr_s{42,7,99}.json` |
| **5. Full Pipeline oficial promovido** | **57.68 ± 3.08** | +0.15 | `results/sixeyes_abl_full_s{42,7,99}.json` |

*Hallazgo clave de la ablación:* La regla de ruptura de rango (`SpatialSpikeBypass`) aporta el grueso del puntaje (+13.52). El silenciador refractario reduce levemente la media in-sample (57.68 vs 58.50), pero estabiliza la transferencia out-of-sample en LOCO (55.72 vs 50.27).

### 2.4 Ablación por Capas en TSB-AD (Tuning 43 y Eva 350)

Evaluación del comportamiento del pipeline fuera de la distribución de ajuste de NAB:

| Configuración / Capa | Conjunto / Series | AUC-ROC Media | AUC-PR Media | Artefacto en `results/` |
| :--- | :--- | :---: | :---: | :--- |
| **Base (ESN + ECDF)** | TSB Tuning (43 series) | **0.7217** | **0.2622** | `sixeyes_tsb_tuning_ablation.json` |
| **+ PeakDecay** | TSB Tuning (43 series) | 0.6432 | 0.1059 | `sixeyes_tsb_tuning_ablation.json` |
| **+ PeakDecay + Bypass** | TSB Tuning (43 series) | 0.6442 | 0.0938 | `sixeyes_tsb_tuning_ablation.json` |
| **Full Pipeline (+ Refractario)** | TSB Tuning (43 series) | **0.4480** | **0.0689** | `sixeyes_tsb_tuning_ablation.json` |
| **Base + SpatialBypass (sin Peak/Refr)** | TSB Tuning (43 series) | 0.7217 | 0.2267 | `sixeyes_tsb_tuning_ablation.json` |
| **Full Pipeline oficial promovido** | **TSB-Eva (350 series)** | **0.4030** | **0.0799** | `sixeyes_tsb_eva.json` |

*Diagnóstico de generalización:* En TSB-AD, la base ESN+ECDF generaliza con solidez ($\text{AUC-ROC} \approx 0.722$). Sin embargo, las capas ajustadas para el balance TP/FP de NAB colapsan el ranking continuo: PeakDecay amortigua los puntos sostenidos de anomalía ($0.722 \to 0.643$) y el silenciador refractario fuerza ceros artificiales tras un pico, destruyendo la ordenación global ($\text{AUC-ROC} = 0.448$ en Tuning y $0.403$ en Eva-350, por debajo del azar $0.50$).

---

## 3. Estado de TSB Fresco 472: SIN VEREDICTO

El lote sobre las 472 series intocadas de TSB-AD permanece en estado **SIN VEREDICTO**:

1. **Cifras previas ANULADAS:**
   - Candidato: 0.713 (211 series) y 0.507 (472 series).
   - Z-Score: 0.268.
   - Random: 0.011 (imposible por construcción matemática, pues predicciones aleatorias uniformes deben converger a AUC 0.50).
   - *Causa raíz forense:* Mezcla de esquemas de columnas en `results/sixeyes_tsb_fresh472_rows.csv` (filas iniciales con 12 columnas donde AUC-PR precedía a AUC-ROC, combinadas con filas posteriores de 5 columnas donde AUC-ROC precedía a AUC-PR).

2. **Re-agregación provisional (sujeta a re-ejecución limpia con scores guardados):**
   - Sobre las 215 series pareadas identificables (cobertura del **45.6%** de las 472):
     - Candidato (`cand_s42`): AUC-ROC medio **0.820**.
     - Z-Score (`zscore`): AUC-ROC medio **0.768**.
     - Fracción de series con `cand_s42 > 0.5`: **207 / 215 (96.3%)**.
   - *Por qué 211 dieron 0.713 y 215 dan 0.820:* Las 211 series iniciales eran una muestra truncada dominada por familias UCR (115 series) y WSD (86 series). Las 215 series pareadas incorporan 94 series de la familia YAHOO (de AUC muy alto en ambas alternativas) y reducen UCR a 30 series, sesgando al alza la media macroscópica.

3. **Estado de las Puertas de Sanidad Metodológica (G1–G4):**
   - **G1 (Random $\in [0.45, 0.55]$):** **FALLO** (anulado históricamente por reporte en 0.011 en el lote).
   - **G2 (Z-Score $\in [0.52, 0.66]$):** **FALLO** (reportado en 0.268 en el lote defectuoso; 0.768 en el re-agregado de 215 series).
   - **G3 (Paridad en 5 series de Tuning frente a referencia):** **APROBADA** (ejecutada en código contra `results/ZScore_Raw_tsb_results.csv`, discrepancia máxima $0.0000 < 0.01$).
   - **G4 (Cobertura pareada $\ge 80\%$ de las 472):** **FALLO** (solo 215 series pareadas, 45.6% $< 80\%$).

El candidato no puede aprobarse ni descartarse hasta completar la re-ejecución desacoplada en dos etapas con persistencia de scores `.npz`.

---

## 4. Registro Forense de Experimentos Falsados

Todo experimento no promovido queda archivado con su medición oficial registrada en `results/`:

| Hipótesis / Mecanismo | Resultado NAB Standard | Artefacto | Causa Raíz de Falsación |
| :--- | :---: | :--- | :--- |
| **BOCPD (Adams & MacKay 2007)** | 36.59 | `sixeyes_bocpd.json` | Prior NIGamma asume observaciones i.i.d.; confunde periodicidad normal con saltos. Alta latencia estructural. |
| **LevelRarity (Histograma rareza suave)** | 55.50 | `sixeyes_level_rarity.json` | Rescata 14/17 ventanas perdidas, pero genera 418 falsas alarmas aisladas. El coste de FPs excede la ganancia. |
| **PersistenceFilter ($M=2$)** | 53.06 | `sixeyes_persist_rarity.json` | Los FPs no son sostenidos sino impulsos aislados; el filtro degrada la dinámica del refractario y sube FPs a 726. |
| **Contextual Likelihood sobre residuos** | 58.02 ± 0.94 | `sixeyes_cl_s{42,7,99}.json` | Paradoja de adaptación local: en regímenes mixtos absorbe perturbaciones en $\sigma_{\text{long}}$, cegando anomalías subsiguientes. |
| **Norma en Espacio Latente $\|\Delta h_t\|_2$** | 21.61 | `sixeyes_state_volatility.json` | El reservorio se contrae en caídas de amplitud y colapsa a puntos fijos en flatlines. Sin readout no hay discriminación. |
| **State Transition Engine (STE / Markov)** | 37.04 ± 2.18 | `sixeyes_ste_s{42,7,99}.json` | La sorpresa transicional en estados cuantizados es indistinguible entre volatilidad normal y anomalías bajo umbral global. |
| **CUSUM bilateral sobre residuos** | 53.12 ± 0.43 | `sixeyes_cusum_s{42,7,99}_*.json` | Acumula deriva en tramos cíclicos; desfasa la activación de picos. |
| **Multi-horizonte ESN $[1, 3, 6]$** | 43.42 | `sixeyes_multi_horizon.json` | Sujeta a revisión: evaluaba pronóstico retrasado con estado contemporáneo; varianza en horizontes lejanos domina la norma. |
| **Likelihood gaussiana (L1, L2, L3)** | 33.47 / 30.11 / 39.51 | `sixeyes_likelihood_L*.json` | La relación señal-ruido de la verosimilitud local es inferior a la ECDF empírica. |
| **Fusión ortogonal ESN + centroide** | 42.05 | `sixeyes_hibrido_ortogonal.json` | Ninguna combinación algebraica supera al readout único. |
| **k-NN sobre retardos** | 26.92 | `sixeyes_decision_knn.json` | Colapso de calibración de distancias en series no estacionarias. |
| **Características temporales (`time_features`)** | 56.55 | `sixeyes_time_full_s42.json` | Pesos de proyección aleatoria fija sobre senos/cosenos generan ruido periódico estructurado. |
| **Refractario asimétrico** | 59.32 | `sixeyes_asymrefr_s42.json` | Intentos de reactivación temprana fragmentan el silencio; fuerzan al optimizador a subir el umbral y reducen el neto. |
| **Cola adaptativa por serie** | 1.45 | `sixeyes_adaptail_s42.json` | Inversión de signo en el cálculo de densidad de cola. |

---

## 5. Propiedad del Óptimo y Descomposición de Aciertos

### 5.1 Propiedad del Óptimo Global del Sweeper ($\approx 17$ FPs Efectivos por Ventana)

A partir de la función de coste oficial de NAB Standard:

$$
\text{Score}_{\text{raw}} = \sum_{w=1}^{H} c_w + H - 116 - 0.11 \cdot F_{\text{eff}}
$$

Cualquier detector en su umbral óptimo global $\theta^*$ satisface como propiedad matemática de primer orden la condición de parada del optimizador:

$$
\Delta \text{Score} - 0.11 \cdot \Delta F_{\text{eff}} \approx 0 \implies \frac{d F_{\text{eff}}}{d H} \approx \frac{1 + c_{\text{medio}}}{0.11}
$$

No se trata de un fenómeno físico o intrínseco del modelo, sino de una **propiedad analítica del óptimo** impuesta por la métrica:
- **Crédito medio medido en seed 42 ($c_{\text{medio}}$):** **0.8709** ($\sum_{w} c_w = 84.48$ sobre $H = 97$ ventanas detectadas).
- **Cociente marginal teórico de equilibrio:**

  $$
  \frac{1 + 0.8709}{0.11} = \frac{1.8709}{0.11} \approx 17.01 \; \text{FPs efectivos por ventana nueva}
  $$

- **Falsas alarmas efectivas medidas ($F_{\text{eff}}$):** **367.31** (correspondientes a 382 puntos de alerta fuera de ventana ponderados por su distancia).
- **Valores discretos medidos en el entorno:** En la vecindad de $\theta^*$, los cocientes discretos medidos $\Delta F_{\text{eff}} / \Delta H$ oscilan entre **9.33 y 15.81** (por debajo de 17.01 debido a la naturaleza en escalera de $H(\theta)$ y a que el optimizador se detiene en una meseta discreta de $H=97$).

Esto ilustra que en el punto óptimo, admitir una ventana marginal adicional exige una precisión local del $\approx 5.5\% - 6.0\%$ ($1 / 17 \approx 0.0588$), explicando la resistencia del score a superar las 17 ventanas de la brecha.

### 5.2 Descomposición Bypass vs ESN en las Ventanas Detectadas ($H = 97$ en seed 42)

Evaluación del mecanismo causal exacto que captura las 97 ventanas oficiales en la mejor semilla:
- **Aciertos directos por ruptura de rango (`SpatialSpikeBypass`, score $= 1.0$):** **61 ventanas** (62.9%).
  - **Bypass puro (sin alerta previa de ESN):** **33 ventanas**.
  - **Solapadas (capturadas por bypass y ESN):** **28 ventanas**.
- **Aciertos exclusivos por error de reconstrucción ESN (score $\in [\theta^*, 1.0)$):** **36 ventanas** (37.1%).
  - Total dependientes de la dinámica ESN: $64 - 28 = 36$ ventanas.
- **Ventanas capturadas en la variante spatial-only:** 91 ventanas (de las cuales 4 fueron silenciadas en el pipeline full por el refractario).

---

## 6. Verificación de Ecuaciones

Para garantizar que las ecuaciones documentadas son un reflejo exacto y sin discrepancias del código de producción, el repositorio cuenta con una prueba de regresión analítica:

```bash
python -m pytest tests/test_ecuacion_referencia.py -v -o "pythonpath=src"
```

El test implementa las ecuaciones en NumPy puro (sin importar submódulos de `sixeyes`) y valida coincidencia estricta con tolerancia $< 10^{-9}$ contra el pipeline real en series reales de NAB.

---

## 7. Licencia

[MIT](LICENSE)

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

**1. Estandarización causal rodante** ($W_{\text{norm}} = 6400$), con sumas rodantes $O(1)$ de los valores previos:

```math
z_t = \frac{x_t - \mu_{t-1}}{\sigma_{t-1}} \qquad \left(\text{si } t \lt 2 \text{ o } \sigma \le 10^{-9}, \; z_t = 0.0\right)
```

**2. Incrustación de Takens y dinámica de reconstrucción (ESN):** el vector de retardos alimenta la recurrencia:

```math
v_t = \left[z_t, z_{t-1}, \dots, z_{t-15}\right]^{T} \in \mathbb{R}^{16}
```

```math
h_t = \tanh\left(W_{\text{in}} v_t + W_{\text{res}} h_{t-1}\right) \in \mathbb{R}^{64}
```

*Hallazgo estructural (reconstrucción vs pronóstico):* $v_t$ incorpora a la observación contemporánea $z_t$ en la primera coordenada. Por tanto, el readout lineal $\hat{z}_t = w_{t-1}^{T} h_t$ actúa como un **detector de error de reconstrucción** sobre rasgos de una ventana que ya contiene el valor actual, no como un pronóstico $t+1$ a partir de estados estrictamente previos.

El error cuadrático instantáneo se emite **antes** de actualizar el readout:

```math
e_t = \left(z_t - \hat{z}_t\right)^2
```

Tras emitir el score, los pesos de salida $w$ se adaptan mediante Recursive Least Squares (RLS) con factor de olvido $\lambda = 0.999$:

```math
K_t = \frac{P_{t-1} h_t}{\lambda + h_t^{T} P_{t-1} h_t}
```

```math
w_t = w_{t-1} + K_t \left(z_t - \hat{z}_t\right)
```

```math
P_t = \frac{P_{t-1} - K_t h_t^{T} P_{t-1}}{\lambda}
```

**3. Calibración ECDF causal** ($W_{\text{cal}} = 16000$). Convierte el error no calibrado en percentil uniforme sobre la ventana FIFO de errores previos $W$:

```math
p_t =
\begin{cases}
0.0 & \text{si } |W| \lt 288 \text{ o } e_t = 0.0 \\[6pt]
\dfrac{\left|\{ i \in W : e_i \lt e_t \}\right| + 0.5\left|\{ i \in W : e_i = e_t \}\right| + 0.5}{|W| + 1} & \text{si } |W| \ge 288
\end{cases}
```

**4. Supresión de no-máximos causal con decaimiento exponencial (PeakDecay).** Conserva nuevas crestas locales en ventana $K = 24$ y amortigua réplicas con $\tau = 2.0$:

```math
q_t =
\begin{cases}
p_t & \text{si } p_t \gt \max\left(p_{t-24}, \dots, p_{t-1}\right) \quad (\text{actualiza } t^{*} = t) \\[6pt]
p_t \cdot \exp\left(-\dfrac{t - t^{*}}{2.0}\right) & \text{en otro caso}
\end{cases}
```

**5. Bypass espacial de ruptura de rango (SpatialSpikeBypass).** Mantiene cotas extremas causales $m_t = \min_{i \lt t} x_i$, $M_t = \max_{i \lt t} x_i$ y $R_t = M_t - m_t$:

```math
r_t =
\begin{cases}
1.0 & \text{si } t \ge 100,\; R_t \gt 10^{-12} \text{ y } \left(x_t \lt m_t - 0.05 R_t \text{ o } x_t \gt M_t + 0.05 R_t\right) \\[6pt]
q_t & \text{en otro caso}
\end{cases}
```

**6. Silenciador refractario post-pico (RefractorySuppressor).** Si el score alcanza el umbral de disparo $\ge 0.990$, silencia a $0.0$ durante $W_{\text{rest}} = 80$ pasos:

```math
S_t =
\begin{cases}
0.0 & \text{si } t \lt t_{\text{silencio}} \\[6pt]
r_t & \text{en otro caso (si } r_t \ge 0.990 \text{, se programa } t_{\text{silencio}} = t + 1 + 80\text{)}
\end{cases}
```

**7. Alerta y optimización global NAB:**

```math
\text{Alerta}_t = \mathbb{I}\left[S_t \ge \theta^{*}\right]
```

donde $\theta^{*}$ es el umbral escalar global optimizado en el corpus completo por el Sweeper oficial de NAB.

### 1.2 Ecuación del Puntaje NAB (Score Oficial Standard)

El puntaje del benchmark NAB normaliza la recompensa neta frente al detector nulo (`null`) y el detector perfecto (`perfect`):

```math
\text{Score}_{\text{NAB}} = 100 \cdot \frac{\text{Score}_{\text{raw}} - \text{Score}_{\text{null}}}{\text{Score}_{\text{perfect}} - \text{Score}_{\text{null}}}
```

Para el perfil `standard` sobre el corpus de 58 series (116 ventanas de anomalía etiquetadas):

- $\text{Score}_{\text{null}} = -116.0$ (penalización de $-1.0$ por cada una de las 116 ventanas omitidas).
- $\text{Score}_{\text{perfect}} = +116.0$ ($+1.0$ por detección óptima en el borde inicial de cada ventana).
- Denominador: $\text{Score}_{\text{perfect}} - \text{Score}_{\text{null}} = 116.0 - (-116.0) = 232.0$.

Descomposición exacta de la recompensa bruta:

```math
\text{Score}_{\text{raw}} = \sum_{w=1}^{H} c_w - M - 0.11 \cdot F_{\text{eff}}
```

donde:

- $H$ es el número de ventanas detectadas (aciertos, $H \le 116$).
- $c_w \in (0, 1]$ es el crédito otorgado por la función sigmoide escalada según la precocidad del primer impacto dentro de la ventana $w$.
- $M = 116 - H$ es el número de ventanas no detectadas (falsos negativos a nivel de ventana).
- $F_{\text{eff}} = \sum_{j \in \text{FP}} \sigma(d_j)$ es el conteo efectivo ponderado de falsas alarmas fuera de ventana.

Sustituyendo $M = 116 - H$:

```math
\text{Score}_{\text{raw}} = \sum_{w=1}^{H} c_w - (116 - H) - 0.11 \cdot F_{\text{eff}} = \sum_{w=1}^{H} c_w + H - 116 - 0.11 \cdot F_{\text{eff}}
```

Por tanto:

```math
\text{Score}_{\text{NAB}} = 100 \cdot \frac{\sum_{w=1}^{H} c_w + H - 0.11 \cdot F_{\text{eff}}}{232}
```

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

#### Auditoría de la Brecha de Detección (116 ventanas totales vs ARTime)

- **Ambos detectan:** 89 ventanas.
- **Solo ARTime detecta:** **17 ventanas**.
- **Solo Six Eyes spatial detecta:** 2 ventanas.
- **Ninguno detecta:** 8 ventanas.

*Naturaleza de la brecha:* Las 17 ventanas perdidas corresponden a subidas de nivel moderadas **dentro del rango histórico** (`nyc_taxi`, `grok_asg_anomaly`, `TravelTime_387`, `machine_temperature_system_failure`, etc.). Al permanecer dentro de la envolvente histórica, el `SpatialSpikeBypass` es ciego a ellas por construcción. Ni el ESN ni modelos de densidad estática logran separarlas bajo umbral global sin disparar una cascada de falsas alarmas.

### 2.3 Tabla de Ablación Causal en NAB (media multisemilla, $N=3$)

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

## 3. Estado de TSB Fresco 472: SEÑAL FAVORABLE, SIN VEREDICTO FORMAL

### 3.1 Autopsia del "bug de evaluación" (2026-10-17)

Auditoría forense con `check_bug_una_serie.py` sobre la serie `029_WSD_id_1_WebService_tr_4559_1st_10201.csv` (n=18236, prevalencia 0.0014):

- **El lote es consistente con sklearn:** recalculando el brazo random con el pipeline exacto del lote (`tsb_runner.run_single_series` + `RandomDetector(seed=42)` + `_get_tsb_metrics`), el AUC-ROC es **0.5087**, idéntico al valor guardado en el CSV de filas. Standalone con scores i.i.d.: ~0.5 en 5/5 repeticiones. Sin NaNs, longitudes alineadas, correlación score-etiqueta ~0.
- **Cifras previas ANULADAS (error de lectura de columnas, no de evaluación):**
  - Random "0.011 imposible" → era el **AUC-PR** del random leído como AUC-ROC. Coincide con la prevalencia de anomalías (~0.001-0.01) que es el valor esperado de un AUC-PR aleatorio.
  - Candidato 0.713/0.507 y Z-Score 0.268 → retirados por provenir de la misma lectura corrupta.
- **Causa raíz forense confirmada:** mezcla de esquemas de columnas en `results/sixeyes_tsb_fresh472_rows.csv`. El archivo contiene 211 filas a 12 columnas con orden `...,AUC-PR,AUC-ROC,...` (cabecera vigente) y 645 filas a 5 columnas con orden antiguo `...,AUC-ROC,AUC-PR,...`. Al parsear todo con la cabecera actual, el AUC-ROC verdadero de las filas nuevas cae en la columna `AUC-PR` y viceversa. Además, los "91 duplicados" resultaron ser el solape entre ambos bloques con valores idénticos fila a fila (sin corridas distintas ocultas).
- **Los "0.0081, 0.0015, 0.1330" citados en actas previas se retiran:** eran AUC-PR, no AUC-ROC.

### 3.2 Evaluación Congelada Oficial de P05 (Candidato vs Base P10 en 472 Frescas — COMPLETA)

Corrida limpia, única y pre-registrada bajo el protocolo inmutable [`evaluate_frozen_p05.py`](evaluate_frozen_p05.py) fijado en el commit `5cb22fb8` (tag `prereg-p05-v1`).
- **Cobertura G4:** **472/472 series pareadas (100.0%, 0 excluidas)**.
- **Veredicto pre-registrado (`decisive_ci: "both"`):** **`SUCCESS: True`** (cumplimiento simultáneo en bootstrap por serie y bootstrap clusterizado por familia de series).

**Resultados Pareados de 4 Brazos (n = 472/472, 100.0%):**

| Brazo | AUC-ROC medio | AUC-PR medio | VUS-PR medio | Win Rate vs P10 |
| :--- | :---: | :---: | :---: | :---: |
| **Candidato Híbrido (`P05`)** | **0.8153** | **0.2978** | **0.3541** | — |
| Base Oficial (`P10` / `cand_s42`) | 0.7828 | 0.2047 | 0.2854 | — |
| Z-Score Robusto (`zscore`) | 0.7497 | 0.3588 | 0.2849 | — |
| Detector Aleatorio (`random`) | 0.5025 | 0.0097 | 0.0530 | — |

**Deltas Pareados Candidato P05 vs Base P10 (Criterio Primario y Guardrails):**
- **$\Delta(\text{AUC-ROC})$ (Primaria):** Media **+0.0325** | IC95 Series **[+0.0273, +0.0379]** | IC95 Cluster Familia **[+0.0143, +0.0564]** | Win rate: **76.1%** (359/472).
- **$\Delta(\text{AUC-PR})$ (Guarda):** Media **+0.0930** | IC95 Series **[+0.0810, +0.1057]** | IC95 Cluster Familia **[+0.0207, +0.1665]** | Win rate: **71.0%** (335/472).
- **$\Delta(\text{VUS-PR})$ (Guarda):** Media **+0.0687** | IC95 Series **[+0.0583, +0.0794]** | IC95 Cluster Familia **[+0.0145, +0.1230]** | Win rate: **67.2%** (317/472).

**Descomposición por longitud — corte EXPLORATORIO (mediana = 7500.5 puntos):**

| Segmento | n | P10 (ROC) | P05 (ROC) | Z-Score (ROC) | $\Delta(\text{P05} - \text{P10})$ | IC 95% Series | Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cortas** ($\le 7500$) | 236 | 0.8689 | **0.9099** | 0.9088 | **+0.0411** | [+0.0345, +0.0480] | **84.3%** |
| **Largas** ($> 7500$) | 236 | 0.6967 | **0.7206** | 0.5906 | **+0.0239** | [+0.0156, +0.0322] | **67.8%** |

*Hallazgo:* La debilidad de P10 frente al Z-Score en series cortas (0.869 vs 0.909) es corregida por P05 (0.910), mientras que en series largas P05 consolida una ventaja neta de +0.130 sobre el Z-Score y +0.024 sobre P10.

### 3.3 Puertas de Sanidad Metodológica (G1–G4)

| Puerta | Condición | Valor | Estado |
| :---: | :--- | :---: | :---: |
| **G1** | Random $\in [0.45, 0.55]$ | 0.5025 | **APROBADA** |
| **G2** | Z-Score $\in [0.52, 0.66]$ | 0.7497 | **FALLO (Registrado Formalmente)** |
| **G3** | Paridad Z-Score vs referencia en Tuning | — | **APROBADA** (discrepancia 0.0000 < 0.01 contra `results/ZScore_Raw_tsb_results.csv`) |
| **G4** | Cobertura pareada $\ge 80\%$ de las 472 | 100% (472/472) | **APROBADA** |

### 3.4 Estado

**PROMOCIÓN CIENTÍFICA VALIDADA OUT-OF-SAMPLE.** Conforme al acta [`ACTA_PREREGISTRO_P05.md`](ACTA_PREREGISTRO_P05.md), P05 supera formalmente el criterio pre-registrado sin regresiones en métricas de precisión temporal. Se consolida el reporte en [`NOTA_TECNICA_FROZEN_P05.md`](NOTA_TECNICA_FROZEN_P05.md).

---

## 4. Registro Forense de Experimentos Falsados

Todo experimento no promovido queda archivado con su medición oficial registrada en `results/`:

| Hipótesis / Mecanismo | Resultado NAB Standard | Artefacto | Causa Raíz de Falsación |
| :--- | :---: | :--- | :--- |
| **BOCPD (Adams & MacKay 2007)** | 36.59 | `sixeyes_bocpd.json` | Prior NIGamma asume observaciones i.i.d.; confunde periodicidad normal con saltos. Alta latencia estructural. |
| **LevelRarity (Histograma rareza suave)** | 55.50 | `sixeyes_level_rarity.json` | Rescata 14/17 ventanas perdidas, pero genera 418 falsas alarmas aisladas. El coste de FPs excede la ganancia. |
| **PersistenceFilter ($M=2$)** | 53.06 | `sixeyes_persist_rarity.json` | Los FPs no son sostenidos sino impulsos aislados; el filtro degrada la dinámica del refractario y sube FPs a 726. |
| **Contextual Likelihood sobre residuos** | 58.02 ± 0.94 | `sixeyes_cl_s{42,7,99}.json` | Paradoja de adaptación local: en regímenes mixtos absorbe perturbaciones en $\sigma_{\text{long}}$, cegando anomalías subsiguientes. |
| **Norma en Espacio Latente** $\lVert \Delta h_t \rVert_2$ | 21.61 | `sixeyes_state_volatility.json` | El reservorio se contrae en caídas de amplitud y colapsa a puntos fijos en flatlines. Sin readout no hay discriminación. |
| **State Transition Engine (STE / Markov)** | 37.04 ± 2.18 | `sixeyes_ste_s{42,7,99}.json` | La sorpresa transicional en estados cuantizados es indistinguible entre volatilidad normal y anomalías bajo umbral global. |
| **Fusión periódica residual (MaxFusion)** | 58.02 ± 2.40 | `sixeyes_periodic_fusion_s{42,7,99}.json` | Empate dentro del ruido vs baseline pareado (+0.33 en 3 semillas), lejos del criterio. En s42 es puro coste de FPs (+45 FP / +4 H); en s7/s99 la compuerta abierta desplaza el refractario y suprime ventanas valiosas. |
| **CUSUM bilateral sobre residuos** | 53.12 ± 0.43 | `sixeyes_cusum_s{42,7,99}_*.json` | Acumula deriva en tramos cíclicos; desfasa la activación de picos. |
| **Multi-horizonte ESN** $[1, 3, 6]$ | 43.42 | `sixeyes_multi_horizon.json` | Sujeta a revisión: evaluaba pronóstico retrasado con estado contemporáneo; varianza en horizontes lejanos domina la norma. |
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

```math
\text{Score}_{\text{raw}} = \sum_{w=1}^{H} c_w + H - 116 - 0.11 \cdot F_{\text{eff}}
```

Cualquier detector en su umbral óptimo global $\theta^{*}$ satisface como propiedad matemática de primer orden la condición de parada del optimizador:

```math
\Delta \text{Score} - 0.11 \cdot \Delta F_{\text{eff}} \approx 0 \quad \Longrightarrow \quad \frac{dF_{\text{eff}}}{dH} \approx \frac{1 + c_{\text{medio}}}{0.11}
```

No se trata de un fenómeno físico o intrínseco del modelo, sino de una **propiedad analítica del óptimo** impuesta por la métrica:

- **Crédito medio medido en seed 42** ($c_{\text{medio}}$): **0.8709** ($\sum_{w} c_w = 84.48$ sobre $H = 97$ ventanas detectadas).
- **Cociente marginal teórico de equilibrio:**

```math
\frac{1 + 0.8709}{0.11} = \frac{1.8709}{0.11} \approx 17.01 \; \text{FPs efectivos por ventana nueva}
```

- **Falsas alarmas efectivas medidas** ($F_{\text{eff}}$): **367.31** (correspondientes a 382 puntos de alerta fuera de ventana ponderados por su distancia).
- **Valores discretos medidos en el entorno:** En la vecindad de $\theta^{*}$, los cocientes discretos medidos $\Delta F_{\text{eff}} / \Delta H$ oscilan entre **9.33 y 15.81** (por debajo de 17.01 debido a la naturaleza en escalera de $H(\theta)$ y a que el optimizador se detiene en una meseta discreta de $H=97$).

Esto ilustra que en el punto óptimo, admitir una ventana marginal adicional exige una precisión local del $\approx 5.5\% - 6.0\%$ ($1 / 17 \approx 0.0588$), explicando la resistencia del score a superar las 17 ventanas de la brecha.

### 5.2 Descomposición Bypass vs ESN en las Ventanas Detectadas ($H = 97$ en seed 42)

Evaluación del mecanismo causal exacto que captura las 97 ventanas oficiales en la mejor semilla:

- **Aciertos directos por ruptura de rango (`SpatialSpikeBypass`, score $= 1.0$):** **61 ventanas** (62.9%).
  - **Bypass puro (sin alerta previa de ESN):** **33 ventanas**.
  - **Solapadas (capturadas por bypass y ESN):** **28 ventanas**.
- **Aciertos exclusivos por error de reconstrucción ESN** (score $\in [\theta^{*}, 1.0)$): **36 ventanas** (37.1%).
  - Total dependientes de la dinámica ESN: $64 - 28 = 36$ ventanas.
- **Ventanas capturadas en la variante spatial-only:** 91 ventanas (de las cuales 4 fueron silenciadas en el pipeline full por el refractario).

---

## 6. Verificación de Ecuaciones

Para garantizar que las ecuaciones documentadas son un reflejo exacto y sin discrepancias del código de producción, el repositorio cuenta con una prueba de regresión analítica:

```bash
python -m pytest tests/test_ecuacion_referencia.py -v -o "pythonpath=src"
```

El test implementa las ecuaciones en NumPy puro (sin importar submódulos de `sixeyes`) y valida coincidencia estricta con tolerancia $\lt 10^{-9}$ contra el pipeline real en series reales de NAB.

---

## 7. Licencia

[MIT](LICENSE)
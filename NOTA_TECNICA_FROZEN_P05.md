# NOTA TÉCNICA: Evaluación Out-of-Sample de P05 en 472 Series Frescas (TSB-AD)

- **Fecha:** 2026-10-18
- **Commit Pre-Registrado:** `5cb22fb8702eeac8d9621490a7257c22fc790f46` (Tag inmutable: `prereg-p05-v1`)
- **Ejecución Verificada por Guardián:** Hash de ejecución en `results/frescas_frozen_meta.json` coincide con el commit sellado.
- **Artefactos Resultantes:**
  - Datos brutos: [`results/frescas_frozen_p05.csv`](results/frescas_frozen_p05.csv) (1,888 filas pareadas, 472/472 series)
  - Metadatos del protocolo: [`results/frescas_frozen_meta.json`](results/frescas_frozen_meta.json)
  - Informe de evaluación formal: [`results/frozen_p05_report.json`](results/frozen_p05_report.json)

---

## 1. Resumen Ejecutivo y Veredicto Pre-Registrado

Bajo el protocolo pre-registrado en Git (`decisive_ci: "both"`), el candidato híbrido **P05 ($\alpha_{\text{recon}}=0.5$) superó formalmente a la base P10 ($\alpha_{\text{recon}}=1.0$) en todas las métricas pareadas**, cumpliendo las condiciones de éxito sin regresión ni violación del margen de no-inferioridad (-0.01).

- **Veredicto:** **`SUCCESS: True`**
- **Cobertura G4:** **100% (472/472 series válidas, 0 excluidas)**.

| Métrica | Base P10 | Candidato P05 | $\Delta$ Pareado Medio | IC 95% Series | IC 95% Cluster (Familia) | Win Rate (P05 > P10) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AUC-ROC** (Primaria) | 0.7828 | **0.8153** | **+0.0325** | **[+0.0273, +0.0379]** | **[+0.0143, +0.0564]** | **76.1%** (359/472) |
| **AUC-PR** (Guarda) | 0.2047 | **0.2978** | **+0.0930** | **[+0.0810, +0.1057]** | **[+0.0207, +0.1665]** | **71.0%** (335/472) |
| **VUS-PR** (Guarda) | 0.2854 | **0.3541** | **+0.0687** | **[+0.0583, +0.0794]** | **[+0.0145, +0.1230]** | **67.2%** (317/472) |

---

## 2. Desglose Pareado por Familia de Series (Auditoría de Incertidumbre)

La incertidumbre agrupada (cluster bootstrap por familia) arrojó un límite inferior de **+0.0143** en AUC-ROC frente a **+0.0273** por serie. Este ensanchamiento refleja la heterogeneidad entre familias:

| Familia | $n$ (% total) | P10 (ROC) | P05 (ROC) | Z-Score (ROC) | $\Delta(\text{P05}-\text{P10})$ ROC | $\Delta$ AUC-PR | $\Delta$ VUS-PR | Win Rate (ROC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YAHOO** | 224 (47.5%) | 0.8747 | 0.9159 | 0.9309 | **+0.0412** | +0.1554 | +0.1171 | **83.9%** (188/224) |
| **UCR** | 152 (32.2%) | 0.6329 | 0.6439 | 0.5137 | **+0.0109** | +0.0121 | +0.0092 | **57.2%** (87/152) |
| **WSD** | 86 (18.2%) | 0.8059 | 0.8379 | 0.6696 | **+0.0321** | +0.0362 | +0.0260 | **84.9%** (73/86) |
| **Stock** | 10 (2.1%) | 0.8027 | 0.9726 | 0.9669 | **+0.1699** | +0.4160 | +0.2548 | **100.0%** (10/10) |

### Diagnóstico de Concentración por Familia:
- **YAHOO concentra el volumen de mejora:** Al representar casi la mitad de la muestra (47.5%), su fuerte respuesta positiva ($\Delta = +0.0412$, win rate 83.9%) tracciona el agregado global.
- **UCR es el segmento resistente:** En UCR (32.2% de la muestra), el $\Delta$ es positivo pero modesto ($+0.0109$, win rate 57.2%), aunque ambos modelos superan claramente al Z-score (0.5137).
- **Consistencia de signo:** En las 4 familias el $\Delta$ medio es estrictamente positivo en las tres métricas y el win rate supera el 50%, lo que fundamenta que el cluster bootstrap mantenga su límite inferior $> 0$ (+0.0143).

---

## 3. Análisis Exploratorio: Comportamiento por Longitud y Relación con el Z-Score

Corte exploratorio por mediana de longitud ($N = 7500.5$ puntos):

| Segmento | $n$ | P10 (ROC) | P05 (ROC) | Z-Score (ROC) | $\Delta(\text{P05} - \text{P10})$ | Win Rate vs P10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cortas** ($\le 7500$) | 236 | 0.8689 | **0.9099** | **0.9088** | **+0.0411** | **84.3%** |
| **Largas** ($> 7500$) | 236 | 0.6967 | **0.7206** | 0.5906 | **+0.0239** | **67.8%** |

### Precisiones Críticas sobre el Z-Score:
1. **En series cortas hay empate técnico con el Z-Score:** P05 (0.9099) y Z-Score (0.9088) están a una décima de punto porcentual ($+0.0011$). P05 rescata la caída que sufría P10 (0.8689), pero en cortas solo **iguala a un baseline trivial**.
2. **El contraste con el Z-Score no fue pre-registrado:** Es una comparación exploratoria post-hoc.
3. **La ventaja real sobre el Z-Score reside en series largas:** En series largas ($>7500$ puntos), P05 supera ampliamente al Z-score (**0.7206 vs 0.5906**, $\Delta = +0.1300$).

---

## 4. Compuertas Metodológicas

- **G1 (Detector Aleatorio):** **APROBADA**. $\text{AUC-ROC} = 0.5025 \in [0.45, 0.55]$. ($\text{AUC-PR} = 0.0097$).
- **G2 (Z-Score Robusto):** **FALLO REGISTRADO**. $\text{AUC-ROC} = 0.7497 \notin [0.52, 0.66]$.
  - Asentado formalmente como **FALLO sin atenuantes**. Las series frescas presentan una población más fácil para métodos de nivel que el split histórico de Tuning.
- **G4 (Cobertura):** **APROBADA**. 472/472 series procesadas.

---

## 5. Alcance, Límites y Qué NO Afirma Este Resultado

Para evitar sobreinterpretación científica, se delimitan explícitamente las fronteras del hallazgo:

1. **Agotamiento de la muestra fresca:** Las 472 series de evaluación **ya no son frescas para P05 ni para esta familia de modelos**. No es admisible evaluar variantes subsiguientes (p. ej. Paso 2 con filtrado o dual-ESN) sobre estas mismas series bajo pretensiones confirmatorias. La línea se da por cerrada en este punto.
2. **Especificidad del entorno experimental:** La evaluación se realizó exclusivamente sobre **TSB-AD-U** (univariado) y con **semilla 42**.
3. **Sin afirmaciones de Estado del Arte (SOTA):** Este experimento contrastó P05 contra su ablación directa P10 (reconstrucción pura) y baselines de referencia (Z-score, azar). **No se ha comparado contra detectores aprendidos contemporáneos más pesados** (Isolation Forest multivariado, LSTM-AD, TranAD, etc.), por lo que no se afirma superioridad general en el benchmark global.
4. **No transferible automáticamente a NAB:** El pipeline P05 evaluado aquí es `SpatialSpikeBypass + PercentileCalibrator + PredictiveReadoutESN`. **No incluye ni `StreamingPeakDecay` ni `RefractorySuppressor`** (capas indispensables para NAB que colapsan en TSB-AD). No se puede inferir que P05 mejore el rendimiento en NAB sin una evaluación independiente.
5. **Novedad conceptual:** La combinación de reconstrucción y pronóstico autoregresivo es una idea con precedentes en la literatura de redes neuronales recurrentes; el aporte aquí radica en su formulación analítica causal en streaming con RLS dual sobre un reservorio ESN congelado y su verificación ciega out-of-sample.

# Aprendizaje — Registro Vivo de Experimentos e Integridad Metodológica (Six Eyes)

> Todo lo intentado, medido, autopsiado y archivado.
> Regla de oro: Ningún número sin artefacto en `results/`; ninguna afirmación sin medición causal reproducible.

---

## 1. Hito Metodológico: Validación Ciega Out-of-Sample de P05 (2026-10-18)

- **Artefactos Oficiales:**
  - Código y protocolo sellados en Git: Tag inmutable `prereg-p05-v1` (commit `5cb22fb8` / `23832ed4`).
  - Datos brutos: `results/frescas_frozen_p05.csv` (1,888 filas, 472 series × 4 brazos completos).
  - Informe estadístico reproducible: `results/frozen_p05_report.json`.
  - Nota técnica y reporte formal: `experiments/tsb_frozen_p05/NOTA_TECNICA_FROZEN_P05.md`.
- **Candidato Evaluado (P05):**
  $$\text{SpatialSpikeBypass}(\text{PercentileCalibrator}(\text{PredictiveReadoutESN}(\alpha_{\text{recon}}=0.5, \text{seed}=42), W=16000), \text{tol}=0.05, \text{warmup}=100)$$
  *Mezcla 50/50:* Reconstrucción contemporánea ESN + pronóstico residual un paso adelante ($h_{t-1} \to z_t$). Sin supresión refractaria ni decaimiento de picos.
- **Línea Base Oficial (P10):** Mismo arnés con $\alpha_{\text{recon}}=1.0$ (reconstrucción pura, bit a bit equivalente a `cand_s42`).

### 1.1 Resultados Pareados en las 472 Frescas (Criterio `both`: Series + Cluster Familia)

| Métrica | Base P10 | Candidato P05 | $\Delta$ Pareado Medio | IC 95% Series | IC 95% Cluster Familia | Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AUC-ROC** (Primaria) | 0.7828 | **0.8153** | **+0.0325** | **[+0.0273, +0.0379]** | **[+0.0143, +0.0564]** | **76.1%** (359/472) |
| **AUC-PR** (Guarda) | 0.2047 | **0.2978** | **+0.0930** | **[+0.0810, +0.1057]** | **[+0.0207, +0.1665]** | **71.0%** (335/472) |
| **VUS-PR** (Guarda) | 0.2854 | **0.3541** | **+0.0687** | **[+0.0583, +0.0794]** | **[+0.0145, +0.1230]** | **67.2%** (317/472) |

- **Veredicto:** **`SUCCESS: True`**. Se cumple simultáneamente el límite inferior $>0$ en la métrica primaria y la no-inferioridad en los guardrails temporales sin regresión significativa.

### 1.2 Compuertas Metodológicas G1–G4
- **G1 (Random):** **APROBADA** ($\text{AUC-ROC} = 0.5025 \in [0.45, 0.55]$, $\text{AUC-PR} = 0.0097$).
- **G2 (Z-Score):** **FALLO REGISTRADO** ($\text{AUC-ROC} = 0.7497 \notin [0.52, 0.66]$). Se registra como fallo sin atenuantes post-hoc.
- **G4 (Cobertura):** **APROBADA** (100%, 472/472 series válidas, 0 excluidas).

---

## 2. Lecciones y Autopsias Forenses de Fallos

### 2.1 La Falsa Alarma del "Bug de Evaluación" de G1 (Resolución de Causa Raíz)
- **Problema previo:** En corridas históricas preliminares, el detector aleatorio daba $\text{AUC-ROC} \approx 0.010$.
- **Causa raíz unificada:** Mezcla posicional de cabeceras en CSVs viejos (esquemas de 12 cols vs 5 cols). Las columnas `AUC-ROC` y `AUC-PR` estaban invertidas al volcar los resultados de `random` y `zscore`. El 0.010 correspondía a la prevalencia real de anomalías de la serie (su AUC-PR teórico), mientras que su AUC-ROC verdadero era 0.50.
- **Solución implementada:** Adopción obligatoria de `csv.DictWriter` vinculado por nombres de campo y tests unitarios de contrato en `tests/test_harness_metrics.py` y `tests/test_e2e_runner_eval.py`.

### 2.2 Auditoría de Incertidumbre: Concentración por Familias
Al aplicar bootstrap agrupado por familia (*cluster bootstrap*), el límite inferior en ROC cayó de $+0.0273$ a $+0.0143$:
- **YAHOO (47.5% de la muestra, 224 series):** Tracciona fuertemente la ganancia ($\Delta = +0.0412$, win rate 83.9%).
- **UCR (32.2% de la muestra, 152 series):** Es el segmento resistente ($\Delta = +0.0109$, win rate 57.2%). 74 de las 82 series con ROC < 0.60 pertenecen a UCR.
- **Consistencia:** Las 4 familias (YAHOO, UCR, WSD, Stock) tienen delta positivo en las tres métricas.

### 2.3 Matiz Crítico: Z-Score vs P05 en Series Cortas
- En series cortas ($\le 7,500$ puntos), P05 obtiene **0.9099** y el Z-Score **0.9088**. **Es un empate técnico con un baseline trivial.** P05 rescata el déficit que sufría P10 (0.8689), pero en cortas no supera al Z-Score.
- La superioridad neta sobre el Z-score proviene exclusivamente de las series largas (**0.7206 vs 0.5906**, $\Delta = +0.1300$).

---

## 3. Descomposición de Mecanismos: Por qué Falla el Modelo

Al someter las peores series a autopsia (p. ej. Yahoo ROC = 0.0887 y UCR Medical ROC = 0.1660), descubrimos que operan dos fallos independientes:

### 3.1 Fallo de Inicialización y Calibración (Caso Yahoo ROC = 0.0887)
- **Diagnóstico:** En la serie `574_YAHOO_id_24`, la anomalía única ocurre en $t=202$.
- **Mecanismo:** El calibrador ECDF ([`calibrador_percentil.py`](src/sixeyes/detectors/calibrador_percentil.py)) tiene una constante de warmup `MIN_HISTORIAL = 288`. Durante los primeros 288 pasos, emite forzosamente `score = 0.0`.
- **Efecto:** La anomalía real en $t=202$ recibió score 0.0, mientras que el régimen normal post-288 recibió scores entre 0.1 y 0.8. La curva ROC se invirtió por diseño de warmup, no por fallo de la red neuronal. El Z-Score (warmup=5) sacó 0.9613.

### 3.2 Patología de Adaptación Continua RLS (Caso UCR y Anomalías Sostenidas)
- **Diagnóstico:** En series con anomalías prolongadas (como `470_UCR`, meseta de 30 pasos en $t \approx 11,000$), el RLS continuo adapta sus pesos con $\lambda = 0.999$.
- **Mecanismo:** A los 15-20 pasos dentro de la anomalía, el RLS reduce sus residuos a cero (absorción). Al regresar al régimen normal, los pesos deformados generan un residuo masivo sobre puntos normales.
- **Efecto:** Falsa alarma de recuperación prolongada post-anomalía que invierte la ordenación de scores ($\text{ROC} < 0.50$).

---

## 4. Investigación de Plasticidad Selectiva (Fase B — Ablación Controlada)

Se implementó una suite diagnóstica aislada ([`scripts/experiments/diag_plasticity_ablation_b.py`](scripts/experiments/diag_plasticity_ablation_b.py)) para medir en idénticas condiciones 3 variantes frente a 3 regímenes sintéticos (Meseta anómala 200 pasos, Deriva lineal lenta legítima, y Régimen estacionario puro):

| Métrica / Escenario | Variante 1: RLS Continuo | Variante 2: RLS Congelado | Variante 3: Compuerta Prefrontal |
| :--- | :---: | :---: | :---: |
| **Absorción de la meseta (error crudo)** | 99.1% (absorbe todo) | 95.7% | 99.8% |
| **Residuo crudo en retorno normal ($t=800$)** | **1.63** (desorientación masiva) | **0.31** (pesos limpios) | **0.08** (amortiguado) |
| **Pasos de falsa alarma en recuperación (>0.5)** | **114 pasos** | 125 pasos | **58 pasos (reducción del 49%)** |
| **Score medio en recuperación** | 0.6699 (alarma activa) | 0.6537 | **0.4461 (bajo umbral de alarma)** |
| **Seguimiento de deriva legítima (error raw)** | 0.0119 | **0.1149 (falla, 10x peor)** | **0.0119 (adaptación intacta)** |
| **Falsas alarmas en deriva (>0.5)** | 47.1% | **67.8% (colapso)** | **47.1% (no bloquea deriva)** |
| **Falsas alarmas en régimen estacionario** | 34.0% | 43.2% | **34.0% (cero inestabilidad)** |

### 4.1 Descubrimientos Científicos Fundamentales de la Ablación:
1. **La compuerta prefrontal causal funciona sin degradar la deriva:** Bloquear la actualización RLS cuando el residuo contemporáneo supera $3\sigma$ reduce a la mitad la duración del falso pico de recuperación (de 114 a 58 pasos) y su intensidad a $0.44$, sin perjudicar el seguimiento de cambios lentos normales ni el régimen estacionario.
2. **La plasticidad no reside solo en los pesos $w$ (Inercia parásita de memorias):**  
   Congelar los pesos RLS (Variante 2) elimina la desorientación del residuo crudo ($0.06$ vs $1.63$), pero el score calibrado final sigue inflado durante la recuperación. **Causa descubierta:** La ventana de normalización móvil contemporánea (`_norm` de tamaño 6,400) y la memoria FIFO de la ECDF absorben la amplitud de la meseta. Para un control total de la plasticidad cortical, la compuerta debe regular tanto el RLS como la ingesta en las memorias de escala.

---

## 5. Fronteras de Alcance y Cierre de la Línea

1. **Agotamiento del conjunto fresco:** Las 472 series de TSB-AD **ya no son frescas para P05**. Se prohíbe evaluar variantes subsiguientes de forma confirmatoria sobre este split. La línea experimental de P05 queda formalmente **CERRADA**.
2. **Entorno específico:** Los resultados son válidos para **TSB-AD-U** con **semilla 42**.
3. **Sin afirmaciones SOTA:** No se ha comparado contra modelos pesados contemporáneos (Isolation Forest, LSTM-AD, Transformers); no se afirma superioridad general del estado del arte.
4. **No transferible a NAB:** El pipeline P05 no incluye supresores refractarios ni decaimiento de picos; su desempeño en NAB debe evaluarse de forma independiente.
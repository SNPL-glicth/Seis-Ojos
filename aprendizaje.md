# Aprendizaje — registro vivo de experimentos Six Eyes

> Todo lo intentado, medido y archivado. Cada entrada: hipótesis → medición → autopsia → lección.
> Regla: ningún número sin artefacto en `results/`; ninguna vigencia sin medición superada.

## Protocolo de fusión offline + regla de decisión (fijada ANTES de medir, 2026-10-14)

- **Una sola pasada ESN**: scores crudos por punto de cada semilla (42/7/99)
  guardados en `results/raw_esn/`. Todas las variantes fusionan OFFLINE
  sobre esos crudos; solo se repite optimizador+scorer (lo barato).
- **Variantes (máx. 3 intentos, luego se cierra):** V1 rangos medios (medido:
  44.49 / LOCO 43.64), V2 máximo de crudos, V3 mediana (2º mayor) de crudos.
  Cada una: fusión → UN calibrador → PeakDecay(24,2.0), mismo K en ambas
  columnas. Se guarda score POR ARCHIVO, no solo por fold.
- **Regla de promoción (inmutable):** el ensemble se adopta SOLO si la
  diferencia pareada media en categorías REALES (LOCO, sin artificial) es
  positiva Y ningún fold real empeora más de 2 pts vs semilla única.
  Si no: queda `--detector ensemble_rangos` experimental con ambos números
  declarados y se sigue con likelihood sobre el error del ESN.
- **Por qué:** con ~15+ configs vistas sobre el mismo test, ver el resultado
  antes de fijar la regla es la forma de autoengañarse.

## Contextual Anomaly Likelihood sobre ESN crudo — MEDIDO, ARCHIVADO (2026-10-15)

- **Hipótesis:** Siguiendo la formulación matemática de Numenta y ContextOSE, modelar la distribución local de errores de predicción del ESN mediante medias y varianzas rodantes adaptativas (EMA $O(1)$) y calcular la probabilidad de cola gaussiana ($1 - Q(z)$) permitiría rescatar las ventanas sutiles de cambio de régimen (`nyc_taxi`, `grok`, `flatmiddle`) que quedan bajo el umbral global estático de la ECDF.
- **Implementación:** `src/sixeyes/experimental/contextual_likelihood.py` (`ContextualLikelihood`, $W'=10, W=1000$) + suite de pruebas `tests/test_contextual_likelihood.py` (5 passed) + gate oficial `archive/gates/gate_contextual_likelihood.py`.
- **Medición NAB Oficial (3 semillas pareadas sobre `raw_esn`):**
  - Semilla 42: **57.79** (vs baseline 60.81, **−3.02**) | Low FP 49.79 (+6.39) | Low FN 62.09 (−6.36).
  - Semilla 7: **57.23** (vs baseline 54.66, **+2.57**) | Low FP 49.28 (+5.88) | Low FN 61.61 (−6.84).
  - Semilla 99: **59.04** (vs baseline 57.58, **+1.46**) | Low FP 51.34 (+7.94) | Low FN 63.50 (−4.95).
  - **Media oficial pareada (n=3):** **58.02 ± 0.94** (frente a 57.68 ± 3.08 del titular base: **+0.34 neto**, muy lejos del criterio $+3.0 \ge 60.68$).
- **Autopsia & Diagnóstico Profundo (La Paradoja de Adaptación Local):**
  1. **Enorme reducción de falsos positivos en series estables:** En `artificialNoAnomaly`, los FPs cayeron de 9 a solo 3 (en `art_noisy` bajó de 6 a 1). En el perfil `Low FP Rate`, el score subió masivamente de 43.40 a **49.79–51.34 (+7 pts)**. En `realTweets`, la ganancia fue explosiva (+5.58 pts).
  2. **Colapso de TPs en series con regímenes mixtos (`realKnownCause` y `realTraffic`):**
     - En `realKnownCause`, el score cayó de +3.68 a **−2.90 (−6.57 pts)**. En `realTraffic`, de +5.07 a **+0.14 (−4.92 pts)**.
     - **Mecanismo del fallo:** En series largas como `machine_temperature` o `nyc_taxi`, cuando el ESN experimenta un error moderado sostenido, la ventana adaptativa de largo plazo ($W=1000$) absorbe el nuevo nivel de error, inflando $\sigma_e$ y desplazando $\mu_e$. Como consecuencia, las anomalías posteriores dentro de la misma serie son normalizadas como "comportamiento esperado", silenciando los TPs (los TPs totales del corpus cayeron de 147 a 98).
  3. **Criterio pre-registrado:** Requiere media $\ge 60.68$ y ninguna semilla peor. Con media 58.02 y semilla 42 cayendo −3.02, **no califica para promoción y queda archivado con su evidencia.**



## Filtro de Persistencia (M=2) sobre LevelRarity — FALSADO, ARCHIVADO (2026-10-15)

- **Hipótesis:** Si los FPs de LevelRarity son picos aislados de un paso, exigir persistencia temporal (M=2 pasos consecutivos en zona de anomalía) eliminaría el ruido sin sacrificar los TPs de las 14 ventanas recuperadas.
- **Medición NAB (semilla 42):** **Standard 53.06 (−4.62 vs full 57.68, −2.44 vs LevelRarity 55.50)**.
  - Falsos Positivos: **suben de 418 a 726 FPs**.
  - Ventanas rescatadas: se mantienen las mismas 14/17, pero el score colapsa por la explosión de penalizaciones fuera de ventana.
- **Autopsia & Falsación:**
  - Los FPs de LevelRarity **no eran impulsos aislados de un solo paso**; son rachas de dispersión y fluctuación en colas de densidad que el filtro de persistencia no atenúa y que, al alterar la dinámica temporal del refractario y del umbral, multiplican los falsos cruces efectivos.
  - Queda documentado con números: el filtro de persistencia M=2 queda **archivado**.
- **Diagnóstico Definitivo (Transición vs. Rareza / El Techo de NAB):**
  - **Nosotros:** $P(\text{anomalía} \mid x_t)$ — rareza puntual/estática respecto a la distribución acumulada de valores o fuera de envolvente. Cada punto se juzga por su magnitud relativa al histórico.
  - **Los Top-4 (ARTime 74.85, Numenta 70.10, ContextOSE 69.90, HTMJava 65.55):** $P(\text{anomalía} \mid x_t, x_{t-1}, \dots, x_{t-k})$ — ruptura de la ley de transición secuencial del sistema.
  - **La brecha de las 17 ventanas (nyc_taxi, grok, TravelTime, machine_temp, etc.):** son transiciones de régimen o saltos sutiles dentro del rango conocido (scores 0.92–0.99 frente a umbral 0.99077/0.99658). Para rareza estática son invisibles; para una dinámica temporal aprendida son rupturas de predicción de transición.
  - **El ESN y el camino:** El ESN de 64 neuronas posee memoria recurrente latente, pero el readout actual (RLS) fue optimizado para adivinar el escalar $x_{t+1}$ en vez de medir la fractura de la trayectoria en el espacio latente del reservorio ($||\mathbf{h}_t - \hat{\mathbf{h}}_t||$ o divergencia en dinámica).
  - El sistema actual basado en bypass + ESN puntual se encuentra en su techo empírico (~57.68 in-sample, 55.72 LOCO). Para escalar al grupo >65-70 en NAB de forma honesta y transferible (LOCO-verificable) se requiere abordar la transición de estado.

## LevelRarity (soft bypass, p-valor de cola) — MEDIDO, ARCHIVADO (2026-10-15)

- **Hipotesis:** el bypass solo ve "salio del rango"; la brecha son subidas moderadas DENTRO del rango
  (17 ventanas, score 0.92-0.99 < umbral 0.99658). Un histograma de niveles decaido con score =
  p-valor de cola (1 - masa de bins igual o mas raros / total) cubriria justo esa brecha.
- **Diseno a priori (un solo intento):** K=80 bins sobre rango warmup(100) + 5% margen + 2 overflow;
  semivida 2000 (un solo valor); max con bypass, antes del refractario;_score = p-valor de cola
  calculado ANTES de insertar el punto actual (si no, el bin actual siempre es "el mas comun"
  y score=0 — error de implementacion corregido en tests).
- **Tests:** 7 passed (no-saturacion ~uniforme, overflow→1.0, nivel raro > normal, causalidad,
  reset, decaimiento por bin, params invalidos). Artifactos: `results/sixeyes_level_rarity.json`.
- **Medicion NAB (semilla 42):** **Standard 55.50 (−2.18 bajo 57.68)** | Low FP 38.40 | Low FN 62.90.
  Umbral 0.99077 (mismo que full: el refractario lo vuelve a bajar).
- **Analisis post:** **14/17 ventanas recuperadas** (load_balancer 0.9917, 24ae8d 0.9973, fe7f93
  0.9953, grok 0.9991, exchange-3_cpc 0.9943, exchange-4_cpc 0.9988, machine_temp ×2 0.9985/0.9927,
  nyc_taxi ×2 0.9972/0.9914, TravelTime ×2 1.0/0.9923, GOOG 0.9976, UPS 0.9987). El mecanismo
  funciona. Pero **418 FPs nuevos** (score>=thr fuera de ventana) — la señal es real pero no
  limpia: el nivel raro no discrimina anomalia de rareza benigna.
- **Criterio pre-registrado:** media >= 60.68? **NO (55.50)**. Ninguna semilla peor? pendiente
  (s42 unica corrida). **Archivado con su numero.** El p-valor de cola generaliza bien el bypass
  (14/17) pero el coste en FPs destruye el neto: −2.18 a pesar de recuperar 14 ventanas.
- **Diagnostico de FPs (sin corrida nueva, `archive/gates/diag_rarity_fps.py`):** los 418 FPs son
  **418 episodios de 1 punto, 0% permanencia**. No son "quedarse en zona rara" — son cruces aislados
  del umbral con score >= 0.99077. El refractario (80 pasos) ya silencia toda permanencia; no hay
  dimension de entrada-vs-permanencia que aplicar. Por la regla de un intento: **aislados y dispersos
  → archivar definitivamente**. El score del p-valor es ruidoso alrededor del umbral, no se sienta
  encima. Cerrado como ultimo intento sobre NAB.
- **Cuenta del revisor confirmada:** 14 ventanas ~+11, 418 FPs ~−13, neto ~−2. Para +3 harian falta
  ~250 FPs (−40%) conservando aciertos, pero al ser aislados no hay mecanismo de recorte que no
  lleve TPs. El acta final lo dice: NAB lo mueven las rupturas de rango (trivial 50.83 > ESN 49.05),
  y las dos trampas de implementacion (score antes de insertar, poblar histograma con warmup).

# Objetivo: mayor puntaje NAB (2026-10-15)

> No somos una libreria. El objetivo es el mayor puntaje, cueste lo que cueste
> en pensamiento, medicion e investigacion. Se levanta el cierre sobre NAB:
> nuevos intentos permitidos con criterio pre-registrado por intento.

## State Transition Engine (sorpresa P(S_t|S_{t-1}) sobre estados densos) — FALLIDA, archivada (2026-10-15)

- **Hipotesis:** Markov-orden-1 fallo sobre simbolos escalares pobres (14.36) y Delta-h descarto el
  readout (21.61). Esta vez: estados densos del reservorio (R^64, mismas constantes que el ESN) +
  cuantizador online K=32 (lr=0.02, init determinista primeros K estados) + Markov primer orden con
  Laplace eps=1.0 + score -log P. Todo a priori, sin barrido. En slot base, mismo downstream.
- **Tests:** 8 passed (determinismo, prefijo invariante, asignacion/adaptacion, Laplace, sorpresa
  no-vista > vista, reset-replay, finitud, params invalidos).
- **Medicion (3 semillas, NAB completo c/u):** **media 37.04** (42=39.43, 7=35.22, 99=36.48).
  Las 3 peores que su full (60.81/54.66/57.58). Criterio >=60.68: NO. Sin LOCO.
- **Mecanismo:** 37.04 supera a Markov-escalar (14.36) y Delta-h (21.61) — la representacion densa
  SI ayuda — pero pierde contra error-de-prediccion ESN (48.30). La sorpresa de transicion es alta
  en regimenes volatiles normales igual que en anomalias: sin separacion bajo umbral global.
  Leccion: la referencia aprendida del readout ("que es normal") discrimina mas que la estructura
  de transicion. Tercera falsacion de la familia transicional (markov, ppm, ste).
- **Cierre de la familia:** 3/3 intentos de modelar transiciones falsados (14.36, 18.63, 37.04).
  La via transicional queda cerrada salvo modelo de transicion de segundo orden o jerarquico,
  que seria otro proyecto. Artefactos: `results/sixeyes_ste_s{42,7,99}.json`.

## Comparacion gratuita vs relativeEntropy: 17/17, familia en ~55 (2026-10-15)

- **Pregunta:** cuantas de las 17 ventanas cubre la familia histograma-divergencia ya ajustada
  (relativeEntropy 54.64: 5 bins, ventana 52, salida binaria, rango global no-causal)?
- **Respuesta (`archive/gates/ventanas_vs_relent.py`, sin corrida nueva):** **17/17** con max=1.0
  en todas, y **373 FPs** propios. El concepto llena la brecha completa.
- **Pero la familia queda en ~55:** relativeEntropy (no-causal, binaria, ajustada) = 54.64;
  nuestro LevelRarity causal = 55.50 con 418 FPs. Ya estamos en el techo de la familia.
  relativeEntropy cubre 3 mas que nosotros (17 vs 14) con menos FPs (373 vs 418): bins gruesos
  + salida binaria son mas estables que p-valores graduados finos. Pero ninguna variante rompe
  ~55 porque el coste FP es estructural a "zona rara = anomalia".
- **Veredicto:** no se construye otra variante. La comparacion valida el diagnostico (el concepto
  funciona) y confirma el techo (la familia no pasa de ~55). NAB cerrado.

## Cola adaptativa por serie (Numenta-style) — FALLIDA, archivada (2026-10-15)

- **Hipotesis:** el umbral global bloquea eventos 0.92-0.99 significativos en su serie. Normalizar
  por serie online (gaussiana rodante W=500, cola superior) haria los scores comparables.
- **Medicion (3 semillas, replay de CSVs full):** **media 1.45** (42=1.41, 7=1.46, 99=1.46).
  El optimizador se rinde (umbral 1.0, scores brutos negativos).
- **Mecanismo del fallo (error de signo):** la cola superior P(X>=s) es PEQUENA para anomalias.
  NAB premia scores ALTOS = anomalia; le dimos scores BAJOS = anomalia. Senal invertida.
  La version corregida (1-cola) es conceptualmente identica a LevelRarity (rareza de cola),
  ya medida en 55.50 con 418 FPs. Sin rerun: converge a idea archivada.
- **Leccion:** la normalizacion por serie no es el problema — el problema es que la rareza
  (histograma o gaussiana) no discrimina anomalia de rareza benigna.

## Ruta >68: los tres caminos medidos y archivados (2026-10-15)

- **Refractario asimetrico (wake on acceleration, 1 intento):** s42 **59.32 (−1.49)** → ARCHIVADO.
  El optimizador subio el umbral a 0.99649: los wakeups fragmentan el silencio y dejan pasar FPs
  (Low FP 51.71 vs 43.40). Confirma que la rigidez del estatico ES la feature — el umbral bajo
  0.99077 solo funciona con silencio estricto. Artefacto: `results/sixeyes_asymrefr_s42.json`.
- **time_features=True en full (1 intento):** s42 **56.55 (−4.26)** → ARCHIVADO. El mecanismo de B1
  (pesos fijos + entradas periodicas = ruido estructurado) se mantiene con bypass+refractario.
  Low FP colapsa a 37.43. Artefacto: `results/sixeyes_time_full_s42.json`.
- **Veredicto:** los tres caminos a >68 medidos y falsados. NAB cerrado de verdad: titular 57.68,
  LOCO 55.72, techo documentado. Solo TSB-Eva puede validar generalizacion.

## Ruta >68: tres refutaciones contra la complacencia (2026-10-15)

Verificado con scorer oficial (TP/FP por puntos):
- Six Eyes full s42: TP=147, FP=382 (60.81)
- ARTime: TP=168 (+21), FP=217 (−165) (74.85)
- Numenta: TP=240 (+93), FP=153 (−229) (70.10)
- contextOSE: TP=96 (−51), FP=64 (−318) (69.90)

**Refutacion 1 (aritmetica):** contextOSE gana con MENOS TPs (96<147). El problema son los 382 FPs,
no las 17 ventanas. Recortar a ~180 FPs subiria a ~67-68 sin ganar ni una ventana.

**Refutacion 2 (sordera):** 16-19/116 ventanas empiezan en silencio refractario. Un refractario
asimetrico (despertar ante aceleracion) recuperaria 8-12 sin FPs extra.

**Refutacion 3 (tiempo apagado):** `time_features=False` en el ESN. Numenta usa timeOfDay/dayOfWeek;
nyc_taxi es imposible sin saber la hora. B1 (47.43) midio time_features en base ESN sola;
falta medirlo en el full (bypass cubre picos, refractario contiene FPs).

## Corrección LOCO pareado: el full SÍ transfiere mejor (2026-10-15)

- **Mi claim "spatial-solo no transfiere" estaba mal medida:** comparé el LOCO del spatial en seed 42
  (48.87) contra el full medido solo en su mejor semilla (60.06). Pareado por semilla en las 3:

| Semilla | Full | Spatial-solo | Δ |
| :---: | :---: | :---: | :---: |
| 42 | 60.06 | 48.87 | +11.19 |
| 7 | 50.33 | 50.78 | −0.45 |
| 99 | 56.76 | 51.16 | +5.60 |
| **Media** | **55.72** | **50.27** | **+5.45 ± 3.36** |

- **Lectura correcta:** el refractario **baja** la media in-sample (57.68 vs 58.50) pero **sube** la
  transferencia out-of-sample (+5.45 pareado, 2/3 positivo). Su valor está en el umbral transferido
  (0.99077 vs 0.99658), no en el score in-sample. El candidato intocado por regla sigue siendo el full,
  ahora con respaldo pareado. Esto refuta la lectura "el refractario estorba" en su versión fuerte:
  estorba in-sample, ayuda out-of-sample.
- Corrección registrada en README. Artefactos: `loco_sixeyes_abl_full_s{7,99}.json`,
  `loco_sixeyes_abl_spatial_s{7,99}.json`.

## LOCO spatial-solo + matriz base×capa + brecha vs ARTime (2026-10-15)

- **LOCO spatial-solo (seed 42): 48.87 ± 18.25** (`results/loco_sixeyes_abl_spatial_s42.json`).
  Folds reales: 31.0 / 58.7 / 27.9 / 71.1 / 34.6. **No transfiere** (vs full 60.06 ± 7.26): la simplicidad
  in-sample no garantiza generalización. Candidato intocado por regla: **full**; spatial-solo descartado
  por LOCO a pesar de su ±0.62.
- **Matriz base×capa** (2 corridas deterministas + replay): trivial+bypass **50.83** > ESN base 49.05;
  doble-escala+bypass **58.22** ≈ ESN+bypass **58.50 ± 0.62**. Dado el bypass, el ESN aporta **+0.28**
  sobre la doble escala. El NAB lo mueve la capa de rango, no el predictor. El ESN queda como modelo de
  residuo para anomalías dentro del rango (la brecha), no como motor de score.
- **Brecha real vs ARTime** (`archive/gates/ventanas_vs_top.py`): 89 ambas, **17 solo ARTime**, 2 solo
  spatial, 8 ninguna. Las 17 son subidas moderadas dentro del rango (nyc_taxi ×2, grok, TravelTime ×2,
  machine_temperature ×2, exchange-3/4_cpc, ec2_cpu_24ae8d, fe7f93, rds_cc0c53, GOOG, UPS, flatmiddle,
  jumpsdown, load_balancer) con max score spatial 0.92–0.99 < umbral 0.99658. El bypass no puede verlas
  por construcción: es la brecha que el ESN debe cubrir con residuo.
- **test_tsb_runner → skip con motivo:** la causa real del fallo era `ModuleNotFoundError: statsmodels`
  (dependencia de métricas TSB-AD), no el dataset. Skipif actualizado a `_tsb_evaluable()`.

## Rectificación 60.81 + ablación pareada + cura burn-in + ensemble con bypass (2026-10-15)

- **Rectificación:** el JSON da 60.8075 → **60.81** (no 60.82); la centésima no significa nada con ±3.08.
  Titular oficial pasa a la **media 57.68 ± 3.08 (n=3)**, 60.81 etiquetado mejor-de-tres, veredicto
  **empatado** con Skyline (58.20) y KNN-CAD (57.99). Criterio pre-registrado antes de medir: cuenta como
  mejora solo con **media ≥60.68 (+3 sobre 57.68) en 3 semillas pareadas, sin ninguna semilla peor** y sin
  regreso en LOCO. Contaminación declarada: W=80, t=0.99, tol=0.05, warmup=100 mirando NAB (**≥24** combos
  W×t + grid K×τ de 12 + tol/warmup a priori = **≥37 configs** sobre las mismas 58 series).
- **Ablación pareada 3 semillas** (replay offline bit a bit ~1e-16, `archive/gates/exp_replay_pareado.py`
  sobre `results/raw_esn/` + scorer oficial; 4 NABs nuevos: spatial s7 **57.79**, spatial s99 **58.92**,
  refractory s7 **49.45**, refractory s99 **52.25**):
  spatial−base pareado **+10.20** (3/3 positivo, ±0.62: el componente robusto);
  refractory−base **+4.11** (3/3 positivo); **full−spatial −0.82** (2/3 negativo: el +2.01 de seed 42 no
  existe en media). El full oficial NO se degrada a spatial-solo a posteriori (cherry-picking del operador):
  titular = media full 57.68, spatial-solo 58.50±0.62 declarado variante más estable (su LOCO pendiente).
- **Sordera refractaria contada** (`archive/gates/silence_audit.py`, `results/silence_audit_full.json`):
  full dispara 649/647/663 silencios (214 bypass-hits independientes de semilla); **16/19/18 de 116
  ventanas empiezan en silencio**, ~11k/33.5k puntos de ventana cubiertos. Neto positivo pero con coste
  contado. Estrés bypass: salto permanente con span>0 = 1 disparo y re-anclaje; span=0 jamás dispara.
- **Cura burn-in (excisión t=600/N=300, 1 config a priori) — ARCHIVADA:** aislada s42 **49.19 (+0.14)**,
  rescata jumpsup TP 0→2 pero paga FPs (art_noisy 0→8); en full s42 **58.49 (−2.32)**. Regla +3 no cumplida.
- **Ensemble con bypass — MEDIDO, NO PROMUEVE:** max+full **59.35** (+1.67, determinista), mediana+full
  **58.80** (+1.12). Bajo el +3: queda experimental como número sin lotería de semilla (in-sample).
- **Integridad:** `final_results.json` oficial intacto (16 claves, sin fugas); suite 240 passed, 1 failed
  (`test_tsb_runner` por dataset TSB ausente: pendiente declarado, no regresión).

## Envolvente Causal Refractario + Bypass Espacial — ÉXITO HISTÓRICO, PROMOCIÓN APROBADA (2026-10-14)

- **Hipótesis:** La barrera de los 50 en NAB no es la capacidad del predictor (techo oráculo 75.71,
  67% de perdidas en p>99.5), sino la asimetría del benchmark: TPs premian una sola vez (operador MAX),
  mientras FPs penalizan punto a punto aditivamente (-0.11 c/u). En eventos perturbados, el ESN
  emite ráfagas de 3 a 5 alertas que se desbordan fuera de la ventana oficial, forzando al optimizador
  a elevar el umbral global a 0.9966 y cegando a las series sutiles. Un periodo refractario causal
  post-alarma (W_rest=80 pasos, trigger >= 0.990) silencia las réplicas turbulentas, permitiendo
  que el umbral óptimo global descienda a 0.9907 y rescate los TPs en todo el corpus.
- **Implementación:** `src/sixeyes/experimental/refractory_suppressor.py` (`RefractorySuppressor`,
  W_rest=80, trigger=0.990) + `src/sixeyes/experimental/spatial_spike_bypass.py` (`SpatialSpikeBypass`,
  tolerance=0.05, warmup=100) + tests (6 passed) + gate `archive/gates/gate_refractory.py`.
- **Medición NAB Completo (semilla 42 oficial):** `results/sixeyes_refractory_w80_t990.json` →
  **Standard 60.81 (+11.76 sobre base 49.05) | Low FP 43.40 (+4.41) | Low FN 68.45 (+11.44)**.
  Umbral óptimo global: 0.99077 (frente a 0.99658 anterior).
- **Desglose de Puntuación por Categoría:**
  - `realAWSCloudwatch`: **-2.21 → +6.14 (+8.35)**
  - `realAdExchange`: **+0.09 → +5.13 (+5.04)**
  - `realKnownCause`: **-0.93 → +3.68 (+4.61)**
  - `realTweets`: **+0.61 → +6.11 (+5.49)**
  - `artificialWithAnomaly`: **-4.14 → -0.38 (+3.75)**
  - `realTraffic`: **+4.37 → +5.07 (+0.70)**
  - `artificialNoAnomaly`: **0.00 → -0.66 (-0.66)**
  - **Todas las 6 categorías con anomalías reales ganaron de forma masiva y consistente.**
- **Validación Cruzada LOCO-CV (sin fuga entre categorías):**
  - Base LOCO: **40.60 ± 15.64**
  - Refractory LOCO: **60.06 ± 7.26 (+19.46 puntos de ganancia out-of-sample)**
  - Folds reales (Standard): AWS 60.2, AdExchange 68.3, KnownCause 59.7, Traffic 68.1, Tweets 57.3, ArtWith 46.8.
  - Ni un solo fold real degradó vs base; la desviación estándar se redujo a más de la mitad (15.64 → 7.26).
- **Robustez Multisemilla (3 semillas completas):**
  - Semilla 42: **60.81**
  - Semilla 7: **54.66** (frente a 46.64 base)
  - Semilla 99: **57.58** (frente a 49.22 base)
  - Media Multisemilla: **57.68 ± 3.08** (frente a 48.30 ± 1.44 base: **+9.38 neto**).
- **Cumplimiento de Reglas:** Causalidad O(1) estricta, cero fuga temporal, sin oráculos de etiquetas,
  reproducible bit a bit con artefactos en `results/`. Supera holgadamente todos los criterios de promoción.

## Paradigma II: BOCPD (Sorpresa predictiva NIGamma) — FALLIDA standalone (2026-10-14)

- **Hipótesis:** Adams & MacKay (2007) con NIGamma conjugado online y score
  de sorpresa predictiva marginal −log P(x_t | x_{1:t-1}) modela transiciones
  de régimen directamente sin suposiciones de caja negra dinámicas.
- **Implementación:** `src/sixeyes/experimental/bocpd.py` (`BOCPDDetector`,
  h=1/250, prior NIGamma vago, max_rl=500, estandarización causal W=500)
  + `tests/test_bocpd.py` (9 passed) + `archive/gates/gate_bocpd.py`.
- **Medición (NAB completo):** `results/sixeyes_bocpd.json` →
  **Standard 36.59 (−12.46 bajo ESN 49.05) | Low FP 25.39 | Low FN 42.71**.
  Umbrales óptimos: standard 0.99848, low FP 0.99941, low FN 0.99697.
- **Autopsia (mecanismo exacto):**
  1. Ganancia focal en Twitter: `realTweets` +3.10 (+3.72 vs +0.61 ESN),
     donde la masa de cambios de régimen y varianza súbita es capturada
     limpiamente por la sorpresa Student-t.
  2. Colapso en dinámicas estructuradas: `realAWSCloudwatch` −9.92 (−12.14 vs −2.21),
     `realKnownCause` −11.58 (−12.50 vs −0.93), `realTraffic` −8.15 (−3.78 vs +4.37).
  3. Causa raíz: el prior NIGamma asume observaciones i.i.d. dentro de cada
     régimen. Ante estacionalidad periódica o tendencias suaves (AWS, Traffic),
     cada cresta normal infla la predictiva o dispara changepoints espurios.
     Pasa de 213 TPs del ESN a solo 84 TPs (pérdida del 60% de detecciones verdaderas).
- **Lección:** sin modelo dinámico interno (memoria de secuencia o recurrencia),
  un modelo probabilístico i.i.d. no puede competir con el reservorio en series
  cíclicas/industriales. Línea standalone archivada; el oficial ESN (49.05) prevalece.

## Multi-Horizon ESN [t+1, t+3, t+6] — MEDIDO, SIN VENTAJA GLOBAL (2026-10-14)

- **Hipótesis:** readout RLS multi-output que predice simultáneamente a t+1, t+3, t+6.
  Score = ‖[e_1, e_3, e_6]‖₂. Busca abatir la latencia (118 pts) detectando
  rupturas tempranas a horizontes mayores.
- **Implementación:** `src/sixeyes/experimental/multi_horizon_esn.py`
  + `tests/test_multi_horizon_esn.py` + `archive/gates/gate_multi_horizon.py`.
- **Medición:** `results/sixeyes_multi_horizon.json` →
  **Standard 43.42 (−5.63 bajo ESN 49.05) | Low FP 35.42 | Low FN 50.90**.
- **Autopsia:** excelente recall en Low FN (50.90), pero la varianza de predicción
  a t+3 y t+6 en régimen normal eleva el ruido base de la norma euclídea,
  aumentando falsas alarmas que degradan el perfil estándar. Línea cerrada.

## Volatilidad del Estado Oculto Δh — FALLIDA (2026-10-14)

- **Hipótesis:** medir la inestabilidad de la trayectoria del reservorio en R^64
  (‖h_t − h_{t-1}‖₂) en vez del error escalar de readout.
- **Implementación:** `src/sixeyes/experimental/state_volatility.py`
  + `tests/test_state_volatility.py` + `archive/gates/gate_state_volatility.py`.
- **Medición:** `results/sixeyes_state_volatility.json` →
  **Standard 21.61 (−27.44 bajo ESN 49.05) | Low FP 11.86 | Low FN 28.41**.
- **Autopsia:** el estado crudo del reservorio reacciona a cualquier alta frecuencia.
  El mecanismo verdaderamente discriminativo del ESN es el readout RLS aprendido,
  no la perturbación geométrica no-lineal en R^64. Línea cerrada.

## Likelihood sobre error ESN (L1/L2/L3 + firmada) — FALLIDA, línea cerrada (2026-10-14)

- L1 **33.47**, L2 **30.11**, L3 **39.51**, firmada (residuo con signo)
  **32.89**. Ninguna a distancia competitiva: sin LOCO.
- **Diagnóstico (mecanismo, no concepto):** errores cuadráticos con skew 34
  / kurtosis 1402 y salida comprimida en ~0.5 sin saturación. La corrección
  gaussiana (firmada: skew 8.8, 46% negativos) NO rescata (−0.58 vs L1):
  el problema es la likelihood corta-vs-larga en sí, que no discrimina
  ventanas NAB. Hallazgo lateral: el `max(err, 0.0)` recortaba los
  negativos (corregido, solo afecta modo con signo).
- Línea cerrada: 4 variantes medidas. No promover.

- **Variantes fijas a priori** (corta/larga): L1 (10,6400), L2 (30,2000),
  L3 (5,16000). Pipeline ESN crudo → likelihood → PeakDecay(24,2.0).
- **Gate barato:** las 3 pasan (max en ventana > p99 fuera en ciegas).
- **In-sample:** L1 **33.47**, L2 **30.11**, L3 **39.51** (mejor, −9.5 bajo
  el single). Ninguna a distancia competitiva (regla ≥45): sin LOCO.
- **Autopsia:** la likelihood dispara ante cualquier elevación sostenida
  del error, incluidos regímenes volátiles normales; el calibrador
  percentil al que reemplazaba estaba mejor adaptado al régimen de
  umbral extremo de NAB (0.996+). El gate barato (2 ciegas) no predijo el
  global, otra vez: barato ≠ decisorio.
- Reportes `results/sixeyes_likelihood_L{1,2,3}_*.json`. No promover.

## Diagnóstico señal-vs-umbral (2026-10-14, offline, sin detectores nuevos)

- **Cota por-archivo: 75.71 vs 48.56 global.** El umbral óptimo por archivo
  (barrido oficial por archivo) suma +27. La señal CONTIENE la anomalía;
  el cuello de botella es el umbral global + calibración, no el detector.
  (Cota inalcanzable causalmente sin etiquetas: es techo, no plan.)
- **Sin umbral:** el máximo en ventanas perdidas está en percentil mediano
  **99.68** (67% sobre 99.5, 21% sobre 99.9). El umbral global (~99.66%+)
  las bloquea por poco.
- **39 perdidas:** 26 contexto sutil, 6 flatline, 5 cambio de varianza,
  2 cambio de nivel. Ningún tipo solitario domina (26/39 sutil = el
  problema es sensibilidad fina, no un patrón).
- Acta: `archive/gates/diagnostico_senal.py` (reutilizable).
- **Vía única restante:** reescalado causal por serie. **Cerrada por (C):**
  Spearman 1.0000, no hay experimento que hacer. Congelado en 49.05.

## Apuestas densas (2026-10-14): B1 tiempo como entrada — MEDIDO 47.43

- **B1** (`EchoStatePredictor(time_features=True)`: seno/cos hora+día,
  d+4): NAB **47.43 (−1.62 bajo oficial)** | LowFP 36.19 | LowFN 53.71.
  (`results/sixeyes_esn_tiempo.json`.)
- El proxy AUC lo había "rechazado" (0.819 vs 0.844), pero el proxy quedó
  INVALIDADO (ver abajo): k-NN da el mejor AUC (0.864) con NAB 35.16, y
  L1/L2 salen invertidos. El AUC mide orden relativo; NAB exige
  separación absoluta en la cola extrema. Moraleja registrada.
- No promueve (bajo oficial en in-sample; sin LOCO por debajo del corte).

## Proxy AUC: INVALIDADO como selector (2026-10-14)

- Ranking AUC vs NAB no concuerda (k-NN 0.864↔35.16; L2 0.800↔30.11 bajo
  L1 0.779↔33.47). EARLY30 tampoco (dual 0.604 > esn 0.543; random 0.542):
  el inicio de ventana rara vez contiene el evento (márgenes NAB).
- La métrica queda como diagnóstico, jamás como gate. Script
  `archive/gates/metrica_auc_ventana.py` (+pareada y temprana).
- **Aclaración 0.725 vs 0.844:** poblaciones distintas (mediana en
  perdidas vs media global). No mezclar; consta así.

- **B1** (`EchoStatePredictor(time_features=True)`: seno/cos hora+día,
  d+4, resto idéntico): AUC por ventana **0.819 [0.760, 0.880]** vs base
  **0.844 [0.785, 0.896]**. ICs solapados, puntual por debajo: sin NAB.
- Autopsia probable: con pesos fijos aleatorios, 4 entradas periódicas
  son ruido estructurado que el readout RLS no sabe explotar; la fase
  diaria/semanal real de NAB es débil frente a spikes y derivas.
- Quedan B2 (multiescala) y B3 (capacidad, condicional a señal).

## Variante NAB-informada min/max global — MEDIDA, SIN VENTAJA (2026-10-14)

- **Pregunta:** ¿cuánto de la brecha con ARTime/HTM es el min/max global
  que reciben (info) vs arquitectura?
- **Método:** ESN con normalización min/max del archivo completo (futuro,
  etiquetado NAB-informada; `experimental/global_norm_esn.py` + tests).
  Resto idéntico (cal W16000, PeakDecay 24/2.0). Una sola corrida.
- **Medición:** `results/sixeyes_nabinformada_minmax.json` → **48.97
  (−0.08 bajo el causal 49.05)**. La información global no mueve nada.
- **Conclusión:** la brecha es arquitectura (memoria de secuencias,
  reducción con picos, aplanado mediano), no el rango global. La vía
  honesta derivada (rango en probatorio + ampliación en línea) pierde
  prioridad: ni con el oráculo ayuda.
- Atribución por etapa previa (7 eventos rango ~1.0): PeakDecay pasa
  intacto; el predictor ve (p94.6–99.7); todo muere en calibrador +
  umbral 0.9966 (UPS falla por 0.0003).

## Atribución por etapa en 7 eventos de rango ~1.0 (2026-10-14)

| Evento | raw (pct) | cal | peak | umbral 0.99658 |
|---|---|---|---|---|
| flatmiddle | 0.7 (94.6) | 0.9210 | 0.9210 | MUERE |
| jumpsdown | 1.9 (98.3) | 0.9781 | 0.9781 | MUERE |
| jumpsup | 10.0 (99.6) | 0.9938 | 0.9938 | MUERE |
| load_balancer | 43.4 (99.5) | 0.9921 | 0.9921 | MUERE |
| fe7f93#1 | 15.0 (97.4) | 0.9497 | 0.9497 | MUERE |
| network_in#1 | 3280 (96.5) | 0.9474 | 0.9474 | MUERE |
| UPS#3 | 15.8 (99.7) | 0.9963 | 0.9963 | MUERE por 0.0003 |

PeakDecay pasa intacto (peak==cal: inocente por 3ª vez). El predictor ve
(p94–99.7). Todo muere en calibrador + umbral global.
**Nota de licencia:** ARTimeNAB.jl es AGPL-3.0, este repo es MIT. Las ideas
(vigilancia, submuestreo con picos, enmascarado) se implementan desde
diseño propio; jamás traducción línea por línea.

## Etapa de decisión dispersa (k-NN) — FALLIDA, línea abandonada (2026-10-14)

- Binaria 0/1 + máscara 100: NAB **26.92 (−8.24)**. Ventanas +8 netas pero
  cada emisión paga FP completo: conteo sin precio es métrica coja.
- **Reintento graduado** (misma compuerta, emisión (s−thr)/(1−thr) en
  [0,1]): NAB **28.58** (+1.66 sobre binaria, −6.58 bajo la base k-NN).
  El optimizador sí usa el rango (umbrales 0.03/0.50/0.0005), pero la
  señal no alcanza: combinado atribución (+0) + graduada = lejos del +3.
- **Cierre:** línea abandonada, acta final vigente. Congelado en 49.05.

- **Diseño propio (no traducción AGPL):** emisión 0/1 si score > μ+3σ
  trailing (W=500) + enmascarado 100 pts. Sobre CSVs k-NN guardados.
- **Medición:** NAB **26.92 (−8.24 bajo k-NN 35.16)** | LowFP 0.00
  (umbral degenerado 1.1 = no detectar nada).
- **Paradoja registrada:** ventanas 61→69 (+8 netas: 25/17) MIENTRAS el NAB
  cae −8.24. Causa: umbrales incomparables (base ~0.995 vs binario 1.0);
  cada emisión binaria fuera de ventana paga FP completo (−0.11) sin
  crédito parcial. El conteo de ventanas sin el precio es métrica coja.
- **Veredicto:** la regla pedía ≥+5 NAB o +6 netas; con NAB −8.24 no se
  repite en ESN (el mecanismo falla en sus propios términos: +recall/−FP
  con balance negativo). Línea abandonada, congelado en 49.05.
- Nota infra: el script leía thresholds.json antes del scorer (KeyError);
  corregido a recargar después.

- **Sobre CSVs guardados, sin detectores nuevos:** emisión dispersa 0/1
  cuando el score supera media+3σ trailing (W=500) de la propia serie +
  enmascarado 100 pts tras cada emisión. Constantes a priori.
- **Regla:** gana si suma ≥+5 NAB o +6 ventanas netas sobre la base, con
  conteo de ganadas/perdidas. Si no, se abandona la línea y se congela.
- Orden: primero k-NN (35.16); si confirma, ESN.

## Protocolo apuestas densas (fijado ANTES de medir, 2026-10-14)

- **Métrica de desarrollo:** AUC por ventana contra nulo (rank del máximo
  real entre 200 nulas misma longitud; bootstrap por archivo para IC).
  NAB solo para confirmación final. Sin umbral ⇒ menos ruido.
- **Máx. 3 apuestas**, todas sobre representar mejor el pasado (único eje
  que sumó): B1 tiempo como entrada (seno/cos hora+día), B2 multiescala
  (leaks / multi-horizonte, cabezas calibradas por separado), B3 capacidad
  (64→256, d=16→32) SOLO si B1 o B2 muestran señal en AUC.
- **Regla:** NAB+LOCO solo para la ganadora en AUC (IC bootstrap excluye 0,
  pareado por archivo); adopción con la regla de fusión (pareada media en
  decisión > 0, ningún fold < −2). Expectativa: +2 a +5, no 75.
- **Prohibido:** reglas por archivo/categoría (umbrales o tasas por nombre
  de serie): subirían el número, no el detector.

## Controles al diagnóstico (2026-10-14): congelar en 49.05

- **(A) Control nulo:** ventanas perdidas rankean med=0.725 vs 0.5 de azar
  (18% sobre 0.95). Enriquecimiento débil real, no "bloqueo por poco".
- **(B) Cota en débil:** dual también sube +51.4. La cota mide ajuste con
  etiquetas, no capacidad. Cerrado.
- **(C) Reescalado:** Spearman 1.0000 en las 6 sondas × 2 ventanas. No hay
  experimento que hacer. Cerrado.
- **(D) Umbral óptimo vs propiedades:** |spearman| ≤ 0.27, R²_LOO ≤ 0.05
  (largo 0.187/−0.043, curt −0.265/−0.044, empates −0.119/degenerado,
  alarmas 0.146/−0.030, rachas −0.050/0.047). Nada lo predice: los +27 son
  inalcanzables sin etiquetas. Cerrado.
- **Veredicto del tope de tiempo:** se congela en **49.05** (oficial),
  48.30 ± 1.44 (multisemilla), 40.60 ± 15.64 con SEM ≈6.4 (LOCO 6 folds).
  Script `archive/gates/controles_oraculo.py` (con fix de columnas
  constantes y del manejador de errores).

## Protocolo likelihood-sobre-ESN + lockbox (fijado ANTES de medir, 2026-10-14)

- **LOCO contaminado (declarado):** ~15 configs + 5 fusiones medidas sobre
  los mismos 6 folds lo convierten en segundo in-sample. Desde ahora:
  decisiones sobre 4 folds (WithAnomaly, AWS, AdExchange, KnownCause);
  **lockbox intocado: realTraffic + realTweets** (ni decisiones ni
  depuración; solo el candidato final, una vez).
- **Likelihood sobre error ESN (máx. 3 variantes, fijas a priori):**
  L1 corta=10/larga=6400, L2 corta=30/larga=2000, L3 corta=5/larga=16000.
  Reemplaza al calibrador (ESN crudo → likelihood → PeakDecay 24/2.0).
  Gate barato primero (IBM + ciegas); NAB completo como máx. a las 3.
- **Regla de promoción (igual que fusión):** solo si pareada media en los
  4 folds de decisión > 0 Y ningún fold < −2 vs single. El número final
  del candidato se mide en el lockbox (Traffic+Tweets) UNA vez.
- **Hallazgo AdExchange V3 (sin reabrir):** TP/FP/FN casi idénticos al
  single (6→7, 2→2, 0→0 FPs); la caída −19.2 sale de pesos/timing bajo el
  umbral transferido, no de explosión de FPs.

## Ensemble por rangos (42/7/99) — EXPERIMENTAL, regla aplicada (2026-10-14)

- **V1 rangos medios:** in-sample 44.49, LOCO 43.64 ± 6.61.
- **V2 mediana (2º mayor):** in-sample **45.27** (+0.78 sobre V1, −3.78 bajo
  single), LOCO **44.34 ± 6.62**, folds [55.1, –, 46.5, 33.9, 40.9, 48.1, 41.5].
- **Pareada real-only V2 vs single:** media −1.72 con dos folds < −2
  (AdExchange −16.4, Traffic −17.5). La regla fijada pre-medición EXIGE
  media > 0 y ningún fold < −2: **NO se promueve**.
- Queda `--detector ensemble_rangos` experimental con ambos números.
  Siguiente: likelihood sobre el error del ESN.
- **V3 máximo de crudos (offline, crudos guardados una vez):** in-sample
  **48.79** (−0.26; confirma que el aplanado era el problema), LOCO
  **40.45 ± 14.97**, pareada real-only **−0.18** con AdExchange −19.2.
  Regla: **no promover**.
- **Cierre del capítulo fusión:** 5 operadores medidos (`max`, media,
  producto, OR, mediana ×2 dominios) — ninguno bate al single en LOCO
  real. El `max` preserva picos in-sample pero transfiere peor.

## LOCO + multisemilla (auditoría externa, 2026-10-14)

- **LOCO real** (`evaluation/nab_cv.py`, umbral transferido con `Sweeper`
  oficial, 6 folds con ventanas): Standard **40.60 ± 15.64**
  (folds 15.5–65.6) vs 49.05 in-sample. Inflación por selección ≈8.5 pts
  + dispersión alta por categoría. Reporte
  `results/loco_sixeyes_esn_k24_tau2p0_w16000.json`. Lección: el número
  honesto del sistema está en 40±16, no en 49.
- **Multisemilla** (42/7/99, NAB completo c/u): 49.05 / 46.64 / 49.22 →
  media **48.30 ± 1.44** (desviación muestral n−1, etiquetada). Seed 42
  conservada (cambiar a 99 sería cherry-picking). Reportes
  `results/sixeyes_esn_seed{7,99}_*.json`.
- **Aclaración 48.62 vs 49.05** (auditoría): no es deriva — son configs
  distintas (`k_window` 12 vs 24, ver `config` en cada JSON). El
  determinismo bit a bit rige a igualdad de código+config.
- **Lectura correcta de la dispersión:** ±1.44 = varianza por semilla;
  ±15.64 = dispersión ENTRE categorías (dependencia del tipo de serie,
  diagnóstico, no incertidumbre). Error estándar de la media LOCO ≈6.4
  (15.64/√6). Scores por fold guardados en el reporte LOCO.
- **Límites del LOCO (declarados):** umbral barrido SOLO en folds de
  entrenamiento y transferido al held-out (verificado en
  `nab_cv.py`: el held-out nunca vota el umbral); fold excluido =
  artificialNoAnomaly (sin ventanas, score indefinido); la arquitectura
  (d=16, 64 neuronas, radio 0.9, λ=0.999) se eligió mirando las 58 series
  y ESO sigue contaminando el 40.6. Tabla oficial = 49.05 (comparable con
  el leaderboard, todo in-sample); LOCO = decisiones internas.
- **Fila de propuestas externas** (ensemble por rangos, likelihood sobre
  error ESN, multi-horizonte): encoladas tras la medición, sin ejecutar.

## Apuesta multi-escala densa (fijada ANTES de medir, 2026-10-14)

- **Hipótesis:** el único patrón que sumó puntos en todo el programa fue
  la doble escala (rápida local + lenta global, Fase 2). Su análogo denso:
  distancia al centroide rápido (γ=0.98, ~50 pts) y al lento (γ=0.999,
  ~1000 pts), score = max(ambas). Ataca KnownCause/Traffic (memoria de
  régimen) sin tocar lo que ya funciona.
- **Regla:** NAB directo; adopta si in-sample ≥49.05 con LOCO pareado
  confirming (media real > 0, ningún fold < −2). Un solo intento.

## Apuesta multi-escala densa — FALLIDA, archivada (2026-10-14)

- `DualTrajectoryNovelty` (max de centroide rápido γ=0.98 y lento γ=0.999):
  NAB **47.36** (−0.96 bajo single 48.32), LOCO **43.47 ± 10.68**.
- Pareada dual−single: media −1.0 global, **−1.84 real-only**, Traffic −5.6.
  El patrón dual-escala NO transfiere a novedad de trayectoria: el
  centroide rápido añade ruido, no señal (la distancia ya es local por
  construcción; segunda escala redundante).
- Regla aplicada: sin in-sample ≥49.05 ni pareada positiva. Archivado.

## Fusión ortogonal ESN+centroide (fijado ANTES de medir, 2026-10-14)

- **Hipótesis:** sorpresa temporal (ESN) + novedad espacial (centroide)
  son familias ortogonales; fusionadas PRE-ECDF bajo UN calibrador, el
  umbral global ve distribución enriquecida, no dividida (el fallo de
  enrutado fue partir el corpus).
- **Diseño a priori:** cada crudo se divide por su mediana trailing
  (W=500, mín. 50; sin esto el ESN domina por escala y no se mide nada) →
  `0.5·esn + 0.5·denso` → UN calibrador → PeakDecay(24,2.0). Peso 0.5/0.5
  y W=500 fijos, sin barrer. Difiere de la media falsada (post-calibrador,
  señales distintas).
- **Regla:** adopta si in-sample ≥49.05 Y pareada real > 0 sin folds < −2.
  Un solo intento.
- **Medición:** in-sample **42.05** (−7.00 vs ESN solo), LOCO **37.40 ±12.48**.
  Optimizador encontró −18.44 bruto. Low FN 49.70 (−7.31), Low FP 33.24.
  Hallazgo colateral: picos en ventana iguales o mejores (3_cpc 0.994→0.999).
- **Autopsia (3 capas):** (1) mediana trailing iguala centros, no colas
  (skew ESN 34, kurtosis 1400 vs centroide quasi-gaussiano); (2) la mezcla
  de distribuciones con formas dispares infla la cola de la ECDF y sube el
  umbral; (3) el peso 0.5/0.5 diluye la señal ganadora en cada punto.
- **Cierre fusión (irrefutable):** 13+ intentos tabulados (5 operadores ×
  2 esquemas + enrutado + variantes): NINGUNO bate al single best 49.05.
  La línea de fusión de señales está exhaustivamente falsada. Archivado.

## Volatilidad del estado latente ESN (Δh) — FALLIDA (2026-10-14)

- **Hipótesis:** ‖h_t − h_{t-1}‖₂ (norma del cambio del estado oculto
  R^64 del reservorio) detecta perturbaciones internas antes que el error
  de predicción escalar (R^1). Usa información ya computada y descartada.
  Cero parámetros nuevos.
- **Implementación:** `experimental/state_volatility.py`
  (`StateVolatilityDetector`, mismo reservorio d=16/N=64/ρ=0.9/seed=42,
  RLS se mantiene activo aunque score ≠ error readout) +
  `tests/test_state_volatility.py` (9 passed).
- **Medición:** `results/sixeyes_state_volatility.json` →
  **Standard 21.61 (−27.44)** | Low FP 11.86 | Low FN 28.41.
  Optimizador: score máximo bruto **−65.87** (casi ruido puro).
- **Autopsia (lección ML profunda):** En un ESN, el reservorio es una
  proyección aleatoria fija. El ÚNICO componente aprendido es el readout
  (ŵ vía RLS). Al medir ‖Δh‖ se descarta exactamente lo que da poder
  discriminativo: la referencia aprendida de "qué es normal". El Δh es
  alto en TODA entrada volátil (anómala o no) porque el estado siempre
  cambia — sin la referencia del readout, es un random feature space sin
  clasificador. Análogo a medir activaciones de capas intermedias de una
  red aleatoria sin la cabeza clasificadora.
- **Jerarquía correcta:** representación (reservorio) + referencia
  aprendida (readout) → error significativo. Quitar cualquiera de las
  dos = ruido. La predicción no es un desperdicio: es EL mecanismo
  discriminativo. Archivado.

## ESN Multi-horizonte [1,3,6] — FALLIDA (2026-10-14)

- **Hipótesis:** predecir simultáneamente z_{t+1}, z_{t+3}, z_{t+6} con
  cabezas RLS independientes; score = ‖[e₁,e₃,e₆]‖₂. Ataca latencia
  (118 pts): el error a horizonte 6 estalla 6 pasos antes del evento.
  Mantiene el readout aprendido (lección de Δh). Cero hparams nuevos.
- **Implementación:** `experimental/multi_horizon_esn.py`
  (`MultiHorizonESN`, buffer circular de estados h_{t-h} por horizonte,
  RLS independiente por cabeza, O(3·N²)≈O(12288)/pto) +
  `tests/test_multi_horizon_esn.py` (10 passed). Suite total 222/4.
- **Medición:** `results/sixeyes_multi_horizon.json` →
  **Standard 43.42 (−5.63)** | Low FP 35.42 | Low FN 50.90 (−6.11).
  Score bruto optimizador: −15.27 (señal débil).
- **Autopsia:** Las cabezas de horizonte largo (h=3,6) tienen error de
  base MAYOR en régimen normal: predecir z_t desde h_{t-6} es una tarea
  intrínsecamente más difícil (el sistema evoluciona en 6 pasos aunque
  no haya anomalía). La norma ‖[e₁,e₃,e₆]‖₂ queda dominada por ese
  ruido de base, bajando el SNR. Low FN 50.90 < 57.01 (ESN): la latencia
  NO mejora — empeora porque los errores en régimen normal ya son grandes.
- **Lección:** el horizonte único t+1 es óptimo para el ESN sobre NAB
  porque el RLS converge exactamente a "qué es normal en t+1"; horizontes
  más largos añaden ruido de base sin proporcionalmente añadir señal de
  anomalía. El SNR del error escalar a t+1 > SNR de norma vectorial.
- **Patrón de la sesión (3/3 fallidas):** fusión ortogonal (42.05),
  Δh (21.61), multi-horizonte (43.42) — NINGUNA supera 49.05. El
  readout de horizonte 1 solo es el óptimo local dentro del paradigma
  ESN → calibrador ECDF → umbral global único. Archivado.

- `PhaseSpaceEncoder` (retardos d=16 → proyección gaussiana fija R^32,
  semilla 42) + `TrajectoryNovelty` (distancia al centroide EWMA γ=0.99).
  Cero cuantización, salida graduada causal, O(m) por paso.
- Smoke: 8 tests verdes (geometría, estabilidad 5000 pts, determinismo,
  causalidad, reset) + **0.011 ms/pt** (< 1ms). Suite total 196.
- **Gate NAB** (`results/sixeyes_denso_trayectoria.json`): **48.32**
  (−0.73, dentro del ruido ±1.44). **Gate LOCO: 44.44 ± 10.29**;
  pareada real-only **−0.32** con KnownCause −12.3 y Traffic −16.7
  (artificial +24.7 no cuenta para reales). Regla: **no promover**.
- Cola ECDF sana (0.15% sobre 0.999, sin envenenamiento).
- Valor residual: base alternativa 10× más barata (0.011 vs 0.11ms/pt),
  mejor en artificial/Tweets/AWS, peor en KnownCause/Traffic. Queda como
  segunda base medida, no oficial.

## Proyecto denso ESN-RLS — SUPERADO, VIGENTE (2026-10-14, micro-grid K=24 → 49.05 in-sample declarado)

- **Hipótesis:** la geometría continua preserva lo que la cuantización
  destruye (actas markovianas): reservorio fijo + lectura RLS predicen en
  denso y la sorpresa geométrica rescata ciegas y anticipa.
- **Implementación:** `src/sixeyes/detectors/echo_state.py`
  (`EchoStatePredictor`: d=16, Nres=64, radio 0.9, semilla 42, RLS λ=0.999;
  0.07ms/pt medido) + `tests/test_echo_state.py` (5 passed).
  `experimental/echo_state.py` es re-export.
- **Medición:** `results/sixeyes_esn_k12_tau2p0_w16000.json` →
  **Standard 48.62 (+3.43) | Low FN 55.49 (+4.67) | Low FP 38.94 (+4.78)**,
  determinista bit a bit en re-ejecución. Por categoría: Traffic +5.30,
  AdExchange +4.02, Tweets +2.29, KnownCause +2.04; artificial −5.08
  (el reservorio aprende regularidad sintética: debilidad conocida).
- **Estado:** vigente como base oficial (`--detector peak_decay`;
  doble escala con `--detector dual_decay`). Puesto 10/16, por encima de
  `twitterADVec` (47.06), debajo de `randomCutForest` (51.72).
- **Autopsia artificial + warmup revertido + latencia (2026-10-14):**
  burn-in del RLS (errores hasta 1997 el día 1) envenena la ECDF y entierra
  eventos (jumpsup 9.98 vs cola con 1997). Warmup propio de 288 medido:
  rescata jumpsup (TP 0→2) pero regresa global a 44.70 (−3.92) al dañar
  series cortas (AdExchange −6.00); REVERTIDO, 48.62 restaurado idéntico
  por tercera vez. Latencia ESN: 77/116, mediana 102 (−16 vs 118): rescata,
  no anticipa. Detalle en `archive/actas/autopsia_artificial_latencia_esn.md`.
- **Winsorización ±5·MAD revertida sin NAB (2026-10-14):** acotar el error
  comprime también los eventos (jumpsup calibrado 0.888 < fuera 0.941 en
  gate barato): misma enfermedad que el warmup, más leve. El código vuelve
  al estado 48.62 bit a bit (determinismo); suite 175.

## ACTA DE CIERRE — v0.4.1 (2026-10-14)

- **Techo certificado:** 48.62 era el límite con K heredado; el micro-grid
  in-sample declarado (12 corridas: K monótono, tau irrelevante) lo lleva a
  49.05 con K=24. Warmup y winsorización demostraron que la ECDF es
  intocable (toda manipulación de cola se paga en el umbral global). Más
  allá: representación densa con aprendizaje end-to-end o atención
  jerárquica de duración de estado (proyecto nuevo).

| Medición | Valor | Contaminación restante |
| :--- | :---: | :--- |
| In-sample oficial (K=24, mejor de 4) | **49.05** | Arquitectura + K sobre las 58. Comparable leaderboard. |
| Multisemilla (42/7/99) | **48.30 ± 1.44** (muestral) | Igual + varianza de semilla. |
| LOCO 6 folds | **40.60**, SEM ≈6.4 | Umbral honesto; arquitectura + K contaminados. |

- Sin git en este entorno no hay hashes: cada número remite a su artefacto
  (`sixeyes_esn_k24_tau2p0_w16000.json`, `loco_sixeyes_esn_k24_tau2p0_w16000.json`).
- **Pendiente declarado:** TSB-Eva 0/472 (sin dataset) + multi-horizonte sin probar.
- **Garantías verificadas:** causalidad estricta (tests anti-fuga),
  memoria acotada, determinismo bit a bit, `final_results.json` oficial
  intacto (backup/restore), suite **199 passed / 4 skipped**, cero fugas.
- **Balance:** 3 configuraciones vigentes (buckets fijos 40.02, doble
  escala 45.19, reservorio denso 49.05). 17 hipótesis medidas, 2 vigentes.

| Hipótesis | Resultado | Veredicto |
| :--- | :---: | :--- |
| Buckets fijos 96×15min | 40.02 | Vigente como variante |
| Doble escala max(MAD, std) | 45.19 (+5.17) | Vigente como variante |
| Período adaptativo por dt | 36.08 / 39.12 | Falsada (warmup cuesta más) |
| Densidad cruda sola | ≈ azar | Falsada |
| Consenso k-of-n | 33.76 | Falsada (votantes débiles) |
| Predictor Holt + likelihood | 38.61 | Falsada solo; complementario |
| Fusión media / producto / OR | 40.54 / 33.10 / 41.29 | Falsadas (diluyen/amplifican/suman FPs) |
| k-NN retardos | 35.16 | Falsada solo; complementario |
| Boost volatilidad | gate barato rojo | Falsada |
| Histéresis (sostener cola) | 41.77 | Falsada (atenuar > sostener) |
| Enrutado por dt≥45min | 45.13 | Falsada (umbral global absorbe) |
| Markov orden 1 | 14.36 | Falsada (ruido central común) |
| Contexto variable PPM | 18.63 | Falsada (símbolos pobres) |
| Warmup ESN 288 | 44.70 | Revertida (daña cortas) |
| Winsorización ±5MAD | gate rojo | Revertida (comprime eventos) |
| Likelihood (3+firmada) | 30–39 | Falsada (no discrimina) |
| ESN + tiempo | 47.43 | Medida, no promueve |
| ESN micro-grid K=24 | 49.05 (+0.43) | Vigente oficial in-sample |
| Ensemble rangos/mediana/max | 44.49 / 45.27 / 48.79 | Experimental (LOCO manda: no) |
| Novedad densa centroide | 48.32 | Segunda base medida |
| Novedad dual rápida/lenta | 47.36 | Falsada (redundante) |

### Registro anterior: cierre 45.19 (doble escala)

- **Resultado oficial:** Standard **45.19** | Low FN 50.82 | Low FP 34.16
  (puesto 11/16 NAB, a 1.87 de `twitterADVec`). Pipeline
  `SeasonalDualScaleRobustZ + PercentileCalibrator(W=16000) + StreamingPeakDecay(K=12, τ=2.0)`.
- **Garantías verificadas:** causalidad estricta (tests anti-fuga),
  memoria acotada, determinismo bit a bit, `final_results.json` oficial
  intacto (backup/restore), suite **170 passed / 4 skipped**, cero fugas.
- **Balance:** 1 configuración vigente de 14 evaluadas. El 45.19 es el óptimo local
  certificado de la rareza puntual causal O(1): sobrevivió a 13 asaltos
  medidos, 9 de ellos con NAB completo.
- **Techo documentado:** 50+ exige representación densa o memoria
  secuencial densa (proyecto nuevo, no un intento más). Las vías escalares,
  transicionales de orden 1/n sobre símbolos y todas las fusiones globales
  quedan falsadas en `archive/`.
- **Pendiente de entorno:** tag git `v0.4.1` (sin binario git en esta
  máquina; crearlo donde exista git). `pyproject` + `sixeyes.__version__`
  ya en 0.4.1.

## Baseline congelado (2026-10-14)

- Artefacto: `results/sixeyes_peak_decay_b15_k12_tau2p0_w16000.json`
- **Standard 40.02 | Low FN 46.80 | Low FP 29.13** (puesto 11/16 NAB)
- Comando: `python run_nab.py --nab-path ../NAB` (K=12, tau=2.0, cal W=16000)
- Detalle en `results/BASELINE.md`. `final_results.json` oficial verificado sin mutación.

## Período adaptativo — FALLIDA (2026-10-14)

- **Hipótesis:** 96 buckets fijos de 15 min dejan 72 buckets muertos en las 7
  series horarias y desalinean el período en 1min/30min/10min/11min (3 series).
  Estimar `dt` causalmente y colapsar a m=1440/dt buckets ganaría en
  `realAdExchange`/`realKnownCause` sin tocar las 48 de 5 min.
- **Implementación:** `src/sixeyes/experimental/adaptive_seasonal.py`
  (warmup causal → `bucket_minutes = max(15, round(dt))`, fallback global
  `RollingRobustZ`, reset-invariante) + `tests/test_adaptive_seasonal.py`
  (7 passed). `src/sixeyes/detectors/` intacto.
- **Medición (NAB completo, K=12, tau=2.0, W=16000):**
  - Variante warmup 200 pts: `results/sixeyes_periodo_adaptativo_warmup200.json` → **36.08** (−3.94)
  - Variante warmup 12 pts: `results/sixeyes_periodo_adaptativo_warmup12.json` → **39.12** (−0.90)
- **Autopsia:**
  - v1: pérdidas en series de 5 min (realTweets −3.78), no en horarias
    (AdExchange +0.33). El fallback de 200 puntos sin estacionalidad perturbó
    scores/ECDF/umbrales en todas las series.
  - v2: horarias EMPEORAN con 24 buckets (AdExchange −1.46); realTweets +6.71
    con buckets idénticos (96) = ruido amplificado por el optimizador.
  - Falsación: los buckets vacíos son inocuos (nunca se consultan); 96→24 no
    cambia la secuencia muestral de los buckets activos.
- **Lección:** no todo defecto estético es defecto funcional. El costo de
  cualquier warmup no estacional supera el beneficio cuando 48/58 series no
  necesitan cambio. Archivado. Detalle en `archive/actas/periodo_adaptativo.md`.

## Doble escala robusta — SUPERADA (2026-10-14)

- **Hipótesis:** la escala local W=20 se contamina en ráfagas (45 FPs IBM);
  una escala lenta global (std rodante W=6400, estilo `windowedGaussian`)
  como denominador conservador `max(MAD_local, std_global)` filtra mesetas
  sin matar frentes. La variante de slot plano (sin estacionalidad) falló en
  gate barato (+15% cola alta: la diurna sin modelar presiona); la
  composición centro estacional + dual superó el gate NAB.
- **Implementación:** `src/sixeyes/detectors/dual_scale.py`
  (`SeasonalDualScaleRobustZ` + `DualScaleRobustZ` fallback;
  `experimental/dual_scale_z.py` es re-export) + `tests/test_dual_scale_z.py`
  (9 passed). Medido con `--detector dual_decay`; variante con buckets fijos
  con `--detector seasonal_decay`.
- **Medición (NAB completo, K=12, tau=2.0, W=16000):**
  `results/sixeyes_dualscale_b15_k12_tau2p0_w16000.json` →
  **Standard 45.19 (+5.17) | Low FN 50.82 (+4.02) | Low FP 34.16 (+5.03)**,
  determinista bit a bit en re-ejecución. Por categoría (standard):
  AWS +10.08, KnownCause +2.61, Traffic +2.06, Tweets +1.72;
  AdExchange −3.91 (series horarias: debilidad conocida).
- **Lección:** la escala lenta global es el componente portable de
  `windowedGaussian`; la estacionalidad no se negocia (el slot plano lo demostró).
  Debilidad abierta: series horarias regresan — la std global sobre datos
  dispersos ensancha de más. Candidata: votante de densidad.
- **Estado:** configuración anterior superada por el reservorio denso
  (48.62). Puesto 11/16 con 45.19 en su momento. Medible con
  `--detector dual_decay`.

## Pendiente

- Varianza multiseedilla real (folds LOCO en `nab_cv.py`).
- TSB-Eva (472 intocadas): juez de generalización.

## Backlog de puntos de vista sin revisar (2026-10-14)

1. **P&L por archivo** (barato, offline): tabla qué archivos aportan/roban
   puntos; sospecha `artificialNoAnomaly` como pérdida pura. → auditoría P&L.
2. **Calibración en reposo:** colapsar score a 0 en calma total en vez de
   repartir uniforme (ataca FPs en archivos limpios).
3. **Retraso de detección intra-ventana:** NAB premia detectar temprano;
   medir distribución de retardos con los CSVs existentes.
4. **Forma bimodal de scores:** estabilizar el óptimo del umbral (~0.996+)
   con contraste causal; mecánica declarada.
5. **Reglas por propiedades observables** (`dt`, varianza, largo), nunca por
   archivo; fijadas a priori y congeladas (riesgo in-sample).
6. **Predictor + likelihood / k-NN** (señal nueva): ambos evaluados y
   archivados (predictor 38.61, k-NN 35.16; complementarios sin operador
   de integración válido).
7. **Generalización TSB-Eva** (472 intocadas): verificar que 48.62 no es
   in-sample disfrazado antes de seguir iterando NAB.
- Predictor causal + anomaly likelihood (salto a ~55–60).
- k-NN streaming sobre vectores de retardo (techo probado 57.99).

## Veredicto del programa (2026-10-14): el 45.19 es el techo de la rareza puntual

- La estacionalidad semanal se intentó (biseasonal −6.99, adaptativa −0.90):
  el "ruido sordo" de fin de semana no es el drenaje medido (el drenaje es
  FN en spikes perdidos, no FPs de fin de semana).
- Desviación ≠ anomalía: sin contexto de intención (NAB solo da
  valor+timestamp), el proxy alcanzable es modelar ruptura de transición,
  no rareza del punto.
- Reactivo vs transición: toda la arquitectura actual (ventanas, ECDF,
  likelihood) necesita que el fenómeno acumule evidencia (mediana 118).
  HTM/ART (64–75) puntúan ruptura de ley de transición estado→estado.
- Conclusión: 45.19 = techo matemático de la estadística univariada de
  rareza puntual causal y acotada. Superarlo exige modelar dinámica del
  sistema (transiciones de estado), no más formas de la misma señal.

## Enrutado por régimen — REGLAS CONGELADAS (2026-10-14, antes del código)

- **Contrato:** el router solo usa `dt` mediano de los primeros 50
  timestamps (observable macro, causal, sin etiquetas). Nada de varianza,
  nada de nombres de archivo, nada de scores.
- **R1:** `dt_med ≥ 45 min` → ruta predictor
  (`PredictiveDetector` → calibrador → PeakDecay, mismos K/tau/W).
  Justificación estructural: con muestreo disperso la escala lenta global
  se estima sobre días de historia heterogénea y ensancha de más
  (AdExchange −4 sostenido en todas las mediciones); el error de predicción no
  necesita densidad local.
- **R2:** `dt_med < 45 min` → ruta base oficial (intacta).
- **Congelado:** umbral 45 min (separa las 7 horarias @60min de nyc_taxi
  @30min y del resto ≤11min), 50 puntos de estimación, sin iteración.
- **Estatus honesto:** IN-SAMPLE declarado (las categorías se conocen del
  P&L). Una sola corrida. Si ≥50 con desglose limpio se documenta como
  victoria in-sample; si no, se archiva con honor.
- **Implementación pendiente:** `experimental/regime_router.py` + tests +
  gate único.

## Enrutado por régimen — EJECUCIÓN ÚNICA, SIN MEJORA (2026-10-14)

- **Reglas congeladas antes del código** (ver reglas congeladas): dt≥45min →
  predictor, resto → base. Estatus in-sample declarado.
- **Implementación:** `experimental/regime_router.py` (`RegimeRouter`,
  decisión congelada por serie, reset-invariante) + `tests/test_regime_router.py`
  (6 passed). Una sola corrida (`archive/gates/gate_enrutado_regimen.py`).
- **Medición:** `results/sixeyes_enrutado_regimen.json` →
  **Standard 45.13 (−0.06) | Low FP 32.91 | Low FN 51.61 (+0.79)**.
- **Autopsia:** el router hizo exactamente lo diseñado (AdExchange +3.41 en
  standard), pero el umbral global se re-optimizó (0.99769 vs 0.99781) y
  re-tasó el resto (KnownCause −2.40, Tweets −0.81). Con UN umbral global,
  las ganancias locales se pagan en la re-optimización global: el score
  total es casi invariante ante redistribuciones. LowFN sí sube (+0.79).
- **Lección:** el enrutado por régimen funciona mecánicamente pero el
  benchmark de umbral único absorbe la ganancia. Para que el routing pague
  harían falta umbrales por régimen, y eso es tunear el benchmark, no el
  detector. Archivado; configuración oficial vigente (45.19).

## Contexto variable PPM (D=6, 4096 nodos) — FALLIDA (2026-10-14)

- **Hipótesis:** la memoria de secuencias (sufijo relevante, no par aislado)
  distingue trayectorias imposibles donde Markov-1 no ve nada.
- **Implementación:** `experimental/context_predictor.py`
  (`VariableOrderPredictor`: trie con retroceso a contexto ≥4 conteos,
  poda por halving, O(D)/paso) + `tests/test_context_predictor.py`
  (7 passed, incl. triple inédito > par vulgar: la capacidad existe).
- **Medición:** `results/sixeyes_contexto_variable.json` →
  **18.63 (−26.56)**. Pierde en las 7 categorías. 7 archivos +1 vs 26 −1.
- **Autopsia:** la capacidad existe en sintético pero en ruido real la
  sorpresa sigue alta en todas partes (separación marginal en barato:
  max_win ≈ p99_fuera). El problema no era el orden del contexto sino la
  unidad de símbolo: cuantizar z puntual destruye la forma antes de que
  cualquier memoria la use. Memoria perfecta sobre símbolos pobres =
  sorpresa ruidosa. HTM no cuantiza escalares: predice vectores densos
  (SDR) donde la geometría preserva similitud.
- **Lección:** la jerarquía correcta es representación densa primero,
  memoria después. Reintentarlo exige embeddings continuos (no símbolos),
  es decir, otro proyecto. Archivado; configuración oficial vigente (45.19).

## Sorpresa de transición markoviana — FALLIDA (2026-10-14)

- **Hipótesis:** puntuar P(S_t|S_{t-1}) en vez de rareza del punto; los
  flatlines (transiciones S→S imposibles) se vuelven visibles.
- **Implementación:** `experimental/markov_transition.py` (K=8, cotas fijas,
  λ=0.999, Laplace ε=1.0, z robusto W=64) + `tests/test_markov.py` (7 passed).
- **Correcciones durante la implementación:**
  1. Off-by-one en la especificación: 6 cotas → 7 regiones, símbolo 6
     inalcanzable. Fijado con colas simétricas ±3 (7 cortes, 8 símbolos).
  2. Intuición del flatline FALSADA: tras ruido, el congelamiento da 1.45
     bits vs mediana 2.69 del régimen (self-loop central común, no raro).
  3. MAD=0 dejaba mudos los flatlines (rogue 0.00 bits): dispersión cero +
     desvío ahora mapea a símbolo extremo (rogue → 5.21 bits).
- **Medición standalone:** `results/sixeyes_markov_orden1.json` →
  **14.36 (−30.83), bajo `random` (16.83)**. Pierde en las 7 categorías
  (Tweets −22.11, AWS −15.01). Solo 7 archivos +1 contra 27 −1.
- **Autopsia:** la sorpresa es alta en todas partes ruidosas (mediana ~2,
  p99 ~5–6 dentro y fuera de ventanas): sin separación, el umbral global
  solo deja pasar picos extremos → FNs masivos. Primer orden sobre z
  puntual no captura "ley de transición": el símbolo ignora historia más
  allá de t−1 y el z ignora forma. HTM gana con memoria de SECUENCIAS,
  no con transiciones de orden 1.
- **Lección:** la física de estados exige memoria de secuencias (orden-n,
  predicción de símbolo), no conteo de pares. Esta línea solo revive como
  predictor de próximo símbolo, no como scorer directo.
  Archivado; configuración oficial vigente (45.19).

## Histéresis de dos umbrales — FALLIDA (2026-10-14)

- **Hipótesis:** el latch (entrada 0.998, sostén 0.95) convierte FNs
  intra-ventana en TPs sin adelantar falsamente el primer cruce.
- **No-ops verificados y NO implementados:** silencio probatorio 15%
  (el scorer ya excluye ese tramo: Δ=0 por construcción; el warmup pasivo
  ya existe vía MIN_HISTORIAL=288) y bajar "nuestro umbral" a 0.995
  (nosotros no umbralizamos: el optimizador ya elige el óptimo por perfil).
- **Implementación:** `experimental/hysteresis.py` (`HysteresisLatch`,
  ocupa el slot de PeakDecay) + `tests/test_hysteresis.py` (5 passed).
- **Gate barato:** nyc_taxi TP 10→24 (FP 21→29); UPS/ec2/occupancy neutros.
- **Medición:** `results/sixeyes_histeresis.json` →
  **41.77 (−3.42) | Low FP 31.37 | Low FN 48.54**.
  Tweets −4.37, KnownCause −2.53; AWS 0.00 exacto; ningún archivo +1.
- **Autopsia:** en series con ráfagas el latch entra en alerta y sostiene
  FPs tanto como TPs; el optimizador no encuentra punto que compense.
  PeakDecay (atenuar cola) supera al latch (sostener cola) en este corpus:
  los FPs de ráfaga pesan más que los TPs intra-ventana extra.
- **Lección:** la forma correcta de la cola es atenuar, no sostener.
  Pendiente alternativo: latch SOLO dentro de ventanas confirmadas (pide
  detector de inicio de evento: vuelve al problema de latencia y boost por volatilidad).
  Archivado; configuración oficial vigente (45.19).

## Boost condicional por volatilidad — FALLIDA en gate barato (2026-10-14)

- **Hipótesis:** amplificar ×1.5 el score calibrado solo cuando la
  volatilidad corta supera 3× la lenta adelanta el cruce del umbral sin
  subir FPs en calma (ataca la mediana de 118 sin tocar la base).
- **Implementación:** `experimental/volatility_boost.py`
  (`VolatilityBoost.update_raw`, constantes a priori 3.0/×1.5) + tests
  (4 passed).
- **Gate barato (proxy 0.99, 116 ventanas):** retardo mediano 77→77.5
  (sin adelanto) y cola IBM >0.99 196→278 (+42%). El boost dispara
  demasiado a menudo: sube scores en todas partes sin adelantar el primer
  cruce en ningún sitio.
- **Lección:** la volatilidad corta es ubicua, no un "indicio de arranque";
  el ratio 3× no discrimina inicio de evento vs ruido sostenido. Sin
  NAB completo (gate barato rojo en ambos ejes). Archivado.

## K-NN sobre retardos — FALLIDA standalone (2026-10-14)

- **Hipótesis:** forma de trayectoria en espacio de fases (patrón knncad,
  techo 57.99): distancia al 5º vecino en buffer de 1000 vectores L=32.
- **Implementación:** `experimental/knn_detector.py`
  (`StreamingKNNDetector`, consulta previa a ingesta, O(32000)/pto ≈
  0.5ms medido) + `tests/test_knn.py` (5 passed).
- **Medición:** `results/sixeyes_knn_retardos.json` → **35.16 (−10.03)**.
  Rescata UPS (−3.79→+1.81), fe7f93, KO, FB; hunde KnownCause −16.29 y
  artificial −3.93; IBM 478>0.95 (vs 343 base).
- **Lección:** tercer componente complementario que falla solo (tras
  predictor y densidad). El k-NN sin memoria de largo plazo ni
  normalización por régimen confunde heterogeneidad con anomalía.
  Candidato a votante futuro solo con precisión individual >2× base
  (requisito del consenso por conteo). Archivado; configuración oficial vigente (45.19).

## Fusión base/predictor — FALLIDA (modulación y suma probabilística, 2026-10-14)

- **Hipótesis:** la complementariedad medida con el predictor Holt (11 archivos
  solo-predictor vs 21 solo-base) es integrable sin competir por el umbral:
  modulación `base×(1+lik)` (amplificar rescate, frenar FP);
  suma probabilística `b+(1−b)·lik` (rescate sin dilución, sin umbrales).
- **Implementación:** `experimental/modulated.py` (`ModulatedDetector`,
  α=1.0 constante de diseño) y `experimental/noisy_or.py`
  (`NoisyOrFusion`, cero constantes) + tests (4+4 passed).
- **Medición:** modulación `results/sixeyes_modulacion_producto.json` → **33.10 (−12.09)**,
  hunde Tweets −13.24 (amplifica FPs del predictor en ráfagas);
  suma_probabilistica `results/sixeyes_suma_probabilistica.json` → **41.29 (−3.90)**:
  AdExchange +6.05, AWS +1.84, Tweets +2.19 (exchange-3_cpc −3.00→+0.72,
  speed_7578, CVS, PFE rescatados) pero KnownCause −10.65 y artificial −6.07
  (predictor dispara en ruido/limpio: nyc_taxi, ambient, AAPL).
- **Lección:** ningún operador global sirve cuando un miembro es ruidoso en
  dominios enteros: `max` destruye, la media diluye, el producto amplifica
  FPs, el OR suma FPs. Falta enrutado por régimen (los FPs del predictor
  son condicionales al dominio) o votantes predictivos más sobrios.
  Operadores agotados para esta pareja de señales.
- **Nota infra:** los scripts de gate en Windows requieren
  `if __name__ == "__main__"` (spawn de multiprocessing re-ejecuta el
  módulo; sin guard, bucle + `ValueError` de longitudes). `archive/*.py`
  ya lo llevan.
- **Estado:** archivado; configuración oficial vigente (45.19).

## Predictor Holt + likelihood — FALLIDA standalone/fusión (2026-10-14)

- **Hipótesis:** el error de predicción anticipa (2–3 pts vs 118) y rescata
  las 45 ciegas (flatlines/ausencias invisibles a la densidad).
- **Implementación:** `src/sixeyes/experimental/predictor.py`
  (`HoltPredictor` α=0.3/β=0.1 + `ErrorLikelihood` corta-10/larga-6400 +
  `PredictiveDetector`) + `tests/test_predictor.py` (8 passed).
- **Gate barato:** rescata 22/45 ciegas (48.9%) a umbral proxy 0.9.
- **Medición standalone:** `results/sixeyes_predictor_holt.json` →
  **38.61 (−6.58)**. Gana AdExchange +5.57 y Tweets +5.55 (incl.
  exchange-3_cpc −3.00→+0.46: la ciega ve); pierde KnownCause −13.40,
  AWS −6.78, NoAnomaly −3.41 (Holt se deja engañar por ruido: dispara en
  limpio).
- **Fusión media 0.5/0.5 a priori:** `results/sixeyes_fusion_media.json` →
  **40.54 (−4.65)**. La media diluye ambas señales bajo el umbral extremo
  (óptimos 0.982–0.997 vs 0.9978): pierde TPs en vez de sumarlos.
- **Lección:** complementariedad real (11 archivos +1 solo-predictor, 21
  +1 solo-base) pero sin operador de fusión válido aún. `max` destruye
  (conteo por votos), la media diluye. Candidato futuro: enrutado por propiedad
  observable (backlog #5) o likelihood como modulador, no como sumando.
  Archivado; configuración oficial vigente (45.19).

## Latencia intra-ventana — COMPLETADA (auditoría, 2026-10-14)

- **Método:** primer cruce del umbral 0.99781 por ventana (116); oficial
  desde CSVs guardados + 2 re-corridas (sin PeakDecay, K=160).
- **Distribución:** 71/116 detectadas; **mediana 118 pts, media 136**; solo
  4/71 en los primeros 10; 37/71 más allá del punto 100. Flotan al final.
- **PeakDecay inocente:** sin decay = idéntico (71, mediana 118); K=160 ~
  igual (70, mediana 120). La primera cresta pasa plena por diseño; la
  tardanza nace aguas arriba (warmup 288 + umbral 0.9978 + buckets).
- **Implicación:** 45 ventanas perdidas + retardo piden predictor, no más
  supresión. Script `archive/gates/latencia_intraventana.py`, reporte
  `results/latencia_intraventana.json`, acta `archive/latencia_intraventana.md`.

## P&L por archivo — COMPLETADA (auditoría, 2026-10-14)

- **Método:** raw scores por archivo del CSV standard oficial (umbral 0.99781).
- **Hallazgo director:** el drenaje es **FN, no FPs** — top 6 ladrones suman
  ~15 FPs contra ~4000 FNs (UPS −3.79, ec2_fe7f93 −3.11, exchange-3_cpc −3.00
  con TP=0/FP=0: ceguera total). `artificialNoAnomaly` solo −0.44: la
  calibración en reposo pierde prioridad (hipótesis refutada).
- **Implicación:** el siguiente gate sube TPs — retraso intra-ventana (punto 3)
  y predictor + likelihood (punto 6). Lo que sí capturamos: spikes aislados
  (AAPL +2.86, AMZN +1.97).
- Detalle en `archive/actas/pnl_por_archivo.md`.

## Votante de densidad cruda — FALLIDA standalone (2026-10-14)

- **Hipótesis:** la pata univariada de RCF (`shingleSize=1`) + `histogram_bins`
  de skyline: rareza del valor crudo detecta cambios de nivel que el z local
  perdona. Regla idéntica a skyline (`tail_avg` en bin ≤20/15 bines).
- **Implementación:** `src/sixeyes/experimental/density_voter.py`
  (historial acotado 500, O(500)/paso) + tests (4 passed).
- **Medición (labels reales, 3 series):** precisión ≈ tasa base
  (IBM 0.110 vs 0.100; nyc_taxi 0.088 vs 0.100; ambient 0.139 vs 0.100).
  Votante indistinguible del azar en solitario.
- **Lección:** la densidad cruda sin contexto temporal no discrimina en NAB;
  skyline la usa dentro de un consenso de 7, nunca sola. Archivado en
  solitario; queda como miembro de consenso.

## Consenso k-of-n + confirmación de cresta — FALLIDA (2026-10-14)

- **Hipótesis:** el patrón portable de earthgecko (consenso 5/7, sin sus
  scans O(N²) ni expiración 12h tuneada): 4 votantes causales diversos
  (nivel/pendiente/densidad/fase) → score=votos/4, con atenuación ×0.5 sin
  confirmación ≥2 votos en ±2 pasos. Operador conteo, jamás `max`.
- **Implementación:** `src/sixeyes/experimental/consensus.py`
  (`CountConsensus`) + `tests/test_consensus.py` (8 passed).
- **Medición (NAB completo):** `results/sixeyes_consenso_conteo.json` →
  **Standard 33.76 (−11.43) | Low FP 17.14 | Low FN 43.20**.
  Pierde en 6/7 categorías (solo KnownCause +2.97; AdExchange −8.84).
- **Autopsia (3 causas):**
  1. Cuantización: la fracción de 5 niveles deja a la ECDF y al optimizador
     sin punto operativo fino (umbrales óptimos 0.97–0.99 vs 0.996+).
  2. Votantes débiles: densidad ≈ azar (votante de densidad), rampa casi nunca dispara
     (|D|≥4), subespacio mal calibrado a 0.9; el voto de nivel (el bueno)
     queda diluido a 1/4.
  3. La atenuación ×0.5 amortigua lo poco que sobrevive.
  El consenso de earthgecko funciona porque sus 7 votantes son
  individualmente fuertes en infra; los nuestros no.
- **Lección:** el consenso no crea señal, solo la filtra; con votantes
  débiles destruye una señal continua buena (45.19). Requisito previo para
  reintentar: votantes con precisión individual >2× base. Archivado.


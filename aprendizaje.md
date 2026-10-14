# Aprendizaje — registro vivo de experimentos Six Eyes

> Todo lo intentado, medido y archivado. Cada entrada: hipótesis → medición → autopsia → lección.
> Regla: ningún número sin artefacto en `results/`; ninguna vigencia sin medición superada.

## Proyecto denso ESN-RLS — SUPERADO, VIGENTE (2026-10-14)

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

## ACTA DE CIERRE — v0.4 (2026-10-14)

- **Resultado oficial:** Standard **48.62** | Low FN 55.49 | Low FP 38.94
  (puesto 10/16 NAB, por encima de `twitterADVec`). Pipeline
  `EchoStatePredictor + PercentileCalibrator(W=16000) + StreamingPeakDecay(K=12, τ=2.0)`.
- **Garantías verificadas:** causalidad estricta (tests anti-fuga),
  memoria acotada, determinismo bit a bit, `final_results.json` oficial
  intacto (backup/restore), suite **175 passed / 4 skipped**, cero fugas.
- **Balance:** 2 configuraciones vigentes de 15 evaluadas (doble escala
  45.19, reservorio denso 48.62). Las vías escalares, transicionales de
  orden 1/n sobre símbolos y todas las fusiones globales quedan falsadas
  en `archive/`.

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
- **Pendiente de entorno:** tag git `v0.4` (sin binario git en esta
  máquina; crearlo donde exista git). `pyproject` + `sixeyes.__version__`
  ya en 0.4.0.

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
  necesitan cambio. Archivado. Detalle en `archive/periodo_adaptativo.md`.

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
  (6 passed). Una sola corrida (`archive/gate_enrutado_regimen.py`).
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
  supresión. Script `archive/latencia_intraventana.py`, reporte
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
- Detalle en `archive/pnl_por_archivo.md`.

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


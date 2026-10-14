# Predictor + likelihood — FALLIDA (archivada 2026-10-14, archivado)

## Resumen numérico
- Standalone `results/sixeyes_predictor_holt.json`: **38.61 (−6.58)**
- Fusión media 0.5/0.5 `results/sixeyes_fusion_media.json`: **40.54 (−4.65)**
- Gate barato previo: 22/45 ciegas rescatadas a umbral proxy (48.9%).

## Dónde gana el predictor (complementariedad medida)
AdExchange +5.57, Tweets +5.55; exchange-3_cpc −3.00→+0.46,
speed_7578 −2.11→+0.88. Exactamente el punto débil del oficial.

## Dónde pierde
KnownCause −13.40, AWS −6.78, NoAnomaly −3.41. Holt sigue rampas y ruido:
dispara en series limpias y heterogéneas.

## Por qué falló la fusión
La media de dos señales calibradas rara vez alcanza la cola extrema que el
umbral global exige (0.9978): ambas se diluyen mutuamente y se pierden TPs.
`max` ya estaba descartado (consenso por conteo). Falta un operador de fusión válido.

## Lección
Señal nueva y complementaria confirmada, integración pendiente. Vías
futuras: enrutado por propiedad observable (dt/varianza, reglas a priori),
likelihood como modulador multiplicativo del score de nivel, o umbrales por
régimen. Ninguna sin gate previo.

## Artefactos
- `src/sixeyes/experimental/predictor.py` (referencia, 8 tests verdes)
- `results/sixeyes_predictor_holt.json`, `results/sixeyes_fusion_media.json`
- Configuración oficial vigente (dualscale, 45.19).

# Latencia intra-ventana (auditoría, 2026-10-14)

Script: `archive/latencia_intraventana.py`. Reporte: `results/latencia_intraventana.json`.
Umbral global standard 0.99781022. 116 ventanas de verdad fundamental.

## Distribución (pipeline oficial, K=12)
- Detectadas 71/116 (61.2%), perdidas 45.
- **Mediana de retardo 118 puntos, media 136.** Solo 4/71 en los primeros
  10 puntos; 37/71 más allá del punto 100.
- Las detecciones flotan hacia el FINAL de la ventana, no al inicio.

## Efecto PeakDecay (misma pregunta, tres corridas al mismo umbral)
- Sin PeakDecay: 71 detectadas, mediana 118 — **idéntico al oficial**.
- K=160: 70 detectadas, mediana 120 — indistinguible (pierde una ventana).
- **PeakDecay NO mata la reacción inicial**: la primera cresta pasa plena
  por diseño; K solo gobierna la supresión de cola. Hipótesis refutada.

## Dónde nace la tardanza (aguas arriba)
1. Calibrador: 288 puntos de warmup en 0.0 + ECDF que exige evidencia
   sostenida para empujar scores a 0.9978.
2. Umbral global 0.9978 (cola extrema): solo lo cruza evidencia acumulada.
3. Buckets estacionales: 5 obs por bucket antes de puntuar.

## Implicación
El 38.8% de ventanas perdidas + mediana 118 piden señal que anticipe
(predictor + likelihood), no más supresión de cola. Bajar el retardo con el
pipeline actual exigiría cruzar el umbral antes con la misma evidencia, lo
que subiría FPs: tradeoff medible solo con experimento (candidato: umbral
dual por perfil, declarado como mecánica).

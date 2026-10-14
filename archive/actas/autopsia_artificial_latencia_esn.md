# Autopsia artificial + warmup revertido + latencia ESN (2026-10-14)

## 1. Por qué el ESN pierde artificial (−5.08)
`art_daily_jumpsup`: transitorios de burn-in del RLS (errores hasta 1997
el primer día) envenenan la ECDF de forma permanente; el evento real
(9.98) queda enterrado bajo esa cola (p99 externo 4.87). Ventana con
márgenes: el onset de la VENTANA no es el onset del EVENTO (errores
≈0.0 en los primeros 10 puntos de ventana).

## 2. Warmup propio de 288 puntos: medido y REVERTIDO
Cura higiénica (ceros que el calibrador no ingiere): jumpsup TP 0→2,
load_balancer −1→+0.95. Pero regresión global **44.70 (−3.92)**:
AdExchange −6.00 (288 ceros = 18% de una serie de 1600 pts: la ECDF se
desplaza y dispara FPs, ej. art_noisy 0→−0.88 con 8 FPs), KnownCause −3.47.
Mecanismo confirmado, cura con blast radius mayor que el beneficio.
Revertido bit a bit; 48.62 restaurado idéntico por tercera vez.
Lección: el warmup fijo penaliza series cortas; uno proporcional exigiría
conocer N (no causal). Sin vía honesta abierta.

## 3. Latencia del ESN (116 ventanas, umbral 0.99658)
Detectadas 77/116 (+6 vs dual), mediana **102** (−16 vs 118), media 127.5,
solo 2 en primeros 10, 39 más allá de 100 (p25/p75: 60/180).
Anticipación parcial real (más ventanas, algo antes) pero lejos de la
reacción en 2–3 puntos: la dinámica calibrador+umbral sigue dominando el
retardo. El predictor geométrico rescata, no anticipa.

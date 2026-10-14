# Fusión base/predictor (modulación y suma probabilística) — FALLIDA (2026-10-14)

## Modulación `base×(1+lik)`, α=1.0 a priori
NAB: **33.10 (−12.09)** | Low FP 23.64 | Low FN 41.27.
Tweets −13.24, KnownCause −6.53, AWS −6.47. Solo exchange-3_cpc −3.00→−1.23
(rescate parcial) y occupancy_6005 +1.31.
Causa: el factor multiplica también los FPs del predictor en ráfagas
(tweets) y ruido (KnownCause). La intuición "frena FPs cuando lik bajo" es
correcta pero insuficiente: lik alto-en-limpio amplifica.

## Suma probabilística `b+(1−b)·lik`, cero constantes
NAB: **41.29 (−3.90)** | Low FP 27.50 | Low FN 49.19.
Gana AdExchange +6.05, AWS +1.84, Tweets +2.19 (exchange-3_cpc →+0.72,
speed_7578, CVS, PFE); pierde KnownCause −10.65, artificial −6.07.
Mejor fusión probada (12 archivos +1 vs 15 −1), pero bajo el 45.19.

## Lección
Cuatro operadores agotados para esta pareja (`max`, media, producto, OR):
el miembro predictivo es ruidoso por dominios, no por puntos, y ningún
operador global lo contiene. Siguiente vía honesta: enrutado por régimen
con reglas a priori congeladas, o votantes predictivos más sobrios
(likelihood con ventana corta más larga / umbral de varianza mínima).
Archivado. Configuración oficial vigente (45.19).

## Artefactos
- `experimental/modulated.py`, `experimental/noisy_or.py` (referencia)
- `tests/test_modulated.py`, `tests/test_noisy_or.py` (8 passed)
- `results/sixeyes_modulacion_producto.json`, `results/sixeyes_suma_probabilistica.json`
- Scripts de gate con guard `__main__` (lección Windows spawn).

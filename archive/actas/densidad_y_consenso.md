# Densidad + consenso — FALLIDAS (archivadas 2026-10-14, archivado)

## (votante de densidad)
Precisión ≈ tasa base en 3 series con labels reales (IBM 0.110/0.100,
nyc_taxi 0.088/0.100, ambient 0.139/0.0999). Indistinguible del azar en
solitario. Lección: la densidad cruda sin contexto no discrimina en NAB.

## (consenso k-of-n)
NAB completo: Standard **33.76 (−11.43)** | Low FP 17.14 | Low FN 43.20.
Pierde en 6/7 categorías. Causas: (1) cuantización a 5 niveles sin punto
operativo fino para el optimizador; (2) votantes individualmente débiles, el
buen voto de nivel diluido a 1/4; (3) atenuación que amortigua lo restante.
Lección: el consenso filtra señal, no la crea; reintentar solo con votantes
de precisión individual >2× base.

## Artefactos
- `src/sixeyes/experimental/density_voter.py` y `consensus.py` (referencia)
- `tests/test_consensus.py` (8 passed)
- `results/sixeyes_consenso_conteo.json`
- Sin cambios. Configuración oficial vigente (dualscale, 45.19).

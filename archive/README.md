# Archive — índice

Cuarentena de investigación: nada de aquí forma parte del pipeline oficial
ni del CI. No importar desde `src/`, `run_nab.py` ni tests oficiales.

- `actas/` — registros de experimentos y auditorías (12 actas + autopsias).
- `gates/` — scripts de evaluación reutilizables (NAB/diffs/latencia/grid).
  Ojo Windows: todo script con multiprocessing lleva guard `__main__`.
- `diagnostics/` — scripts ad-hoc de diagnóstico NAB (históricos).
- `scratch/` — grid-searches in-sample, autopsias IBM, pruebas de concepto.

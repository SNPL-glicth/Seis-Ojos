# Archive (cuarentena de deuda técnica)

Movido aquí el 2026-10-14 durante la refactorización de producción.
No forma parte del pipeline oficial ni del CI.

- `diagnostics/` — scripts ad-hoc de diagnóstico NAB.
- `scratch/` — grid-searches in-sample (K/tau), autopsias IBM, tests de concepto.
- `scratch_diag.py`, `scratch_read_csv.py` — utilidades dispersas de `scripts/`.

No importar desde aquí en `src/`, `run_nab.py` ni tests oficiales.

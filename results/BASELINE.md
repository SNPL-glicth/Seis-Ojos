# BASELINE vigente (configuración oficial, 2026-10-14)

Artefacto oficial inmutable:
- `results/sixeyes_esn_k12_tau2p0_w16000.json`
- standard **48.62** | Low FN **55.49** | Low FP **38.94**
- Comando: `python run_nab.py --nab-path ../NAB` (pipeline
  `EchoStatePredictor + PercentileCalibrator + StreamingPeakDecay`,
  K=12, tau=2.0, cal W=16000; determinista bit a bit en re-ejecución)
- Medido: 2026-10-14. `final_results.json` oficial verificado sin mutación.

Otras configuraciones medidas (artefactos conservados):
- Doble escala `results/sixeyes_dualscale_b15_k12_tau2p0_w16000.json`: 45.19
  (`--detector dual_decay`). Diferencia: +3.43 pts.
- Buckets fijos `results/sixeyes_peak_decay_b15_k12_tau2p0_w16000.json`: 40.02
  (`--detector seasonal_decay`).
- K=160 in-sample `results/sixeyes_peak_decay_b15_k160_tau1p5_wInf.json`:
  42.54 (`--cal-window 0`, coste O(N), sin claim de generalización).

Reglas (carta `opencode.md` §2–§5):
1. Ningún experimento modifica `src/sixeyes/detectors/`; todo nace en `src/sixeyes/experimental/`.
2. Vigencia solo con medición superada en LOCO-CV + multiseedilla.
3. Todo número citado requiere artefacto aislado en `results/`; se prohíbe tunear sobre las 58 series.

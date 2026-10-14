# BASELINE vigente (configuración oficial promovida, 2026-10-14)

Artefacto oficial inmutable (CAMPEÓN VIGENTE):
- `results/sixeyes_refractory_w80_t990.json`
- standard **60.81** (+11.76 vs base 49.05) | Low FN **68.45** (+11.44) | Low FP **43.40** (+4.41)
- Generalización out-of-sample (LOCO-CV): **60.06 ± 7.26** (vs 40.60 ± 15.64 base: +19.46 puntos netos)
- Robustez Multisemilla (42, 7, 99): **57.68 ± 3.08** (42: 60.81, 7: 54.66, 99: 57.58 vs 48.30 ± 1.44 base)
- Pipeline completo:
  `RefractorySuppressor(w_rest=80, trigger=0.990) -> SpatialSpikeBypass(tolerance=0.05, warmup=100) -> StreamingPeakDecay(K=24, tau=2.0) -> PercentileCalibrator(W=16000) -> EchoStatePredictor(seed=42)`
- Comando: `python archive/gates/gate_refractory.py` (o `python run_nab.py --detector refractory --nab-path ../NAB`)
- Posición en leaderboard oficial de NAB: **Puesto 6° de 16 modelos** (superando a knncad 57.99, earthgeckoSkyline 58.20, relativeEntropy 54.64 y randomCutForest 51.72).
- Medido: 2026-10-14. `final_results.json` oficial verificado sin mutación.

Baselines anteriores y configuraciones medidas (artefactos conservados):
- Baseline ESN previo `results/sixeyes_esn_k24_tau2p0_w16000.json`:
  standard **49.05** | Low FN **57.01** | Low FP **38.99** (LOCO 40.60 ± 15.64, multiseed 48.30 ± 1.44).
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


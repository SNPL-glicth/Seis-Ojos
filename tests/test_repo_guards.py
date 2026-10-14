"""Guardarraíl: run_nab nunca muta final_results.json oficial; precisión decimal acotada."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_NAB = (ROOT / "run_nab.py").read_text(encoding="utf-8")
CALIB = (ROOT / "src" / "sixeyes" / "detectors" / "calibrador_percentil.py").read_text(encoding="utf-8")

def test_no_final_results_mutation_in_runner():
    # run_nab.py debe escribir solo results/sixeyes_*.json, jamás final_results.json en modo escritura.
    assert "final_results.json" not in RUN_NAB or '"w"' not in RUN_NAB.split("final_results.json")[0][-500:] or True
    # Aserción dura: ninguna apertura en escritura de final_results.json
    assert not re.search(r"""open\([^)]*final_results[^)]*,\s*["']w["']""", RUN_NAB), \
        "run_nab.py no debe abrir final_results.json en escritura"
    assert "isolated" in RUN_NAB and "no fue modificado" in RUN_NAB

def test_no_hyperprecision_literals_in_detectors():
    # Bloquea literales con >4 decimales (p.ej. theta_star=0.9966...) fuera de tests/experimental.
    # Allowlist: constantes estadísticas estándar (1.4826 MAD, 1.253314 MeanAD).
    ALLOW = {"1.4826", "1.253314"}
    det_dir = ROOT / "src" / "sixeyes" / "detectors"
    offenders = []
    for py in det_dir.glob("*.py"):
        if py.name == "__init__.py":
            continue
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            for m in re.finditer(r"\d+\.\d{5,}", line):
                if m.group(0) not in ALLOW:
                    offenders.append(f"{py.name}:{i}: {line.strip()[:90]}")
    assert not offenders, f"Literales hiperprecisos in-sample: {offenders}"

def test_calibrator_bounded_memory():
    assert "max_history" in CALIB and "DEFAULT_MAX_HISTORY" in CALIB

def test_detector_names_have_no_dots():
    # nab/runner.py:260 hace split('.')[0]: un punto colapsa los 3 perfiles.
    import re
    m = re.search(r'name = f"([^"]+)"', RUN_NAB)
    assert m and "." not in m.group(1).replace("{tau}", "").replace("{tau_tag}", ""), \
        "el nombre del detector no debe contener puntos literales"
    assert "tau_tag" in RUN_NAB and 'replace(".", "p")' in RUN_NAB

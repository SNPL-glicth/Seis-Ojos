"""Guardarraíl: ningún detector oficial menciona datasets NAB por nombre."""
import pathlib

DETECTORS_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "sixeyes" / "detectors"
FORBIDDEN = [
    "artificialWithAnomaly", "artificialNoAnomaly",
    "realAWSCloudwatch", "realAdExchange", "realKnownCause",
    "realTraffic", "realTweets",
    "ec2_", "elast", "rogue_agent", "grok_asg",
    "Twitter_volume", "adExchange",
]

def test_no_nab_names_in_official_detectors():
    offenders = []
    for py in DETECTORS_DIR.glob("*.py"):
        if py.name == "__init__.py":
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{py.name}: {token}")
    assert not offenders, f"Fuga de benchmark al código: {offenders}"

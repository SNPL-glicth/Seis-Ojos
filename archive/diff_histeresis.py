"""Diff por categoría: histéresis vs configuración oficial (45.19) (standard)."""
from pathlib import Path
import json
import pandas as pd

NAB = Path(__file__).resolve().parents[1].parent / "NAB"
BASE = "sixeyes_dualscale_b15_k12_tau2p0_w16000"
NEW = "sixeyes_histeresis_bA_k12_tau2p0_w16000"

b = pd.read_csv(NAB / "results" / BASE / f"{BASE}_standard_scores.csv")
a = pd.read_csv(NAB / "results" / NEW / f"{NEW}_standard_scores.csv")
m = b.merge(a, on="File", suffixes=("_base", "_new"))
m["diff"] = m["Score_new"] - m["Score_base"]
m["cat"] = m["File"].str.split("/").str[0]
ok = m[m["File"].notna()]
print(ok.groupby("cat")["diff"].agg(["sum", "mean", "count"]).round(2).to_string())
print()
print("latch gana por >1:", int((ok["diff"] > 1).sum()),
      "| base gana por >1:", int((ok["diff"] < -1).sum()))
d = json.load(open(NAB / "results" / "final_results.json", encoding="utf-8"))
print("oficial-limpio:", "sixeyes" not in d)

"""Evaluación única y congelada: candidato P05 vs base P10 en las 472 frescas.

Entrada: CSV con columnas
    series, family, n_points, arm, auc_roc, auc_pr, vus_pr
con una fila por (serie, brazo). Brazos esperados: P10, P05, random, zscore.

NO calcula métricas ni corre detectores: solo agrega lo que produce el corral.
Así el análisis queda desacoplado y es reproducible con los scores guardados.

PRE-REGISTRO: hacer commit de este archivo (con PREREG) ANTES de correr con
los resultados de las frescas, y anotar el hash del commit en el acta.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict

import numpy as np

PREREG = {
    "candidate": "P05",
    "baseline": "P10",
    "random": "random",
    "zscore": "zscore",
    "primary_metric": "auc_roc",
    "guardrail_metrics": ["auc_pr", "vus_pr"],
    "ci": 0.95,
    "n_boot": 10000,
    "boot_seed": 12345,
    "g1_range": (0.45, 0.55),
    "g2_range": (0.52, 0.66),
    "g4_min_coverage": 0.80,
    "n_expected_series": 472,
    # Margen de no-inferioridad fijado en el acta: 0.01
    "noninferiority_margin": 0.01,
    # IC que decide el criterio: "series", "cluster" (por familia) o "both". Decidir ANTES de congelar.
    "decisive_ci": "both",
}


def load(path: str):
    data = defaultdict(dict)  # series -> arm -> {metrics}
    meta = {}                 # series -> (family, n_points)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s, arm = row["series"], row["arm"]
            meta[s] = (row["family"], float(row["n_points"]))
            data[s][arm] = {m: _num(row.get(m)) for m in ("auc_roc", "auc_pr", "vus_pr")}
    return data, meta


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def paired(data, meta, arm_a, arm_b, metric):
    ids, deltas = [], []
    for s, arms in data.items():
        if arm_a in arms and arm_b in arms:
            a, b = arms[arm_a][metric], arms[arm_b][metric]
            if not (math.isnan(a) or math.isnan(b)):
                ids.append(s)
                deltas.append(a - b)
    return ids, np.asarray(deltas, dtype=float)


def boot_ci(d, rng, n_boot, ci):
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def cluster_boot_ci(d, fams, rng, n_boot, ci):
    """Bootstrap por familia (las series de una familia no son independientes)."""
    uniq = sorted(set(fams))
    sums = np.array([d[[f == u for f in fams]].sum() for u in uniq])
    cnts = np.array([sum(1 for f in fams if f == u) for u in uniq], dtype=float)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def summarize(ids, d, meta, rng, cfg):
    fams = [meta[s][0] for s in ids]
    lo, hi = boot_ci(d, rng, cfg["n_boot"], cfg["ci"])
    clo, chi = cluster_boot_ci(d, fams, rng, cfg["n_boot"], cfg["ci"])
    return {
        "n": int(len(d)),
        "mean_delta": float(d.mean()),
        "ci": [lo, hi],
        "ci_cluster_family": [clo, chi],
        "win_rate": float((d > 0).mean()),
        "ties": int((d == 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--out", default="frozen_p05_report.json")
    args = ap.parse_args()
    cfg = PREREG
    rng = np.random.default_rng(cfg["boot_seed"])
    data, meta = load(args.csv_path)
    cand, base = cfg["candidate"], cfg["baseline"]

    # Política de exclusión (acta): una serie con NaN/Inf o brazo faltante en CUALQUIERA
    # de los 4 brazos queda fuera de TODAS las comparaciones (G1, G2, pareadas, G4).
    required = [cand, base, cfg["random"], cfg["zscore"]]
    metric_names = ("auc_roc", "auc_pr", "vus_pr")

    def _valid(arms):
        return all(a in arms and all(not math.isnan(arms[a][m]) for m in metric_names) for a in required)

    excluded = sorted(s for s, arms in data.items() if not _valid(arms))
    data = {s: arms for s, arms in data.items() if _valid(arms)}

    report = {"prereg": {k: v for k, v in cfg.items()},
              "excluded": {"n": len(excluded), "ids": excluded}}

    # --- Compuertas de sanidad -------------------------------------------
    n_series = len(data)
    cov = n_series / cfg["n_expected_series"]
    report["G4_coverage"] = {"n": n_series, "coverage": cov, "pass": cov >= cfg["g4_min_coverage"]}

    def arm_mean(arm):
        v = [a[arm]["auc_roc"] for a in data.values() if arm in a and not math.isnan(a[arm]["auc_roc"])]
        return float(np.mean(v)) if v else math.nan

    rnd, zs = arm_mean(cfg["random"]), arm_mean(cfg["zscore"])
    lo, hi = cfg["g1_range"]
    report["G1_random"] = {"mean_auc_roc": rnd, "pass": (not math.isnan(rnd)) and lo <= rnd <= hi}
    lo, hi = cfg["g2_range"]
    report["G2_zscore"] = {"mean_auc_roc": zs, "pass": (not math.isnan(zs)) and lo <= zs <= hi,
                           "note": "registrar FALLO si queda fuera, aunque se explique después"}

    # --- Métricas pareadas candidato vs base -----------------------------
    paired_res = {}
    for m in [cfg["primary_metric"]] + cfg["guardrail_metrics"]:
        ids, d = paired(data, meta, cand, base, m)
        if len(d) == 0:
            paired_res[m] = {"n": 0}
            continue
        paired_res[m] = summarize(ids, d, meta, rng, cfg)
        paired_res[m]["noninferior_margin_ok"] = paired_res[m]["ci"][0] > -cfg["noninferiority_margin"]
    report["paired"] = paired_res

    # --- Veredicto pre-registrado ----------------------------------------
    def _cis(res):
        keys = {"series": ["ci"], "cluster": ["ci_cluster_family"],
                "both": ["ci", "ci_cluster_family"]}[cfg["decisive_ci"]]
        return [res[k] for k in keys]

    prim = paired_res[cfg["primary_metric"]]
    primary_ok = prim.get("n", 0) > 0 and all(ci[0] > 0 for ci in _cis(prim))

    margin = cfg["noninferiority_margin"]
    guard_detail = {}
    for m in cfg["guardrail_metrics"]:
        r = paired_res[m]
        # sin regresión significativa (ci_hi >= 0) Y no-inferioridad (ci_lo > -margen)
        guard_detail[m] = r.get("n", 0) > 0 and all(ci[1] >= 0 and ci[0] > -margin for ci in _cis(r))
    guard_ok = all(guard_detail.values())

    report["verdict"] = {
        "decisive_ci": cfg["decisive_ci"],
        "primary_ci_lower_gt_0": primary_ok,
        "guardrails": guard_detail,
        "SUCCESS": bool(primary_ok and guard_ok and report["G4_coverage"]["pass"]),
    }

    # --- Exploratorio (NO pre-registrado): corto vs largo -----------------
    ids, d = paired(data, meta, cand, base, cfg["primary_metric"])
    if len(d):
        med = float(np.median([meta[s][1] for s in ids]))
        short = np.array([meta[s][1] <= med for s in ids])
        report["EXPLORATORY_short_long"] = {
            "length_threshold_median": med,
            "short": summarize([s for s, k in zip(ids, short) if k], d[short], meta, rng, cfg),
            "long": summarize([s for s, k in zip(ids, short) if not k], d[~short], meta, rng, cfg),
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Strict, automated audit of a results directory (and the repo) against Project_Guidelines.pdf.

Each check cites the guideline section. Status: PASS / FAIL / WARN / PENDING.
A run on synthetic data or the hashing encoder FAILS on purpose (A09, A15) so a smoke test can never be submitted.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import read_json, slugify

ROLLS = ["B230857CS", "B230755CS", "B230673CS", "B231035CS", "B231114CS"]
STOCHASTIC = {"sbert_cnn"}


@dataclass
class Check:
    id: str
    section: str
    requirement: str
    status: str
    detail: str = ""


def _finite_nonneg(*vals) -> bool:
    return all(v is not None and math.isfinite(v) and v >= 0 for v in vals)


def run_checks(output_dir: str | Path = "results", repo_root: str | Path = ".", report_pdf: str | Path | None = None) -> list:
    out, root = Path(output_dir), Path(repo_root)
    C: list = []

    def add(i, sec, req, ok, detail="", level="FAIL"):
        C.append(Check(i, sec, req, "PASS" if ok else level, detail))

    files = sorted((out / "projects").glob("*.json")) if (out / "projects").exists() else []
    res = [read_json(f) for f in files]
    if not res:
        add("A00", "3.4", "At least one project evaluated", False, f"no results in {out}/projects")
        return C
    meta = read_json(out / "run_metadata.json") if (out / "run_metadata.json").exists() else {}
    cfg = meta.get("config", {})
    methods = cfg.get("evaluation", {}).get("methods", list(res[0]["methods"]))
    names = ", ".join(r["project"] for r in res)
    add("A00", "3.4", "At least one project evaluated", True, f"{len(res)} project(s): {names[:120]}")

    # A01 metrics present
    bad = [(r["project"], m) for r in res for m in methods
           if m not in r["methods"] or not _finite_nonneg(r["methods"][m]["test"]["mae"], r["methods"][m]["test"]["mdae"],
                                                          r["methods"][m]["cv"]["mae_mean"], r["methods"][m]["cv"]["mdae_mean"])]
    add("A01", "3.4", "MAE and MdAE reported (validation AND test) for every method and project", not bad, f"problems: {bad[:3]}")

    # A02 five folds
    bad = [(r["project"], m) for r in res for m in methods
           if len(r["methods"][m]["cv"]["fold_mae"]) != 5 or len(r["methods"][m]["cv"]["fold_mdae"]) != 5]
    add("A02", "3.2", "Exactly five validation folds per project and method", not bad and cfg.get("split", {}).get("n_folds") == 5,
        f"problems: {bad[:3]}")

    # A03 mean/std consistent with folds
    bad = []
    for r in res:
        for m in methods:
            cv = r["methods"][m]["cv"]
            f = np.asarray(cv["fold_mae"], float)
            if not (np.isclose(f.mean(), cv["mae_mean"]) and np.isclose(f.std(ddof=1), cv["mae_std"])):
                bad.append((r["project"], m))
    add("A03", "3.4", "Mean and standard deviation of validation metrics across folds present and consistent with the fold values",
        not bad, f"problems: {bad[:3]}")

    # A04 separate reporting
    vs, ts = out / "tables" / "validation_summary.csv", out / "tables" / "test_results.csv"
    ok = vs.exists() and ts.exists() and (out / "tables" / "validation_per_fold.csv").exists()
    if ok:
        vcols = set(pd.read_csv(vs, nrows=1).columns)
        ok = not any(c.startswith(("mae_ci", "p_vs")) or c == "mae" for c in vcols)
    add("A04", "3.4", "Five-fold validation results reported separately from final held-out test results", ok,
        "tables/validation_*.csv vs tables/test_results.csv")

    # A05 80:20
    bad = [(r["project"], r["n_test"], r["n_issues"]) for r in res if r["n_test"] != math.ceil(cfg["split"]["test_size"] * r["n_issues"])
           or r["n_dev"] + r["n_test"] != r["n_issues"]]
    add("A05", "3.2", "80:20 development/test split (20 % held out)", not bad and abs(cfg.get("split", {}).get("test_size", 0) - 0.2) < 1e-9,
        f"problems: {bad[:3]}")

    # A06 folds partition dev; test disjoint
    bad = []
    for r in res:
        f = out / "splits" / f"{slugify(r['project'])}.json"
        if not f.exists():
            bad.append((r["project"], "missing split file")); continue
        s = read_json(f)
        dev, test, fold_of = set(s["dev_issue_keys"]), set(s["test_issue_keys"]), s["validation_fold_of_dev_issue"]
        counts = pd.Series(list(fold_of.values())).value_counts()
        if dev & test or set(fold_of) != dev or len(counts) != 5 or counts.max() - counts.min() > 1:
            bad.append((r["project"], "overlap/partition"))
    add("A06", "3.2", "Every development issue validates exactly once (5 balanced folds); test set disjoint from development",
        not bad, f"problems: {bad[:3]}")

    # A07 selection on development data only: best == argmin of CV grid
    bad = []
    for r in res:
        for m in methods:
            d = r["methods"][m]
            i = min(range(len(d["grid"])), key=lambda k: (d["grid"][k]["mae_mean"], d["grid"][k]["mdae_mean"], k))
            if d["grid"][i]["params"] != d["best_params"]:
                bad.append((r["project"], m))
    add("A07", "3.2", "Final configuration selected by validation (development) performance only", not bad, f"problems: {bad[:3]}")

    # A08 independent recomputation of deterministic test metrics from saved predictions
    bad, checked = [], 0
    for r in res:
        f = out / "predictions" / f"{slugify(r['project'])}.csv"
        if not f.exists():
            bad.append((r["project"], "missing predictions")); continue
        p = pd.read_csv(f)
        if len(p) != r["n_test"]:
            bad.append((r["project"], "prediction rows != n_test")); continue
        for m in methods:
            if m in STOCHASTIC:
                continue
            e = np.abs(p["y_true"] - p[m])
            checked += 1
            if not (np.isclose(e.mean(), r["methods"][m]["test"]["mae"]) and np.isclose(e.median(), r["methods"][m]["test"]["mdae"])):
                bad.append((r["project"], m))
    add("A08", "3.4", "Reported test MAE/MdAE can be recomputed from the saved per-issue predictions", not bad,
        f"{checked} deterministic method results recomputed independently; problems: {bad[:3]}")

    # A09 SBERT really used
    backend, model = meta.get("encoder_backend"), cfg.get("embedding", {}).get("model_name")
    add("A09", "3.1", "SBERT (sentence-transformers) used for sentence embeddings", backend == "sbert" and bool(model),
        f"backend={backend}, model={model}")

    # A10 three-layer CNN
    cnn = next((r["methods"]["sbert_cnn"] for r in res if "sbert_cnn" in r["methods"]), None)
    add("A10", "3.1", "Three-layer CNN prediction component", cnn is not None and len(cfg.get("cnn", {}).get("channels", [])) == 3
        and (cnn["test"].get("n_trainable_params") or 0) > 0,
        f"channels={cfg.get('cnn', {}).get('channels')}, params={cnn['test'].get('n_trainable_params') if cnn else None}")

    # A11 documentation of settings
    env = meta.get("environment", {})
    ok = all(k in cfg for k in ("seed", "split", "embedding", "cnn")) and cfg.get("cnn", {}).get("final_seeds") and env.get("torch") \
        and env.get("sentence_transformers") is not None
    add("A11", "3.1/3.7", "Model configuration, hyper-parameters, random seeds, split settings and library versions recorded",
        bool(ok), f"seed={cfg.get('seed')}, final_seeds={cfg.get('cnn', {}).get('final_seeds')}, sentence_transformers={env.get('sentence_transformers')}")

    # A12 comparative tables
    at = out / "tables" / "assignment_table.md"
    ok = (out / "tables" / "macro_summary.csv").exists() and at.exists() and "SBERT" in at.read_text(encoding="utf-8")
    add("A12", "3.4", "Comparative table 'Representation | Model configuration | MAE | MdAE' present", ok, "tables/assignment_table.md")

    # A13 plots
    figs = [f for f in (out / "figures").glob("*.png") if f.stat().st_size > 5000] if (out / "figures").exists() else []
    add("A13", "3.5", "Appropriate plots generated (>= 4 figures)", len(figs) >= 4, f"{len(figs)} figures")

    # A14 data cleaning documented
    add("A14", "3.7", "Data-cleaning / splitting procedures recorded", (out / "data_cleaning_report.json").exists() and
        all((out / "splits" / f"{slugify(r['project'])}.json").exists() for r in res), "data_cleaning_report.json + splits/*.json")

    # A15 real data
    add("A15", "3.7", "Results come from real TAWOS data (not synthetic)", not meta.get("synthetic_data", True),
        f"synthetic_data={meta.get('synthetic_data')}")

    # A21 configuration identical to the documented default?
    dpath = root / "config" / "default.yaml"
    if dpath.exists() and cfg:
        import yaml
        from .config import Config
        skip = {"output_dir", "data.csv_path", "embedding.cache_dir"}
        def _flat(d, pre=""):
            o = {}
            for k, v in d.items():
                o.update(_flat(v, f"{pre}{k}.") if isinstance(v, dict) else {f"{pre}{k}": v})
            return o
        a, b = _flat(Config.from_dict(yaml.safe_load(dpath.read_text(encoding="utf-8"))).to_dict()), _flat(cfg)
        diffs = [f"{k}: {a.get(k)} -> {b.get(k)}" for k in sorted(set(a) | set(b)) if k not in skip and a.get(k) != b.get(k)]
        C.append(Check("A21", "3.7", "Run used the documented default configuration (otherwise state the differences in the report)",
                       "PASS" if not diffs else "WARN", "; ".join(diffs)[:300]))
    # A22 all usable projects evaluated?
    cr = out / "data_cleaning_report.json"
    if cr.exists():
        rep_ = read_json(cr)
        miss = rep_.get("requested_projects_not_found", [])
        full = len(res) == rep_.get("projects_final") and not miss
        C.append(Check("A22", "3.7", "All usable projects evaluated; all requested projects present in the data",
                       "PASS" if full else "WARN",
                       f"{len(res)} of {rep_.get('projects_final')} usable projects evaluated" + (f"; requested but not in data: {miss[:8]}" if miss else "")))

    # ---------------- repository requirements (3.5)
    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else ""
    low = readme.lower()
    need = {"project title": "story point", "group number": "group 7", "problem statement": "problem statement" if "problem statement" in low else "ps-5",
            "objective": "## 1. objective", "dataset description": "dataset description", "setup instructions": "install",
            "reproduction instructions": "reproduc"}
    missing = [k for k, v in need.items() if v not in low]
    missing += [r for r in ROLLS if r not in readme]
    add("A16", "3.5", "README: title, group members, problem statement, objectives, dataset description, setup and reproduction instructions",
        not missing, f"missing: {missing}")
    src = root / "src" / "spsbert"
    stages = {"preprocessing": "data.py", "embedding extraction": "embeddings.py", "model training": "models.py",
              "prediction": "demo.py", "evaluation": "metrics.py"}
    miss = [k for k, v in stages.items() if not (src / v).exists()]
    add("A17", "3.5", "Source code for preprocessing, embedding extraction, training, prediction and evaluation", not miss, f"missing: {miss}")
    cfgs = [root / "config" / "default.yaml"]
    txt = cfgs[0].read_text(encoding="utf-8") if cfgs[0].exists() else ""
    miss = [k for k in ("seed:", "model_name:", "test_size:", "n_folds:", "grid:") if k not in txt]
    add("A18", "3.5", "Config file with model names, hyper-parameters, random seeds and split settings", not miss, f"missing keys: {miss}")
    add("A19", "3.6", "Demonstration and reproduction instructions (predict / verify commands documented)",
        "spsbert predict" in readme and "spsbert verify" in readme)

    # ---------------- report (3.6)
    pdf = Path(report_pdf) if report_pdf else out / "Group7_Final_Report.pdf"
    if not pdf.exists():
        C.append(Check("A20", "3.6", "Final report PDF with GitHub link on the first page", "PENDING", f"{pdf} not generated yet"))
    elif shutil.which("pdftotext"):
        txt = subprocess.run(["pdftotext", "-f", "1", "-l", "1", str(pdf), "-"], capture_output=True, text=True).stdout
        import re
        placeholder = bool(re.search(r"<[^>\s]+>|your-username|your-repo", txt))
        ok = bool(re.search(r"github\.com/[\w.-]+/[\w.-]+", txt)) and not placeholder
        add("A20", "3.6", "Final report PDF with a real GitHub link on the first page", ok, "placeholder URL still on page 1: rebuild the report with your real URL" if placeholder else "")
    else:
        C.append(Check("A20", "3.6", "Final report PDF with GitHub link on the first page", "WARN", "pdftotext unavailable; check manually"))
    return C


def format_checks(checks: list) -> str:
    lines = []
    for c in checks:
        lines.append(f"[{c.status:7s}] {c.id} (§{c.section:7s}) {c.requirement}" + (f"\n{'':11s}-> {c.detail}" if c.detail and c.status != "PASS" else ""))
    n = {s: sum(c.status == s for c in checks) for s in ("PASS", "FAIL", "WARN", "PENDING")}
    lines.append(f"\nSUMMARY: {n['PASS']} PASS, {n['FAIL']} FAIL, {n['WARN']} WARN, {n['PENDING']} PENDING")
    return "\n".join(lines)

#!/usr/bin/env python
"""Check the final report PDF against the repository's own outputs (needs poppler: pdftotext, pdfinfo, pdfimages, pdffonts).

    python scripts/check_report_consistency.py [--repo .] [--pdf report/Group7_Final_Report.pdf]

Verifies that every number in the report equals the stored results, that README and the executed notebook agree with the
report, and that the first page and the required sections meet guideline 3.4 and 3.6. Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PS = "Investigate the effectiveness of SBERT-based sentence embeddings for story point estimation within the same project using random splitting."
MEMBERS = ["Atluri Venkata Sai Vignesh Chowdary", "Adimulam Yaswanth Veera Nagesh", "M V Gowrish", "Kukkala Sai Dinesh Reddy", "Munukuntla Rithvik Reddy"]
ROLLS = ["B230857CS", "B230755CS", "B230673CS", "B231035CS", "B231114CS"]
SECTIONS = ["1. Problem statement and objectives", "2. Dataset", "3. Methodology", "4. Experimental setup and reproducibility", "5. Results",
            "6. Discussion", "7. Limitations and threats to validity", "8. Conclusions", "9. Demonstration and reproduction", "References",
            "Appendix A", "Appendix B"]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--pdf", default="report/Group7_Final_Report.pdf")
    a = ap.parse_args()
    repo, pdf = Path(a.repo), Path(a.repo) / a.pdf
    if not shutil.which("pdftotext"):
        print("poppler (pdftotext) is required"); return 2
    R = repo / "results" / "tables"
    flat = " ".join(run(["pdftotext", str(pdf), "-"]).split())
    pages = run(["pdftotext", str(pdf), "-"]).split("\f")
    mac = pd.read_csv(R / "macro_summary.csv"); test = pd.read_csv(R / "test_results.csv"); ex = json.load(open(R / "analysis_extra.json"))
    meta = json.load(open(repo / "results" / "run_metadata.json")); clean = json.load(open(repo / "results" / "data_cleaning_report.json"))
    first = " ".join(pages[0].split())
    res: list = []
    add = lambda name, ok, detail="": res.append((name, bool(ok), detail))  # noqa: E731
    f3 = lambda v: f"{v:.3f}"  # noqa: E731

    add("page 1: title, exact problem statement, abstract, brief answer", "Effectiveness of SBERT-based Sentence Embeddings" in first and PS in first and "Abstract." in first and "Answer in brief." in first)
    add("page 1: all 5 members with roll numbers", all(n in first for n in MEMBERS) and all(r in first for r in ROLLS))
    m = re.search(r"https://github\.com/[\w.-]+/[\w.-]+", first)
    add("page 1: real GitHub link (no placeholder)", m and not re.search(r"<[^>]+>|your-username", first), m.group(0) if m else "")
    add("contents page lists all sections", all(s.split(".")[0] in flat for s in SECTIONS) and "Contents" in flat)
    heads = [s for s in SECTIONS if s not in flat]
    add("required sections present (problem, methodology, experiments, results, conclusions)", not heads, f"missing: {heads}")
    add("5.1 validation reported separately from 5.2 test", "5.1 Five-fold validation results" in flat and "5.2 Final held-out test results" in flat)
    add("guideline 3.4 table (Representation / Model configuration / MAE / MdAE)", "required by guideline 3.4" in flat and "Combined BERT + SBERT embeddings" in flat)
    add("discussion answers 'lowest MAE and MdAE' and states limitations", "Which representation achieves the lowest MAE and MdAE" in flat and "Limitations and threats to validity" in flat)
    bad = [f"{r.label} {f3(r.test_mae)}/{f3(r.test_mdae)}" for r in mac.itertuples() if not (f3(r.test_mae) in flat and f3(r.test_mdae) in flat)]
    add("macro test MAE/MdAE of every method appear in the PDF", not bad, str(bad))
    badv = [r.label for r in mac.itertuples() if not (f3(r.val_mae) in flat and f3(r.val_mdae) in flat)]
    add("macro validation MAE/MdAE of every method appear in the PDF", not badv, str(badv))
    p5 = test[test.method == "sbert_cnn"]
    miss = [r.project for r in p5.itertuples() if f"{r.mae:.2f} / {r.mdae:.2f}" not in flat]
    add("Table 5: SBERT+CNN MAE / MdAE of all 39 projects equal results", not miss and len(p5) == 39, str(miss[:3]))
    miss = [r.project for r in test.itertuples() if f"{r.mae:.2f} / {r.mdae:.2f}" not in flat]
    add("Table 5: every method x project cell equals results", not miss, f"{len(miss)} cells differ")
    ci = [r.project for r in p5.itertuples() if f"{r.mae:.2f} [{r.mae_ci_lo:.2f}, {r.mae_ci_hi:.2f}]" not in flat]
    add("Appendix A: confidence intervals equal results", not ci, str(ci[:3]))
    pv = ex["primary_vs"]["median_baseline"]
    add("win/tie/loss and Friedman statistics equal analysis_extra.json", re.search(rf"Median {pv['wins']} {pv['ties']} {pv['losses']}", flat) and f"{ex['friedman']['chi2']:.2f}" in flat)
    add("dataset facts: projects, issues, outlier policy", f"{ex['n_projects']} projects" in flat and f"{clean['rows_final']}" in flat.replace(",", "") and str(clean["dropped_above_max_story_point"]) in flat)
    add("project-name mapping note, instructor confirmation, 2 inferred names, Appendix B", "confirmed by e-mail" in flat and all(x in flat for x in clean["inferred_name_mappings"]) and "Table B1" in flat)
    add("reproducibility facts: encoder, seed, config hash, repeats", meta["encoder"] in flat and meta["config_hash"] in flat and "three times" in flat)
    imgs = len(run(["pdfimages", "-list", str(pdf)]).strip().splitlines()) - 2
    add("figures embedded (>= 5)", imgs >= 5, f"{imgs} images")
    add("no placeholders or leaks (<tags>, nan, None, inf, NOT REAL, SYNTHETIC)", not re.search(r"<[a-z/]+>|\bnan\b|\bNone\b|\binf\b|NOT REAL|SYNTHETIC|your-username", flat))
    info = run(["pdfinfo", str(pdf)]); npages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    add("metadata (title, author) and page count 10-25", "Group 7" in info and 10 <= npages <= 25, f"{npages} pages")
    fonts = [ln.split() for ln in run(["pdffonts", str(pdf)]).splitlines()[2:] if ln.strip()]
    unemb = [f[0] for f in fonts if f[-5] != "yes"]
    add("all text fonts embedded (DejaVu); only ReportLab's unused empty default Helvetica reference may remain", all(u.startswith("Helvetica") for u in unemb) and len(unemb) <= 1 and len(fonts) >= 4, f"not embedded: {unemb}")
    add("footer 'Group 7' on every page", all("Group 7" in p for p in pages if p.strip()))
    rd = (repo / "README.md").read_text(encoding="utf-8")
    rows = {mm.group(1).strip(): (mm.group(2), mm.group(3)) for mm in re.finditer(r"^\| ([^|]+) \| (\d\.\d{3}) \| (\d\.\d{3}) \|", rd, re.M)}
    bad = [r.label for r in mac.itertuples() if rows.get(r.label) != (f3(r.test_mae), f3(r.test_mdae))]
    add("README results table equals results", not bad and len(rows) == len(mac), str(bad))
    nbp = repo / "notebooks" / "Group7_PS5_SBERT_StoryPoints.ipynb"
    nbt = ""
    if nbp.exists():
        for c in json.loads(nbp.read_text(encoding="utf-8")).get("cells", []):
            for o in c.get("outputs", []):
                nbt += "".join(o.get("text", [])) if o.get("output_type") == "stream" else "".join(o.get("data", {}).get("text/plain", []))
    best = mac.sort_values("test_mae").iloc[0]
    add("executed notebook prints the same lowest MAE / MdAE", f"{best.label} = {f3(best.test_mae)}" in nbt.replace("\n", " ") and "REPRODUCIBLE" in nbt, "")
    for name, ok, d in res:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {d}" if (not ok and d) else ""))
    nf = sum(not ok for _, ok, _ in res)
    print(f"REPORT CONSISTENCY: {len(res) - nf}/{len(res)} PASS" + ("" if not nf else f", {nf} FAIL"))
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())

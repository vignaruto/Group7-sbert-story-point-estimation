#!/usr/bin/env python
"""Generate notebooks/Group7_PS5_SBERT_StoryPoints.ipynb (shipped WITHOUT outputs: your own run produces them)."""
from pathlib import Path

import nbformat as nbf

import base64
import io
import zipfile

MD, CODE = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
ROOT_DIR = Path(__file__).resolve().parents[1]


def embedded_bundle() -> str:
    """Base64 zip of the runtime code, so the notebook also works when uploaded on its own (e.g. Kaggle)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for pat in ("src/spsbert/*.py", "config/*", "scripts/*.py", "tests/*.py", ".github/workflows/*.yml", "docs/*.md", "docs/guideline_matrix.csv", "docs/guideline_text.txt"):
            for f in sorted(ROOT_DIR.glob(pat)):
                z.write(f, f.relative_to(ROOT_DIR).as_posix())
        for f in ("pyproject.toml", "requirements.txt", "README.md", ".gitignore"):
            z.write(ROOT_DIR / f, f)
    b = base64.b64encode(buf.getvalue()).decode()
    return "\n".join(b[i:i + 110] for i in range(0, len(b), 110))

cells = []

cells.append(MD("""# Group 7 · PS-5 — SBERT sentence embeddings for story point estimation (within-project, random split)

**Problem statement.** *Investigate the effectiveness of SBERT-based sentence embeddings for story point estimation within the same project using random splitting.*

| # | Name | Roll No. |
|---|---|---|
| 1 | Atluri Venkata Sai Vignesh Chowdary | B230857CS |
| 2 | Adimulam Yaswanth Veera Nagesh | B230755CS |
| 3 | M V Gowrish | B230673CS |
| 4 | Kukkala Sai Dinesh Reddy | B231035CS |
| 5 | Munukuntla Rithvik Reddy | B231114CS |

**How this notebook works.** It is a guided front-end: all logic lives in the tested package `src/spsbert/` (single source of truth; see `docs/NOTEBOOK_VS_SCRIPTS.md`). Run the cells **top to bottom**. Every output required by `Project_Guidelines.pdf` (§3.1–3.7) appears below, and the last section audits them automatically.

**Before you start.** (1) Create `data/tawos_issues.csv` (README §3.2). (2) **Run All as it is = the FINAL run** (`config/default.yaml`, all 39 projects, about 80 minutes on a Kaggle T4 GPU; it stops early if the machine is too slow). For a quick 2-project trial instead, edit the two commented lines in Section 1 and Section 6. (3) Use a GPU runtime on Colab. A run is resumable: finished projects are cached and reused."""))

cells.append(MD("""### Embedded project code (you can ignore the next cell)
It carries a compressed copy of the project's Python package, configs and docs, so **this notebook works on its own** (for example on Kaggle with only this notebook and the TAWOS dataset added). If the repository folder is found next to the notebook, that folder is used instead."""))
cells.append(CODE('EMBEDDED_REPO_B64 = """\n' + embedded_bundle() + '\n"""\nprint("Embedded code bundle:", len(EMBEDDED_REPO_B64) // 1000, "kB")'))
cells.append(MD("## 0. Setup"))
cells.append(CODE('''import os, sys, subprocess, inspect, warnings
from pathlib import Path
WARN_LOG = []   # every warning of the run is collected here and summarised at the end
warnings.simplefilter("always")
warnings.filterwarnings("ignore", category=ResourceWarning)        # kernel-internal noise (unclosed zmq sockets), not our code
warnings.filterwarnings("ignore", message=".*utcnow.*")             # jupyter_client deprecation, not our code
warnings.showwarning = lambda message, category, *a, **k: WARN_LOG.append(f"{category.__name__}: {message}")

# locate the repository root (works locally and in Colab after cloning/uploading the repo)
cwd = Path.cwd()
roots = [cwd, Path("/kaggle/input"), Path("/content")]                                # local, Kaggle, Colab
candidates = [cwd, *cwd.parents] + [h.parent.parent for r in roots if r.exists() for pat in ("*/src/spsbert", "*/*/src/spsbert", "src/spsbert") for h in r.glob(pat)]
ROOT = next((p for p in candidates if (p / "src" / "spsbert").exists()), None)
if ROOT is None and "EMBEDDED_REPO_B64" in globals():      # standalone notebook: unpack the embedded code
    import base64, io, zipfile
    target = (Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()) / "spsbert_repo"
    if not (target / "src" / "spsbert").exists():
        zipfile.ZipFile(io.BytesIO(base64.b64decode(EMBEDDED_REPO_B64))).extractall(target)
    ROOT = target
    print("Repository folder not found next to the notebook: using the code embedded in this notebook ->", target)
if ROOT is None:
    raise RuntimeError("Repository not found. Upload/unzip or git-clone it first (it contains src/spsbert), "
                       "then run this notebook from inside it (on Colab: %cd /content/<repo-folder>).")
if not os.access(ROOT, os.W_OK):             # Kaggle mounts inputs read-only: work on a writable copy
    import shutil
    work = Path("/kaggle/working/repo") if Path("/kaggle/working").exists() else Path.cwd() / "repo_work"
    if not work.exists():
        shutil.copytree(ROOT, work)
    ROOT = work
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
IN_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
try:      # optional Hugging Face token (Kaggle: Add-ons > Secrets > HF_TOKEN): faster, rate-limit-free model download
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

IN_COLAB = "google.colab" in sys.modules
INSTALL_MISSING = IN_COLAB or IN_KAGGLE      # installs ONLY missing packages (never upgrades preinstalled ones); Kaggle needs Internet ON
if INSTALL_MISSING:
    import importlib.util
    PIP = {"sentence_transformers": "sentence-transformers", "reportlab": "reportlab", "yaml": "PyYAML", "scipy": "scipy",
           "sklearn": "scikit-learn", "matplotlib": "matplotlib", "pandas": "pandas", "numpy": "numpy"}
    missing = [pip for mod, pip in PIP.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print("Installing missing packages:", missing)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
pd.set_option("display.max_columns", 60); pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 120); pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
print("Repository root:", ROOT)'''))

cells.append(MD("""## 1. Configuration, seeds and environment (guideline §3.1, §3.7)
All hyper-parameters, random seeds, split settings and model names live in the YAML config (single source of truth). The environment table records library versions and hardware."""))
cells.append(CODE('''import yaml
from spsbert.config import load_config
from spsbert.utils import environment_info, set_seed, setup_logging

CONFIG_PATH = os.environ.get("SPSBERT_CONFIG", "config/default.yaml")   # FINAL RUN (default, ~80 min on a Kaggle T4).
# CONFIG_PATH = "config/quick.yaml"   # <- quick trial instead (uncomment, and set MAX_PROJECTS = 2 in Section 6)
OVERRIDES = os.environ.get("SPSBERT_OVERRIDES", "").split()             # e.g. ["split.n_folds=5"]

USE_GOOGLE_DRIVE = False   # Colab only: True keeps results + embedding cache on your Drive, so a disconnect loses nothing
if USE_GOOGLE_DRIVE:
    if not IN_COLAB:
        print("USE_GOOGLE_DRIVE only applies on Colab; ignored.")
    else:
        from google.colab import drive
        drive.mount("/content/drive")
        OVERRIDES += ["output_dir=/content/drive/MyDrive/spsbert_group7/results",
                      "embedding.cache_dir=/content/drive/MyDrive/spsbert_group7/cache"]

cfg = load_config(CONFIG_PATH, OVERRIDES)
Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
setup_logging("INFO", Path(cfg.output_dir) / "run.log")
set_seed(cfg.seed)

print(f"Config file: {CONFIG_PATH}   |   fingerprint: {cfg.fingerprint()}   |   master seed: {cfg.seed}")
print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
display(pd.Series(environment_info(), name="value").to_frame())
from spsbert.doctor import make_bundle, prepare_data, run_doctor
print("PREFLIGHT (fix every FAIL before continuing; WARN = slower or risky but possible):")
DOCTOR = run_doctor(cfg.data.csv_path)
display(pd.DataFrame(DOCTOR).set_index("check"))
if not any(r["check"] == "GPU (CUDA)" and r["status"] == "PASS" for r in DOCTOR):
    print("\\nACTION REQUIRED - no usable GPU. A short trial still works, but the full run needs a GPU (about 20x faster).")
    print("    Kaggle: Settings > Accelerator > GPU T4 x2, then Run All again (do NOT choose P100 if the table above says FAIL).")
if cfg.embedding.backend != "sbert":
    print("WARNING: embedding.backend is not 'sbert' -> this is a smoke test; its results are NOT valid for submission.")'''))

cells.append(MD("""## 2. Dataset description and preprocessing (guideline §3.5)
TAWOS issues with a story point; text = *title. description* after light cleaning (code/noformat blocks, Jira macros, HTML, mentions removed; URLs → `URL`); empty texts, duplicate keys and exact duplicate texts removed (duplicates would leak across the split)."""))
cells.append(CODE('''from spsbert.data import build_dataset, clean_text, compose_text, load_raw, project_summary

csv_path = Path(cfg.data.csv_path)
prepare_data(csv_path)           # finds tawos_issues.csv or converts a TAWOS .sql/.zip dump found in data/, /kaggle/input, /content
if not csv_path.exists():
    if IN_COLAB:                                   # convenient upload prompt on Colab
        from google.colab import files
        print(f"{csv_path} not found - please upload your tawos_issues.csv now")
        uploaded = files.upload()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_bytes(next(iter(uploaded.values())))
    else:
        raise FileNotFoundError(f"{csv_path} not found. Put the TAWOS download (.sql / .zip / .sql.gz) or tawos_issues.csv into the "
                                "data/ folder (Kaggle: add it as an input dataset). An SQL dump is converted automatically. "
                                "See docs/RUN_GUIDE.md.")

df, clean_report = build_dataset(cfg.data)
print("Cleaning report:")
display(pd.Series(clean_report, name="value").to_frame())
summary = project_summary(df)
display(summary)

DEMO_PROJECT = summary.sort_values("n_issues").iloc[0]["project"]       # smallest usable project -> fast demos
BIGGEST = summary.sort_values("n_issues").iloc[-1]["project"]
print("Demo project:", DEMO_PROJECT)

fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
summary.set_index("project")["n_issues"].plot(kind="bar", ax=ax[0], color="#4c72b0")
ax[0].set_title("Usable issues per project"); ax[0].set_ylabel("issues"); ax[0].set_xlabel("")
sp = df[df.project == BIGGEST]["story_point"]
vals, cnt = np.unique(sp, return_counts=True)
ax[1].bar(range(len(vals)), cnt, color="#dd8452"); ax[1].set_xticks(range(len(vals))); ax[1].set_xticklabels([f"{v:g}" for v in vals], rotation=60)
ax[1].set_title(f"Story-point distribution: {BIGGEST}")
ax[1].set_xlabel("story points"); ax[1].set_ylabel("issues")
plt.tight_layout(); plt.show()'''))
cells.append(CODE('''# Preprocessing example: raw text versus cleaned model input
raw = load_raw(cfg.data.csv_path)
raw_counts = raw.groupby("project").size().sort_values(ascending=False)
print(f"{len(raw_counts)} projects in the CSV (rows before cleaning). Use these exact names to fix any 'not found' warning above:")
display(raw_counts.head(60).to_frame("rows"))
for _, r in raw[raw.project == DEMO_PROJECT].head(3).iterrows():
    print("RAW  :", (str(r["title"]) + " | " + str(r["description"]))[:260].replace("\\n", " "))
    print("CLEAN:", compose_text(clean_text(r["title"]), clean_text(r["description"]))[:260])
    print()'''))

cells.append(MD("""## 3. Random 80:20 split and five-fold cross-validation (guideline §3.2)
20 % of the issues are held out as the final test set; the 80 % development part is divided into five folds (four train, one validates, rotating). Hard assertions guarantee disjointness and exhaustive coverage."""))
cells.append(CODE('''from spsbert.splits import assert_valid_split, make_split

dfp = df[df.project == DEMO_PROJECT].sort_values("issue_key", kind="mergesort").reset_index(drop=True)
split = make_split(len(dfp), cfg.split.test_size, cfg.split.n_folds, cfg.seed)
assert_valid_split(split, len(dfp))
rows = [{"part": "all issues", "issues": len(dfp)},
        {"part": f"development ({1 - cfg.split.test_size:.0%})", "issues": len(split.dev_idx)},
        {"part": f"held-out test ({cfg.split.test_size:.0%})", "issues": len(split.test_idx)}]
for k, (tr, va) in enumerate(split.folds, 1):
    rows.append({"part": f"fold {k}: train / validation", "issues": f"{len(tr)} / {len(va)}"})
display(pd.DataFrame(rows))
print("test ∩ development =", set(split.test_idx.tolist()) & set(split.dev_idx.tolist()), "(must be empty)")
print("Each development issue validates exactly once:", sorted(np.concatenate([va for _, va in split.folds]).tolist()) == sorted(split.dev_idx.tolist()))'''))

cells.append(MD("""## 4. SBERT embedding extraction (guideline §3.1)
Each issue text is encoded once by a frozen Sentence-BERT model into one vector (cached on disk). Because the encoder is frozen and label-free, encoding before splitting leaks no labels."""))
cells.append(CODE('''from spsbert.embeddings import EmbeddingStore, build_encoder

encoder = build_encoder(cfg.embedding)
store = EmbeddingStore(encoder, cfg.embedding)
emb = store.get(dfp["text"].tolist())
print(f"Encoder: {encoder.name}")
print(f"Embedding matrix: shape={emb.shape}, dtype={emb.dtype}, mean L2 norm={np.linalg.norm(emb, axis=1).mean():.3f}")
E = emb / np.linalg.norm(emb, axis=1, keepdims=True)
sims = E @ E[0]; sims[0] = -np.inf; j = int(sims.argmax())
print(f"\\nIssue 0           : {dfp.text[0][:150]}")
print(f"Most similar issue: {dfp.text[j][:150]}   (cosine = {E[0] @ E[j]:.3f})")'''))

cells.append(MD("""## 5. Prediction component: three-layer CNN, and the baselines (guideline §3.1)
The CNN treats the embedding as a one-channel 1-D signal: three `Conv1d → BatchNorm → ReLU → Pool` blocks, then an MLP head. The baselines (Mean, Median, TF-IDF+Ridge, SBERT+Ridge) make "effectiveness" measurable and are evaluated on identical splits."""))
cells.append(CODE('''from spsbert.models import build_cnn_module, build_method_specs

c = cfg.cnn
net = build_cnn_module(emb.shape[1], c.channels, c.kernel_size, c.pool_size, c.adaptive_pool, c.fc_hidden, 0.3, c.pool_type)
print(net)
print("\\nTrainable parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))
specs = build_method_specs(cfg)
display(pd.DataFrame([{"method": s.name, "label": s.label, "features": s.feature, "grid size": len(s.grid), "final seeds": s.seeds}
                      for s in specs.values()]))
print("CNN hyper-parameter grid (selected by 5-fold CV on development data only):", c.grid)
display(Markdown("```python\\n" + inspect.getsource(build_cnn_module) + "\\n```"))'''))

cells.append(MD("""## 6. Run the experiments (all projects)
For each project: 5-fold CV model selection on the development data → refit on all development data → **one** evaluation on the held-out test set. Resumable: cached projects are reused (cache key = config hash + data hash)."""))
cells.append(CODE('''from spsbert.aggregate import aggregate
from spsbert.experiment import run_all

MAX_PROJECTS = None   # FINAL RUN: all projects.
# MAX_PROJECTS = 2    # <- quick trial instead: only the 2 smallest projects
BATCH = None          # free-tier session limits: e.g. (0, 10) then (10, 20) ... in separate sessions, merged afterwards (see docs/RUN_GUIDE.md)
ranked = summary.sort_values(["n_issues", "project"]).project.tolist()
only = ranked[BATCH[0]:BATCH[1]] if BATCH else (ranked[:MAX_PROJECTS] if MAX_PROJECTS else None)
n_sel = summary[summary.project.isin(only or ranked)].n_issues.sum()
from itertools import product
sec_per_epoch_1000 = next((r["value"] for r in DOCTOR if r["check"] == "speed benchmark" and r.get("value")), None)
fits = len(list(product(*cfg.cnn.grid.values()))) * cfg.split.n_folds + len(cfg.cnn.final_seeds)     # CNN fits per project
est_h = sec_per_epoch_1000 * 30 * fits * n_sel / 1000 / 3600 if sec_per_epoch_1000 else None          # assumes ~30 epochs per fit
print(f"Selected {len(only or ranked)} of {len(ranked)} projects ({n_sel} issues); {fits} CNN fits per project with {CONFIG_PATH}.")
print(f"Rough CNN time estimate on THIS machine: ~{est_h:.1f} h (assumes ~30 epochs per fit; baselines add little)." if est_h else "No speed estimate available.")
REQUIRE_FAST_MACHINE = True   # stops early when this machine would need more than 6 h (e.g. no GPU); set False to run anyway
if REQUIRE_FAST_MACHINE and est_h and est_h > 6:
    raise RuntimeError(f"Estimated {est_h:.1f} h on this machine. Enable a GPU (Kaggle: Settings > Accelerator > GPU T4 x2) and Run All again, or set REQUIRE_FAST_MACHINE = False.")
if est_h and est_h > 8:
    print("WARNING: this may not fit one session. Enable a GPU, or use BATCH runs + merge, or CONFIG_PATH = 'config/balanced.yaml'.")
results = run_all(cfg, only=only)          # may take long on the full config; safe to re-run after a disconnect
agg = aggregate(cfg.output_dir)
folds, val, test, macro = agg["folds"], agg["val"], agg["test"], agg["macro"]
PRIMARY = "sbert_cnn"
print(f"\\nFinished: {len(results)} project(s). Tables and figures are in {cfg.output_dir}/")'''))

cells.append(MD("""## 7. Five-fold validation results — reported separately from the test results (guideline §3.4)
Validation metrics come from the development data only and are summarised as mean ± standard deviation over the five folds."""))
cells.append(CODE('''print("SBERT+CNN: validation MAE of every fold")
fm = folds[folds.method == PRIMARY].pivot(index="project", columns="fold", values="mae"); fm.columns = [f"MAE fold {k}" for k in fm.columns]
fd = folds[folds.method == PRIMARY].pivot(index="project", columns="fold", values="mdae"); fd.columns = [f"MdAE fold {k}" for k in fd.columns]
display(pd.concat([fm, fd], axis=1))

pretty = val.assign(MAE=lambda d: d.mae_mean.map("{:.3f}".format) + " ± " + d.mae_std.map("{:.3f}".format),
                    MdAE=lambda d: d.mdae_mean.map("{:.3f}".format) + " ± " + d.mdae_std.map("{:.3f}".format))
print("\\nValidation MAE (mean ± std over 5 folds), all methods")
display(pretty.pivot(index="project", columns="label", values="MAE"))
print("Validation MdAE (mean ± std over 5 folds), all methods")
display(pretty.pivot(index="project", columns="label", values="MdAE"))
print("Selected SBERT+CNN hyper-parameters per project")
display(val[val.method == PRIMARY].set_index("project")[["best_params"]])'''))

cells.append(MD("""## 8. Final held-out test results (guideline §3.4)
The selected configuration is refitted on all development data and scored once on the untouched 20 % test set (CNN: mean over the final seeds)."""))
cells.append(CODE('''order = [l for l in ("Mean", "Median", "TF-IDF+Ridge", "SBERT+Ridge", "SBERT+CNN") if l in set(test.label)]
print("Test MAE per project")
display(test.pivot(index="project", columns="label", values="mae")[order])
print("Test MdAE per project")
display(test.pivot(index="project", columns="label", values="mdae")[order])
print("SBERT+CNN test metrics with 95 % bootstrap confidence intervals and seed variability")
display(test[test.method == PRIMARY].set_index("project")[["n_test", "mae", "mae_ci_lo", "mae_ci_hi", "mae_std_seeds",
                                                           "mdae", "mdae_ci_lo", "mdae_ci_hi", "mdae_std_seeds"]])
print("Macro-average over projects (sorted by test MAE) and number of projects where each method is best")
display(macro)'''))

cells.append(MD("""## 9. Comparative table prescribed by the guideline (§3.4)
*Representation · Model configuration · MAE · MdAE.* Only the SBERT row belongs to PS-5; BERT and combined embeddings are other groups' problem statements. Per-project versions are in `tables/assignment_table.csv`."""))
cells.append(CODE('''display(Markdown((Path(cfg.output_dir) / "tables" / "assignment_table.md").read_text(encoding="utf-8")))'''))

cells.append(MD("## 10. Plots (guideline §3.5)"))
cells.append(CODE('''for fn in ("test_mae_per_project", "test_mdae_per_project", "improvement_vs_median", "validation_fold_mae_boxplot", "pred_vs_true_pooled"):
    f = Path(cfg.output_dir) / "figures" / f"{fn}.png"
    if f.exists():
        print(fn); display(Image(filename=str(f)))'''))

cells.append(MD("""## 11. Statistical analysis
Paired two-sided Wilcoxon signed-rank tests on per-issue absolute errors (same test issues), raw and **Holm-corrected** across projects; cross-project Wilcoxon on per-project MAE. Treat p-values as descriptive when projects are few or small."""))
cells.append(CODE('''cols = ["project", "n_test", "p_vs_median", "p_vs_median_holm", "p_vs_tfidf", "p_vs_tfidf_holm"]
display(test[test.method == PRIMARY].set_index("project")[cols[1:]])
print("Cross-project Wilcoxon (SBERT+CNN vs median, per-project test MAE):", agg["cross_project_wilcoxon"])
print("Rank test, win/tie/loss and correlations:", agg["extra"])
imp = agg["improvement"]
if not imp.empty:
    print("\\nMAE change vs the median baseline (%, positive = better), macro mean per method:")
    display(imp.groupby("method")[["mae_improvement_pct", "mdae_improvement_pct"]].mean())'''))

cells.append(MD("## 12. Discussion (guideline §3.4)"))
cells.append(CODE('''m = macro.set_index("method")
best_mae, best_mdae = m["test_mae"].idxmin(), m["test_mdae"].idxmin()
print(f"Lowest macro test MAE : {m.loc[best_mae, 'label']} = {m.loc[best_mae, 'test_mae']:.3f}")
print(f"Lowest macro test MdAE: {m.loc[best_mdae, 'label']} = {m.loc[best_mdae, 'test_mdae']:.3f}")
if "median_baseline" in m.index and PRIMARY in m.index:
    gain = 100 * (m.loc["median_baseline", "test_mae"] - m.loc[PRIMARY, "test_mae"]) / m.loc["median_baseline", "test_mae"]
    print(f"SBERT+CNN vs median baseline (macro test MAE): {gain:+.1f} %")
if "sbert_ridge" in m.index and PRIMARY in m.index:
    print(f"SBERT+Ridge {m.loc['sbert_ridge', 'test_mae']:.3f}  vs  SBERT+CNN {m.loc[PRIMARY, 'test_mae']:.3f}  (does the non-linear CNN add value?)")
if "tfidf_ridge" in m.index and "sbert_ridge" in m.index:
    print(f"TF-IDF+Ridge {m.loc['tfidf_ridge', 'test_mae']:.3f}  vs  SBERT+Ridge {m.loc['sbert_ridge', 'test_mae']:.3f}  (semantic vs lexical features)")'''))
cells.append(MD("""**Your interpretation — edit before submitting.** Use the numbers printed above and the tables/plots; do not claim more than they show.

* *Which representation/method has the lowest MAE and MdAE, and by how much? Is the gap larger than the confidence intervals / significant after Holm correction?*
* *Why might a constant (median) predictor be hard to beat? (see "mode share" in the dataset table)*
* *Does the CNN beat Ridge on the same SBERT vectors? Why might that be (sample size, parameters vs data)?*

**Limitations and threats to validity** (guideline §3.4): random splitting lets the model see issues created after the test issues (temporal leakage); one random split per project gives noisy small-project estimates; only exact duplicates are removed (near-duplicates may remain); story points are team-specific and noisy; the frozen encoder truncates long texts and is not tuned for effort; per-project p-values are numerous (hence Holm); projects differ widely, so macro-averages hide variation."""))

cells.append(MD("""## 13. Demonstration and reproducibility check (guideline §3.6)
`predict` retrains the selected configuration of a finished project and scores new issue texts; `verify` recomputes a whole project from scratch and compares it with the stored results."""))
cells.append(CODE('''import torch
from spsbert.demo import predict_texts, verify_project

DEMO_TEXTS = ["Fix typo in the label of the save button",
              "Redesign the plugin architecture and migrate all modules to the new concurrency model",
              "Add a configuration option to disable the weekly report email"]
preds = predict_texts(cfg.output_dir, DEMO_PROJECT, DEMO_TEXTS, csv_path=cfg.data.csv_path)
display(pd.DataFrame({"issue text": DEMO_TEXTS, "predicted story points": preds}))

tol = 1e-3 if torch.cuda.is_available() else 1e-6          # GPUs may differ at floating-point noise level
r = verify_project(cfg.output_dir, DEMO_PROJECT, csv_path=cfg.data.csv_path, tol=tol)
display(pd.DataFrame(r["rows"]))
print("REPRODUCIBLE: recomputation matches the stored results" if r["all_match"] else "NOT REPRODUCED - see mismatches")'''))

cells.append(MD("""## 14. Strict audit against the assignment
Automated check of this run and the repository against `Project_Guidelines.pdf` §3.1–3.7 (it recomputes metrics from saved predictions, re-derives model selection from the CV grid, checks folds/splits, README, configs, plots). **A09/A15 fail on purpose for smoke tests; the report check (A20) stays PENDING until you build the report.** For a valid submission everything else must be PASS."""))
cells.append(CODE('''from spsbert.check import run_checks

checks = run_checks(cfg.output_dir, ROOT)
display(pd.DataFrame([c.__dict__ for c in checks]).set_index("id"))
n = {s: sum(c.status == s for c in checks) for s in ("PASS", "FAIL", "WARN", "PENDING")}
print("SUMMARY:", n)
if n["FAIL"]:
    print("FAILED:", [c.id for c in checks if c.status == "FAIL"])'''))

cells.append(MD("## 15. Package the outputs\nCreates `<results-folder>_upload_<MMDD_HHMM>.zip` (tables, figures, logs, metadata) and `<results-folder>_diagnostics_bundle_<MMDD_HHMM>.zip` (warnings, preflight, logs) at the top level of `/kaggle/working` (or next to the results folder elsewhere). Download both and send them with the executed notebook; a third zip, `Group7_github_repo_<MMDD_HHMM>.zip`, is the ready-to-push GitHub repository (add the executed notebook to its `notebooks/` folder); the final report is then built with `python -m spsbert report --github-url <your repo URL>`."))
cells.append(CODE('''import shutil
print(f"Warnings during the run: {len(WARN_LOG)} ({len(set(WARN_LOG))} distinct)")
for w in sorted(set(WARN_LOG))[:15]:
    print(" -", w[:200])
out_dir = Path(cfg.output_dir).resolve()
DEST = Path("/kaggle/working") if Path("/kaggle/working").exists() else out_dir.parent      # Kaggle shows top-level files of /kaggle/working
import datetime
STAMP = datetime.datetime.now().strftime("%m%d_%H%M")      # time stamp: newer files are never confused with older runs
bundle = make_bundle(cfg.output_dir, DEST / f"{out_dir.name}_diagnostics_bundle_{STAMP}.zip", WARN_LOG, DOCTOR)
archive = shutil.make_archive(str(DEST / f"{out_dir.name}_upload_{STAMP}"), "zip", out_dir)
import zipfile
SKIP = {"data", "cache", "__pycache__", ".ipynb_checkpoints", ".virtual_documents"}
github_zip = DEST / f"Group7_github_repo_{STAMP}.zip"      # ready-to-push repository (no data, no cache, no old trial results)
with zipfile.ZipFile(github_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(ROOT.rglob("*")):
        rel = f.relative_to(ROOT)
        if f.is_dir() or SKIP & set(rel.parts) or f.suffix == ".zip" or (rel.parts[0].startswith("results") and rel.parts[0] != out_dir.name):
            continue
        z.write(f, rel.as_posix())
print("\\nFILES TO DOWNLOAD (send the first two here; the third is the repository to push to GitHub):")
print("  1.", archive)
print("  2.", bundle)
print("  3.", github_zip)
print("Kaggle: click the refresh icon next to /kaggle/working in the Output panel, or open Save Version > Output tab.")
print("Send ONLY these two (they carry today's date-time). Ignore older zips from earlier runs, e.g. any file with 'quick' in its name.")
if IN_COLAB:
    from google.colab import files
    files.download(archive)'''))

cells.append(MD("## 16. Final report PDF (do this last)\nCreate the GitHub repository, paste its URL below, and run the cell: it builds `Group7_Final_Report.pdf` from the results of this run (GitHub link on page 1, all tables, figures, statistics and the discussion). Read the generated discussion and add your own interpretation before submitting."))
cells.append(CODE('''GITHUB_URL = ""    # e.g. "https://github.com/<user>/<repo>"
if GITHUB_URL.startswith("http"):
    from spsbert.report import build_report
    pdf = build_report(cfg.output_dir, Path(cfg.output_dir) / "Group7_Final_Report.pdf", GITHUB_URL)
    shutil.copy(pdf, DEST / "Group7_Final_Report.pdf")
    print("Report written:", DEST / "Group7_Final_Report.pdf")
else:
    print("Set GITHUB_URL above, then run this cell again to build the final report PDF.")'''))

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from notebook_text import apply_text  # noqa: E402

apply_text(cells)      # final wording of markdown cells and why-comments of code cells
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}, "accelerator": "GPU"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "Group7_PS5_SBERT_StoryPoints.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out, f"({len(cells)} cells)")

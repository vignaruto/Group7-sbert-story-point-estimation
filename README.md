# Story Point Estimation with SBERT Sentence Embeddings — Within-Project, Random Split

**NLP course project · Group 7 · Problem statement PS-5**

> *Investigate the effectiveness of SBERT-based sentence embeddings for story point estimation within the same project using random splitting.*

| # | Name | Roll No. |
|---|------|----------|
| 1 | Atluri Venkata Sai Vignesh Chowdary | B230857CS |
| 2 | Adimulam Yaswanth Veera Nagesh | B230755CS |
| 3 | M V Gowrish | B230673CS |
| 4 | Kukkala Sai Dinesh Reddy | B231035CS |
| 5 | Munukuntla Rithvik Reddy | B231114CS |

> **Run it:** `docs/RUN_GUIDE.md` (Kaggle / Colab / own PC, file placement, Run-All steps, team plan). Preflight: `python -m spsbert doctor`. Guideline traceability, word by word: `docs/guideline_matrix.csv`; your requests: `docs/REQUEST_TRACEABILITY.md`; weights check: `docs/WEIGHTED_SELF_ASSESSMENT.md`, `docs/PS_SELECTION_CHECK.md`.

## 1. Objective

Predict the story points of a software issue from its text (title + description). We encode each issue with a **frozen Sentence-BERT** model, train a **three-layer CNN** on top, and evaluate with **MAE** and **MdAE** in the **within-project** setting (train and test issues from the same project) with a **random** 80:20 split and **five-fold** cross-validation on the 80 % development part (guideline §3.2). To make "effectiveness" meaningful, the CNN is compared with four baselines evaluated on identical partitions: *Mean*, *Median*, *TF-IDF + Ridge*, *SBERT + Ridge*.

## 1b. Dataset description

**TAWOS** (<https://doi.org/10.5522/04/21308124>; Tawosi et al., MSR 2022) is a public dataset of issues mined from open-source projects that use agile practices and are tracked in Jira. Many issues carry a **story point** estimate, which is our prediction target. The guideline (§3.3) names the 39 projects of the study (listed in `config/guideline_projects.txt`).

| Field used | Role |
|---|---|
| `Project_Name` | defines the within-project setting (each project is modelled separately) |
| `Issue_Key` | unique id; issues are sorted by it so splits do not depend on row order |
| `Title`, `Description_Text` | model input: text = *title. description* after light cleaning |
| `Story_Point` | regression target (positive numbers only) |
| `Creation_Date` | kept as metadata; **not used** (random splitting ignores time) |

**Outlier policy (important):** the real TAWOS story-point field contains data-entry errors (raw maximum 26,710,114; 11 of 39 projects exceed 100). Story points **above 100** (the top of the planning-poker deck) are removed (`data.max_story_point: 100`); the count per project is in `results/data_cleaning_report.json` and must be stated in the report. Set it to `null` to keep everything (MAE values then explode for some projects).

Exact per-project counts, story-point distributions and cleaning statistics are produced by the run (`results/tables/dataset_summary.csv`, `results/data_cleaning_report.json`) and shown in the notebook.

## 1c. Results at a glance (final run)

Final run on **all 39 TAWOS projects** (62,532 usable issues; 141 issues with story points above 100 removed as data-entry errors), frozen SBERT model `sentence-transformers/all-mpnet-base-v2`, seed 42, one Tesla T4 GPU. Held-out test results, macro-averaged over the projects (lower is better):

| Method | Test MAE | Test MdAE | Projects with lowest MAE |
|---|---|---|---|
| TF-IDF+Ridge | 1.962 | 1.158 | 21 |
| SBERT+Ridge | 2.017 | 1.227 | 10 |
| SBERT+CNN | 2.063 | 1.199 | 2 |
| Median | 2.165 | 1.372 | 6 |
| Mean | 2.442 | 1.913 | 0 |

* **Which representation is lowest?** By macro MAE and MdAE the best method overall is **TF-IDF+Ridge** (a lexical baseline). Among the SBERT-based models, **SBERT+Ridge** has the lowest MAE (2.017) and **SBERT+CNN** the lowest MdAE (1.199).
* SBERT + CNN improves on the median baseline by +4.7 % MAE and is better in 28 of 39 projects (Wilcoxon p < 0.001); a Friedman test over the projects rejects equal performance (p < 0.001). Its advantage grows with project size (Spearman rho = +0.55).
* **Reproducibility:** three independent GPU runs on three different days produced bit-identical metrics (390 test and 975 validation values, identical splits and predictions).
* Per-fold validation results, per-project tables, plots and statistics: `results/`. The final report PDF (same tables, discussion, limitations) is added to `report/`.
* Two guideline project names ("Evergreen", "C++ driver (Appcelerator)") have no exact TAWOS match; they were mapped by elimination (`config/project_aliases.csv`). All 39 TAWOS projects are evaluated separately, so this affects only labels, not results.

## 2. Repository layout

```
config/default.yaml          all hyper-parameters, seeds, split settings, model names (the single source of truth)
config/quick.yaml            fast real-SBERT trial (MiniLM, tiny grid) to validate your setup first
config/smoke_test.yaml       OFFLINE pipeline check on synthetic data (not SBERT, not for reporting)
config/guideline_projects.txt  the 39 project names from the guideline (§3.3)
config/project_aliases.csv     maps those short names to the full TAWOS names (2 inferred, see file)
src/spsbert/                 data.py  splits.py  embeddings.py  models.py  metrics.py
                             experiment.py  aggregate.py  report.py  config.py  cli.py ...
scripts/export_tawos_to_csv.py   TAWOS (MySQL) -> data/tawos_issues.csv
tests/                       78 unit, adversarial, mutation, SQL-parser, doctor and notebook tests (pytest)
docs/GUIDELINE_COMPLIANCE.md every guideline requirement -> code -> test
.github/workflows/tests.yml  CI (pyflakes + pytest)   ·   Makefile  shortcuts
notebooks/Group7_PS5_SBERT_StoryPoints.ipynb  guided notebook (thin front-end; runs the same tested package)
results/                     FINAL RUN: per-project JSON, predictions, split files, tables, figures, logs, metadata
report/                      final report PDF (added once the notebook and repository are final)
data/                        empty on purpose: the TAWOS data is not redistributed (see section 3.2)
```

## 3. Quick start

### 3.1 Install
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest -q                                              # expect: 78 passed   (Python >= 3.10)
```

### 3.2 Data (TAWOS)
Download TAWOS from <https://doi.org/10.5522/04/21308124> (a MySQL dump, possibly zipped), then create `data/tawos_issues.csv` by **either** route:

**Route A — no MySQL needed (recommended, works on Colab):**
```bash
python -m spsbert sql2csv path/to/TAWOS.sql --out data/tawos_issues.csv      # also accepts .sql.gz and .zip
```
It streams the dump (about 90 s per GB on a slow CPU), keeps only issues that have a story point and joins the project names.

**Route B — via MySQL:** `mysql -u root -p -e "CREATE DATABASE tawos"`, `mysql -u root -p tawos < TAWOS.sql`, then
`python scripts/export_tawos_to_csv.py --user root --database tawos --out data/tawos_issues.csv`.

The CSV needs the columns `Project_Name, Issue_Key, Title, Description_Text, Creation_Date, Story_Point` (common aliases are accepted; missing required columns raise a clear error that lists the columns found). Both routes assume the TAWOS tables `Issue` and `Project` with the usual column names; they are tested on mysqldump-format fixtures, but **could not be verified against the real TAWOS file** in our development sandbox. If one fails, the error message names the missing column; adjust the alias lists in `src/spsbert/sqldump.py` (route A) or `QUERY` in the export script (route B).
After loading, the run warns about any name in `config/guideline_projects.txt` that is not in your CSV and suggests the closest names present.

### 3.2b Colab walkthrough (GPU)
1. Runtime ▸ Change runtime type ▸ **GPU (T4)**. Upload and unzip the repo: `!unzip -q sp-sbert-within-project.zip` (Files panel), then open `notebooks/Group7_PS5_SBERT_StoryPoints.ipynb`.
2. In Section 1 of the notebook set `USE_GOOGLE_DRIVE = True` (keeps results and the embedding cache on your Drive, so a disconnect loses nothing; re-running the notebook resumes).
3. Section 2 asks you to upload `tawos_issues.csv` if it is missing (create it from the dump with Route A in a first cell: `!python -m spsbert sql2csv /content/TAWOS.sql`).
4. Try `CONFIG_PATH = "config/quick.yaml"` first; switch to `config/default.yaml` for the full run. The last cell produces `results_upload.zip`.

### 3.3 Run
```bash
python -m spsbert run --config config/quick.yaml       # optional 1st trial: small SBERT, finishes quickly
python -m spsbert run --config config/default.yaml     # the full experiment (all-mpnet-base-v2)
python -m spsbert report --github-url https://github.com/<you>/<repo>   # builds results/Group7_Final_Report.pdf
```
* Resumable: finished projects are cached (keyed by a hash of the configuration); `--force` recomputes.
* Subset: `--projects "Moodle" "Mule"` · Override: `--set split.n_folds=5 embedding.batch_size=32`.
* A GPU is strongly recommended (use the Colab notebook). Measured on a single slow CPU core: ~2.8 s per training epoch for a 2,400-issue project (768-d), i.e. roughly 45 minutes per such project for the default grid (33 CNN fits); a GPU should be much faster (not measured by us). On CPU the default grid is slow for large projects; use `config/quick.yaml`, or reduce `cnn.grid`/`cnn.max_epochs`/`cnn.final_seeds`.

### 3.3b Notebook (recommended front-end) and strict audit
`notebooks/Group7_PS5_SBERT_StoryPoints.ipynb` walks through every output the assignment requires (config/seeds → dataset → split → embeddings → CNN → 5-fold validation → test → comparison table → plots → statistics → discussion → demo → audit) by calling the same tested package; it is shipped **without outputs**, so the numbers you see are your own run. Why a thin notebook and not "all code in one notebook": `docs/NOTEBOOK_VS_SCRIPTS.md`. Run it on Colab (GPU) or locally; choose the config in its Section 1.

```bash
python -m spsbert check            # strict audit of results/ + repo against guideline 3.1-3.7 (exit 1 on any FAIL)
```
A run on synthetic data or the hashing encoder **fails on purpose** (A09/A15); everything else must be PASS before you submit (A20 = report, built last).

### 3.4 Demonstration and reproducibility check (guideline §3.6)
```bash
# Demo: retrain the selected configuration of a finished project and predict NEW issue texts
python -m spsbert predict --project "Moodle" --text "Add dark mode toggle to the settings page" "Migrate the database layer to a new ORM"

# Reproduce: recompute one project from scratch and compare with the stored results (exit code 0 = identical)
python -m spsbert verify --project "Moodle"
```
On CPU the recomputation is bit-identical (verified in our tests); on a different GPU/CUDA version expect differences only at floating-point noise level (use `--tol 1e-3`).

### 3.5 Outputs (`results/`)
| Path | Content |
|---|---|
| `tables/validation_per_fold.csv` | every fold's MAE/MdAE per project and method (guideline: validation reported separately) |
| `tables/validation_summary.csv` | mean ± std over folds, chosen hyper-parameters |
| `tables/test_results.csv` | final test MAE/MdAE, 95 % bootstrap CI, seed std, Wilcoxon p-values (raw and Holm-corrected) |
| `tables/macro_summary.csv/.md` | project-averaged comparison table |
| `figures/*.png` | per-project bars, improvement vs median, fold box-plot, predicted-vs-true |
| `splits/*.json` | issue keys of dev/test sets and the validation fold of every dev issue (share with other groups for paired comparisons) |
| `predictions/*.csv` | per-issue test predictions |
| `run_metadata.json`, `run.log` | config, seeds, library versions, hardware |
| `Group7_Final_Report.pdf` | final report (GitHub link on page 1) |

## 4. Method in detail

1. **Cleaning** (`data.py`): keep issues with a positive numeric story point; text = *title. description*; strip `{code}`/`{noformat}` blocks, Jira macros, HTML tags, user mentions; URLs → `URL`; drop empty texts, duplicate keys and **exact duplicate texts** (they would otherwise sit on both sides of a split and inflate scores); skip projects with < `min_issues` issues.
2. **Split** (`splits.py`): issues sorted by key (row-order independent), `train_test_split` 80:20 (seeded), then `KFold(5, shuffle)` on the development part. Hard assertions guarantee disjointness, full coverage and that every development issue is validated exactly once.
3. **Embeddings** (`embeddings.py`): `sentence-transformers/all-mpnet-base-v2` (768-d, L2-normalised, max 384 tokens), encoded once and cached on disk. The encoder is frozen and label-free, so there is no label leakage from encoding before splitting.
4. **CNN** (`models.py`): three `Conv1d → BatchNorm → ReLU → AvgPool` blocks (16/32/64 channels, kernel 5), `Dropout → Linear(64) → ReLU → Dropout → Linear(1)`. log1p-transformed, standardised target; Huber loss; Adam; early stopping on a 10 % hold-out of the **training part only** (so validation folds and the test set never influence epochs). Predictions are clipped to the training range.
5. **Model selection** (guideline §3.2: four folds train, one validates; the CNN's early-stopping hold-out is a 10 % slice taken *from those four training folds*, so the validation fold is never used for early stopping): grid over dropout × weight-decay (6 settings) by mean 5-fold validation MAE (tie → MdAE) on development data only; the winner is refit on all development data with 3 seeds and scored **once** on the test set.
6. **Statistics** (`metrics.py`): 95 % bootstrap CIs; paired Wilcoxon signed-rank on per-issue absolute errors vs the median baseline and TF-IDF+Ridge; macro-averages across projects; cross-project Wilcoxon on per-project MAE.

### Design decisions worth knowing
* **Why baselines?** An MAE number alone says nothing about "effectiveness". Story points are discrete and often concentrated on a few values, so the *median* predictor can be hard to beat on MdAE.
* **CNN design.** During development a CNN with aggressive max-pooling failed to learn dense signals that Ridge learned easily; we therefore keep the whole feature map (`adaptive_pool: 0`) and use average pooling, and add stronger regularisation to the CV grid. On small projects Ridge on SBERT vectors may still match or beat the CNN — if so, report it honestly; that is a valid finding.
* **No test-set peeking**: scaling, TF-IDF fitting, early stopping and hyper-parameter choice all use training/development data only.

## 5. Reproducibility
Seeds (`seed`, `cnn.final_seeds`), split settings, model name and hyper-parameters live in `config/*.yaml`; `run_metadata.json` stores library versions and hardware. Split issue keys are stored in `results/splits/`. Seeding covers Python, NumPy and PyTorch with deterministic algorithms enabled; bit-exact equality across *different hardware/CUDA versions* is not guaranteed, but results should agree to within normal floating-point noise.

## 6. Testing
`pytest -q` runs 78 tests: metric formulas (hand-computed), Holm correction (known values), split invariants, cleaning rules, column aliases, config validation, CNN architecture (exactly three conv layers) and learning, embedding cache, SBERT wiring (stand-in library), SQL export, an end-to-end pipeline, and adversarial checks: a **leakage spy** that records every index each model is fitted/scored on (the test set must never be touched before the final evaluation), TF-IDF vocabulary from training data only, **cache invalidation when the data changes**, constant-target projects, file-name collisions, two independent runs being identical, the CLI, the PDF report, the demo and `verify`, **mutation tests proving the strict audit really catches tampered results**, and an end-to-end execution of the notebook.
`python -m spsbert synthetic && python -m spsbert run --config config/smoke_test.yaml` exercises the whole pipeline offline on **synthetic** data with a non-SBERT encoder. The report it produces is stamped *"NOT REAL RESULTS"*.

## 7. Limitations
Random splitting leaks temporal information; one split per project gives noisy small-project estimates; near-duplicate (not exact) issues may remain; story points are team-specific; the encoder is frozen and truncates long texts; per-project p-values are uncorrected. See Section 7 of the generated report.

## 8. Submission checklist (guideline §3.5–3.6)
- [x] Real TAWOS data converted; full run finished on all 39 projects (`results/`)
- [x] Strict audit `python -m spsbert check` passes (only the report check is pending)
- [x] Run reproduced three times with identical numbers
- [x] Executed notebook in `notebooks/`, code in `src/`, settings in `config/`
- [ ] Final report PDF with this repository's URL on page 1 (`report/`)
- [ ] Every member can explain the pipeline (section 4), the results and the limitations (`docs/VIVA_GUIDE.md`)

## 9. References
Reimers & Gurevych (2019) *Sentence-BERT*; Tawosi et al. (2022) *A Versatile Dataset of Agile Open Source Software Projects* (TAWOS); Choetkiertikul et al. *Deep-SE*; Song et al. (2020) *MPNet*. Verify citation details before submission.

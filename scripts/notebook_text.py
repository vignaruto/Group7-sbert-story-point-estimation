"""Human-facing text of the notebook, kept apart from the build logic.

Every section follows the same pattern: WHAT the cell does, WHY it is done that way, and WHAT to expect. Code cells get a
short comment block that explains the reason for the step (not the syntax).
"""

MEMBERS = [("Atluri Venkata Sai Vignesh Chowdary", "B230857CS"), ("Adimulam Yaswanth Veera Nagesh", "B230755CS"),
           ("M V Gowrish", "B230673CS"), ("Kukkala Sai Dinesh Reddy", "B231035CS"), ("Munukuntla Rithvik Reddy", "B231114CS")]

INTRO = """# Story point estimation with SBERT sentence embeddings
### Group 7 · PS-5 — within-project estimation, random split

**Assigned problem statement.** *Investigate the effectiveness of SBERT-based sentence embeddings for story point estimation within the same project using random splitting.*

| # | Name | Roll No. |
|---|---|---|
""" + "\n".join(f"| {i} | {n} | {r} |" for i, (n, r) in enumerate(MEMBERS, 1)) + """

**What this notebook does.** For each of the 39 TAWOS projects *separately*, it encodes every issue (title + description) with a frozen SBERT model, trains a three-layer CNN to predict the story points, chooses hyper-parameters by five-fold cross-validation on 80 % of the issues (the development set), and reports **MAE** and **MdAE** on the untouched 20 % test set. Four baselines (mean, median, TF-IDF + Ridge, SBERT + Ridge) are scored on identical splits, so "effective" always has a reference point.

**How to run it** (about 80 minutes on a Kaggle T4 GPU; the defaults below *are* the final run):
1. Attach the TAWOS SQL dump as an input dataset (Kaggle: *Add Input*). It is found and converted automatically.
2. Settings → **Accelerator: GPU T4 x2**, **Internet: On**.
3. **Run All.** The notebook stops early, with a clear message, if the machine would be too slow. A run is resumable and fully reproducible.

**Where each requirement of `Project_Guidelines.pdf` is addressed**

| Requirement | Section |
|---|---|
| §3.1 SBERT sentence-level representations; documented configuration, architecture, hyper-parameters, seeds | 1, 4, 5 |
| §3.2 80:20 split; five folds, each fold validates once; selection on development data only | 3, 6 |
| §3.4 MAE and MdAE; five-fold validation reported separately from test; mean ± standard deviation | 7, 8 |
| §3.4 the required "Representation / Model configuration / MAE / MdAE" table | 9 |
| §3.4 which representation is lowest; limitations and threats to validity | 12 |
| §3.5 plots and result files | 10, 15 |
| §3.6 demonstration and reproduction of the principal results | 13 |
| §3.7 record seeds, splits, versions; no fabricated results | 1, 14 |
| §3.6 final report with the GitHub link on page 1 | 16 |

**Reading the output.** Tables are the evidence; the audit in Section 14 labels every check `PASS`, `WARN`, `FAIL` or `PENDING`. A `WARN` means "state this in the report" (for example a reduced configuration), a `FAIL` means "fix before submitting". Rerunning with the same configuration and data reproduces every number exactly."""

MARKDOWN = {
"### Embedded": """It carries a compressed copy of the project's Python package, configuration files and documentation, so this notebook can run **on its own** (for example on Kaggle with only the notebook and the TAWOS dataset attached). If a repository folder is found next to the notebook, that folder is used and the embedded copy is ignored. The code cell below is intentionally long; there is nothing to edit in it.""",

"## 0. Setup": """**What.** Find (or unpack) the project code, install only the packages that are missing, and start collecting warnings.

**Why.** The notebook has to run unchanged on Kaggle, Colab and a laptop. Kaggle mounts inputs read-only, so the code is copied to a writable folder. Installing *only missing* packages keeps the platform's own, tested PyTorch and NumPy. Every warning raised during the run is collected and summarised in Section 15 instead of being silently lost.

**Expected output.** One line with the repository root.""",

"## 1. Configuration": """**What.** Load every setting from `config/default.yaml`, fix the random seeds, record library versions and hardware, and run the preflight checks.

**Why.** One configuration file is the single source of truth for hyper-parameters, seeds, split sizes and model names (guideline §3.1 and §3.7). Its fingerprint is stored with the results, so a result can always be traced to the exact settings that produced it. The preflight table finds setup problems (missing GPU, no internet, data not found) in seconds instead of after an hour.

**Expected output.** The configuration, an environment table, and a preflight table in which `GPU (CUDA)` should be `PASS` for the final run. The notebook stops early if the estimated run time is unrealistic for the machine.""",

"## 2. Dataset": """**What.** Load the TAWOS issues that carry a story point, clean the text, and describe the data per project.

**Why, rule by rule.**
- *Text* = title + description; code blocks, markup, user mentions and URLs are removed because they are noise for a sentence encoder.
- *Exact duplicate texts* inside a project are removed: otherwise the same issue could sit in both the training and the test set and inflate the scores.
- *Outlier policy.* The raw story-point field contains data-entry errors (the maximum is 26,710,114). Story points **above 100**, the top of the planning-poker scale, are removed; the number removed per project is printed and saved. This decision is not specified by the assignment, so it is stated explicitly in the report.
- *Project names.* The assignment uses short names ("XD", "Nexus"); TAWOS uses full names ("Spring XD", "Sonatype Nexus"). `config/project_aliases.csv` maps them. Two mappings were inferred by elimination; because every TAWOS project is evaluated separately, this only affects labels, never numbers.

**Expected output.** A cleaning report, a per-project summary table, and two plots (issues per project; story-point distribution of the largest project).""",

"## 3. Random 80:20": """**What.** Show, on the smallest project, exactly how the data is divided: 80 % development / 20 % held-out test, and the development part into five folds.

**Why.** The test set is reserved for the final evaluation only (guideline §3.2): it is never used to fit parameters, choose hyper-parameters or stop training. The code *asserts* that development and test sets do not overlap and that every development issue is validated exactly once, so a leak would stop the run instead of passing unnoticed. The same logic runs for every project in Section 6.

**Expected output.** A size table, an empty overlap set, and `True` for full fold coverage.""",

"## 4. SBERT embedding": """**What.** Turn every issue text into one 768-number vector with a frozen SBERT model, and show an example of semantic similarity.

**Why frozen?** Most projects have only a few hundred issues; fine-tuning a 110-million-parameter encoder on them would overfit. The encoder never sees the story points, so encoding all issues once, before splitting, leaks no label information. Vectors are cached on disk.

**Is this strictly SBERT?** Yes. SBERT (*Sentence-BERT*; Reimers & Gurevych, 2019) is the method that fine-tunes a transformer in a siamese setup so that a whole text maps to one fixed-size vector; the `sentence-transformers` library implements it. We use the library's recommended general-purpose model, `sentence-transformers/all-mpnet-base-v2` (MPNet backbone, trained contrastively on more than a billion sentence pairs). The assignment asks for "SBERT-based sentence embeddings", not for one backbone, so any `sentence-transformers` model qualifies; the model name is one line in the configuration. For the original BERT-backbone checkpoint of the SBERT paper, use `sentence-transformers/bert-base-nli-mean-tokens` (older and weaker) and re-run.

**Expected output.** The encoder name, the matrix shape `(issues, 768)`, a mean vector length of 1.000 (the vectors are L2-normalised), and the most similar issue to issue 0.""",

"## 5. Prediction component": """**What.** Build the three-layer CNN and the baselines, and list the hyper-parameter grid.

**Architecture.** The embedding is treated as a one-channel signal and passed through three blocks of *Conv1d → BatchNorm → ReLU → AvgPool*, then *Dropout → Linear → ReLU → Dropout → Linear(1)*. Story points are log-transformed and standardised (they are heavily skewed), the loss is Huber, the optimiser is Adam, and training stops early on a 10 % hold-out taken from the *training part only*. Predictions are clipped to the training range.

**Why baselines?** An MAE value alone does not show whether a model is "effective". The median predictor shows whether the text carries any signal; TF-IDF + Ridge shows whether *semantic* embeddings beat plain word counts; SBERT + Ridge shows whether the CNN's extra capacity pays off.

**Expected output.** The network printout with its parameter count, a table of the five methods, the grid, and the source of the CNN builder.""",

"## 6. Run the experiments": """**What.** For every project: split → five-fold cross-validation to choose the CNN settings (development data only) → refit on all development data → score the test set once.

**Why this order.** Choosing settings on the validation folds and touching the test set only once is what makes the reported test numbers unbiased (guideline §3.2 and §3.7). Results are cached per project (configuration hash + data hash), so an interrupted run resumes instead of restarting.

**Expected output.** A time estimate for *this* machine, one progress line per project, and the aggregated tables and figures. About 80 minutes on a T4.""",

"## 7. Five-fold validation": """**What.** The validation results of the selected configuration: every fold, then mean ± standard deviation over the five folds.

**Why separate from the test results.** Validation scores guided the choice of settings, so they are slightly optimistic; the held-out test scores in Section 8 are the unbiased estimate (guideline §3.4).

**Expected output.** A per-fold MAE/MdAE table for SBERT + CNN, mean ± std tables for all methods, and the selected hyper-parameters per project.""",

"## 8. Final held-out": """**What.** The test results: MAE and MdAE per project and method, 95 % bootstrap confidence intervals for SBERT + CNN, and the macro average over projects.

**How to read it.** Lower is better. MAE is sensitive to large errors; MdAE shows the typical error. The macro average gives every project the same weight, so one large project cannot dominate.""",

"## 9. Comparative table": """**What.** The table the assignment prescribes: representation, model configuration, MAE, MdAE.

**Why only the SBERT row is filled.** The BERT and combined rows belong to other problem statements (PS-1 to PS-4 and PS-9 to PS-12). The stored split files allow a paired comparison with those groups on identical partitions.""",

"## 10. Plots": """**What.** Per-project MAE and MdAE (one row per project, one marker per method), the change against the median baseline, the spread of fold-wise validation errors, and predicted versus true story points.""",

"## 11. Statistical analysis": """**What.** Paired Wilcoxon tests on per-issue errors (raw and Holm-corrected across projects), a cross-project Wilcoxon test, a Friedman rank test over the 39 projects, win/tie/loss counts, and which project properties go with SBERT's gains.

**Why.** A lower average error can be chance. These tests say how much of the difference is systematic. They describe these 39 projects; they are not proof about software projects in general.""",

"## 12. Discussion": """**What.** The numbers needed to answer the assignment's question: which representation achieves the lowest MAE and MdAE.

**Why computed, not typed.** Every statement is derived from the result tables above, so the text can never claim more than was measured (guideline §3.7: report actual findings, do not claim unsupported improvements).""",

"**Your interpretation": """**Your interpretation — write this in your own words before submitting.** Use the numbers printed above and do not claim more than they show.

- Which method has the lowest MAE and MdAE, and by how much? Is the gap larger than the confidence intervals, and does it survive the Holm correction?
- How does SBERT + CNN compare with the median baseline, with SBERT + Ridge, and with TF-IDF + Ridge? What could explain the order (sample size, a strong constant baseline, team-specific scales)?
- In which projects does SBERT + CNN lose to the median baseline, and what do those projects have in common?

**Limitations and threats to validity.** Random splitting lets the model see issues created after the test issues (temporal leakage), so scores are optimistic compared with real deployment. One random split per project gives noisy estimates for small projects. Only exact duplicate texts are removed. The outlier cap (story points above 100) is a modelling decision the assignment does not specify. Story points are team-specific and noisy. The frozen encoder truncates long texts and is not tuned for effort. The 39 projects are not a random sample of software projects.""",

"## 13. Demonstration": """**What.** (1) Predict the story points of three new issue texts with the model selected for the smallest project. (2) Recompute that whole project from scratch and compare it with the stored results.

**Why.** The guideline asks for a demonstration and for reproducible principal results (§3.6, §3.7). On a CPU the recomputation is bit-identical; on a GPU the tolerance is 0.001.

**Expected output.** A table of predictions and the line `REPRODUCIBLE`.""",

"## 14. Strict audit": """**What.** An automatic audit of this run and of the repository against `Project_Guidelines.pdf` §3.1–3.7, plus integrity checks (for example, it recomputes MAE from the saved predictions and re-derives the model selection from the cross-validation grid).

**Reading it.** For a valid final submission everything is `PASS` except `A20` (the report), which stays `PENDING` until Section 16 has been run with your GitHub URL. `A21` is a `WARN` if a reduced configuration was used; `A22` is a `WARN` if fewer than all 39 projects were evaluated. Both must be `PASS` for the final run.""",

"## 15. Package": """**What.** Summarise the warnings of the run and create three files at the top level of `/kaggle/working`:
1. `<results-folder>_upload_<MMDD_HHMM>.zip` — tables, figures, logs and metadata (needed for the report);
2. `<results-folder>_diagnostics_bundle_<MMDD_HHMM>.zip` — warnings, preflight table and logs (for finding problems);
3. `Group7_github_repo_<MMDD_HHMM>.zip` — the ready-to-push GitHub repository (no data, no cache).

**Why time-stamped.** Older files from earlier runs can never be confused with the newest ones. Refresh the Output panel (or open *Save Version → Output*) if the files do not appear.""",

"## 16. Final report": """**What.** Builds `Group7_Final_Report.pdf` from the results of this run: problem, methodology, experiments, tables, figures, statistics, discussion, limitations, conclusions, and the GitHub link on page 1 (guideline §3.6).

**Why last.** The report needs the final numbers and the real repository link on page 1. Create the GitHub repository first, paste its URL below, and run the cell. Then read the discussion and add your own interpretation. The audit (`A20`) rejects a placeholder link.""",
}

CODE_NOTES = {
"import os, sys, subprocess": "# Why: locate the code in every environment (local, Kaggle, Colab) and never fail on a read-only input folder.\n# Warnings are collected (not hidden) so that Section 15 can report them; two kernel-internal ones are ignored on purpose.\n",
"import yaml": "# Why: all settings come from one YAML file, so every number can be traced to the exact configuration that produced it.\n# The preflight below fails fast on setup problems instead of after an hour of computation.\n",
"from spsbert.data import build_dataset": "# Why: cleaning decisions (duplicates, outlier cap, name mapping) change the data, so they are executed in the open and their\n# counts are printed and stored with the results.\n",
"# Preprocessing example": "# Why: a visible before/after example lets a reader check that the cleaning removes noise but keeps the meaning.\n",
"from spsbert.splits import": "# Why: the assertions turn a possible data leak into an immediate error; the test set must stay untouched until the final evaluation.\n",
"from spsbert.embeddings import": "# Why: encode once and cache; the encoder is frozen and never sees story points, so no label information leaks into the split.\n",
"from spsbert.models import": "# Why: printing the architecture and parameter count documents the CNN exactly (guideline §3.1) and shows the capacity relative to the data size.\n",
"from spsbert.aggregate import": "# Why: MAX_PROJECTS / BATCH are debugging switches (None = final run); the estimate and the guard avoid starting a run that cannot finish.\n",
"print(\"SBERT+CNN: validation MAE of every fold\")": "# Why: validation results are shown on their own (guideline §3.4); they guided model selection, so they are not the final estimate.\n",
"order = [l for l in": "# Why: the test set was scored exactly once, after the configuration had been chosen; these are the numbers to report.\n",
"display(Markdown((Path(cfg.output_dir)": "# Why: this is the table format the assignment prescribes; only the SBERT row is ours.\n",
"for fn in (\"test_mae_per_project\"": "# Why: plots make patterns visible that tables hide (outlier projects, where SBERT wins or loses).\n",
"cols = [\"project\", \"n_test\"": "# Why: with 39 projects, some p-values fall below 0.05 by chance; the Holm-corrected columns control that.\n",
"m = macro.set_index(\"method\")": "# Why: the answer to the assignment's discussion question is computed from the tables, never typed by hand.\n",
"import torch\nfrom spsbert.demo": "# Why: the guideline requires a demonstration and reproducible principal results; GPUs may differ in the last floating-point digits.\n",
"from spsbert.check import": "# Why: an independent audit catches mistakes (a placeholder link, a partial run, a missing plot) before submission.\n",
"import shutil": "# Why: time-stamped files cannot be confused with older runs; the third zip is the repository ready to push to GitHub.\n",
"GITHUB_URL =": "# Why: the report needs the real repository URL on page 1 (guideline §3.6), and the audit rejects placeholder links.\n",
}


def apply_text(cells) -> None:
    """Replace the wording of markdown cells and prepend why-comments to code cells (in place)."""
    cells[0].source = INTRO
    for c in cells[1:]:
        src = c.source
        if c.cell_type == "markdown":
            for prefix, body in MARKDOWN.items():
                if src.startswith(prefix):
                    c.source = body if prefix.startswith("**") else src.splitlines()[0] + "\n\n" + body
                    break
        elif c.cell_type == "code" and not src.startswith("EMBEDDED_REPO_B64"):
            for prefix, note in CODE_NOTES.items():
                if src.startswith(prefix):
                    c.source = note + src
                    break

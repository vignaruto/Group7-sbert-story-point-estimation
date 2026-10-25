# What you asked for -> where it is handled

| Your request (from the prompts) | Where / how |
|---|---|
| Plan first, then execute; change plan if needed | done each turn (plans stated first) |
| Check all outputs strictly against the assignment, every word | `docs/guideline_matrix.csv` (all normative sentences, verbatim-checked by a test), `python -m spsbert check` (A00-A20), `docs/GUIDELINE_COMPLIANCE.md` |
| Correct whatever is not correct | audits found and fixed: stale-cache bug, divide-by-zero, missing README dataset section, wrong encoder name in a table, misquote in the matrix |
| Jupyter notebook? Check it is strictly per assignment first, else keep `.py` | `docs/NOTEBOOK_VS_SCRIPTS.md` (comparison + decision: hybrid); `tests/test_notebook.py` runs it end to end and checks every required output |
| Better in `.py` or notebook, compared on all parameters, decide for our situation | same document (decision: tested `.py` package as source of truth + one thin notebook) |
| Notebook to find and fix mistakes, errors, warnings | notebook Section 1 PREFLIGHT (`python -m spsbert doctor`), warning collector, error hints in `docs/RUN_GUIDE.md`, `diagnostics_bundle.zip`, `MAX_PROJECTS`/`BATCH` debug switches |
| Run on Kaggle / Colab / personal computer without outside application | `docs/RUN_GUIDE.md` (platform comparison, setup, Kaggle-style layout rehearsed) |
| Which file goes where; dataset link; start-to-end Run-All steps | `docs/RUN_GUIDE.md` (file tree, dataset link, click-by-click steps) |
| Finish before end of October with a team | team plan table in `docs/RUN_GUIDE.md` |
| Complete code first; report later from your uploaded outputs | `python -m spsbert report` builds the PDF from real results only; A20 stays PENDING until then |
| Best practices, design patterns, principles | strategy/registry (`models.MethodSpec`), single source of truth (YAML), thin notebook, factory per method, atomic writes, config+data hashing, type hints, CI |
| Unbiased, no fabricated results | baselines evaluated on identical splits; report text generated from numbers; synthetic runs stamped and fail the audit (A09/A15) |
| Testing, accuracy, error checking | 70 tests: leakage spy, mutation tests of the audit, SQL parser, notebook execution, merge, determinism |
| Weights used to choose PS-5 (feasibility 20, reproducibility 20, ...) | `docs/WEIGHTED_SELF_ASSESSMENT.md` (evidence + residual risk per parameter), `docs/PS_SELECTION_CHECK.md` (all 12 re-scored; PS-5 #2, tied with PS-6), `config/balanced.yaml` |
| "Just run the notebook on Kaggle, right? What do I attach?" | `docs/RUN_GUIDE.md` sections "Is running the notebook on Kaggle enough" and "What to attach here" |
| Final completeness and accuracy check | `python -m spsbert check`, `python -m spsbert verify`, this table |

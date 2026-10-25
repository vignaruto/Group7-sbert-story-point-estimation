# Jupyter notebook or .py files? — decision record

**Decision: hybrid.** Keep the tested Python package (`src/spsbert/`) as the *single source of truth* that produces every
result, and ship **one thin notebook** (`notebooks/Group7_PS5_SBERT_StoryPoints.ipynb`) that imports that package and walks
through every output the assignment asks for. We deliberately do **not** paste all modules into one notebook.

## Comparison for *this* assignment

| Parameter | `.py` package + CLI | Single big notebook | Hybrid (chosen) |
|---|---|---|---|
| Guideline §3.5 allows | "source code **or** notebooks" — yes | yes | yes |
| Reproducibility (fresh process, no hidden state, config-driven) | **best** | weak: out-of-order cells, stale variables | best (notebook only calls the package) |
| Automated tests / CI (40 tests, leakage spy, CI workflow) | **best** | practically impossible | best |
| Single source of truth / no duplicated code | **best** | a pasted copy would diverge from the tested code | best |
| Long runs (up to 39 projects x 33 CNN fits) and resuming after a Colab disconnect | **best**: results cached per project + data hash | session dies, state lost | good: the same cached `run_all` |
| Visible outputs (tables, plots, logs) in one place for graders and for your uploads | weak (files only) | **best** | **best** |
| Live demonstration (§3.6) and viva explanation | OK (CLI `predict`, `verify`) | **best** | **best** |
| Easy GPU access for students (Colab) | needs a terminal cell | **best** | **best** |
| Code review / git diffs | **best** | noisy JSON diffs | good (notebook is short, shipped without outputs) |
| Maintainability | **best** | poor at ~2,000 lines | best |
| Risk of violating the assignment | none | none, but untestable | none (checked automatically) |

## Why not "everything in one notebook"?
1. It would create a second copy of ~2,000 lines of logic that the 40 tests do not cover.
2. Notebook state (running cells twice or out of order) is a classic source of irreproducible numbers — exactly what §3.7 forbids.
3. A multi-hour GPU run is more robust as a resumable command than as an interactive session.

## Strict-compliance gate
The notebook is only accepted if it (a) runs top to bottom without errors, (b) shows every required output, and (c) ends with
the automated audit `python -m spsbert check`, which verifies the results against guideline §3.1–§3.7 (e.g. recomputes MAE from saved
predictions, re-derives model selection from the CV grid, confirms the five folds). This is enforced by `tests/test_notebook.py`.
If the notebook ever fails this gate, the `.py` package alone remains the deliverable.

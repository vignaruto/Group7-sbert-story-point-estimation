# How to run (start to end) — Kaggle, Colab or your own PC

**Dataset:** <https://rdr.ucl.ac.uk/articles/dataset/The_TAWOS_dataset/21308124> — download the TAWOS file(s) from that page (a MySQL dump, possibly zipped; the TAWOS GitHub page says it is loaded with `mysql ... < file.sql`). You do **not** need MySQL: the notebook converts the dump itself. Check the licence/terms stated on the dataset page before redistributing anything, and cite Tawosi et al., MSR 2022 (the paper named on that page).

## Which platform?
| | Kaggle | Google Colab | Own PC |
|---|---|---|---|
| GPU | free T4/P100 (weekly GPU quota; see Kaggle's account page) | free T4 (limited, can disconnect) | only if NVIDIA + CUDA PyTorch |
| Internet (SBERT download) | **Settings ▸ Internet ON** (phone-verified account) | on | on |
| Session length | up to ~9–12 h | ~12 h max, idle disconnects | unlimited |
| Persistence | `/kaggle/working` saved as notebook output | use `USE_GOOGLE_DRIVE = True` | local disk |
| Best for | **the full run** | quick tests / backup | development; full run only with a GPU |

**Recommendation:** one teammate with the best machine runs on **Kaggle** (full run), others run `config/quick.yaml` on Colab/PC to reproduce and review. Everyone can open the same notebook.

## Where the files go (all platforms)
```
sp-sbert-within-project/            <- the unzipped repo (keep this folder name or any name)
├─ notebooks/Group7_PS5_SBERT_StoryPoints.ipynb   <- open this
├─ data/
│   └─ TAWOS.sql  (or TAWOS.zip / .sql.gz)  OR  tawos_issues.csv      <- put the dataset here
├─ config/  src/  scripts/  tests/  docs/        <- do not move
└─ results/                       <- created by the run
```
The notebook searches `data/`, `/kaggle/input`, `/content` and Drive, and converts the SQL dump to `data/tawos_issues.csv` automatically (a few minutes for a GB-sized dump).

## A) Kaggle (recommended for the full run) — only the notebook + the TAWOS file are needed
The notebook carries its own copy of the code (embedded), so **no code dataset is needed**.
1. **Dataset:** upload the TAWOS file (`TAWOS.sql`, or the zip) as a **private Kaggle dataset** and attach it with **Add Input** (you already did this).
2. **Notebook:** **Create ▸ New Notebook ▸ File ▸ Import Notebook** and upload `Group7_PS5_SBERT_StoryPoints.ipynb` (use the newest file; an old copy that says "Repository not found" lacks the embedded code).
3. **Settings** (top menu ▸ Settings or the right panel): **Accelerator = GPU T4 x2** (a first trial on CPU worked but the full run is ~20x slower there: about 17 minutes per 1,000 issues, ~18 h for all 39 projects; avoid P100 — if the PREFLIGHT table says `GPU (CUDA) FAIL`, switch to T4), **Internet = On** (phone-verified account; SBERT is downloaded from Hugging Face), **Persistence = Files only**. Changing the accelerator restarts the session: then Run All again.
4. The notebook finds the dataset under `/kaggle/input/...` (any depth), converts `TAWOS.sql` to `data/tawos_issues.csv` automatically, and works in `/kaggle/working/spsbert_repo`.
5. **Run All.** The default is now the **FINAL run** (`config/default.yaml`, all 39 projects, about 80 minutes on a T4); it stops early if the machine is too slow. For a quick 2-project trial instead, use the commented lines in Section 1 (`config/quick.yaml`) and Section 6 (`MAX_PROJECTS = 2`).
6. Use **Save Version ▸ Save & Run All (Commit)** so it keeps running even if you close the browser. Download the two time-stamped zips and the executed notebook; finally paste your GitHub URL in Section 16 to build the report PDF.

## B) Google Colab
1. Runtime ▸ Change runtime type ▸ **T4 GPU**. Upload `sp-sbert-within-project.zip` and the TAWOS file via the Files panel (left), then in a cell: `!unzip -q sp-sbert-within-project.zip && mkdir -p sp-sbert-within-project/data && mv TAWOS* sp-sbert-within-project/data/`.
2. Open the notebook from the unzipped folder (or upload it), set `USE_GOOGLE_DRIVE = True` in Section 1 (survives disconnects), choose `config/quick.yaml`, **Runtime ▸ Run all**; when clean switch to `config/default.yaml`.

## C) Your own PC (Windows/macOS/Linux, no outside application)
1. Install **Python 3.10–3.12** (python.org). Unzip the repo, open a terminal in it.
2. `python -m venv .venv` then activate (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` otherwise).
3. `pip install -r requirements.txt jupyterlab` then `pip install -e . --no-deps`. (NVIDIA GPU: install the CUDA build of PyTorch from pytorch.org first.)
4. Put the dataset in `data/`. Check everything: `python -m spsbert doctor` (shows GPU/RAM/disk/internet/data and a speed estimate).
5. `jupyter lab` ▸ open `notebooks/Group7_PS5_SBERT_StoryPoints.ipynb` ▸ **Run ▸ Run All Cells**. On CPU use `config/quick.yaml` and `MAX_PROJECTS = 2` first; a full CPU run may take days — use Kaggle for that.
6. Without Jupyter: `python -m spsbert run --config config/quick.yaml`, `python -m spsbert check`.

## Click-by-click: "Run All" and what to look at
1. Variables you may change: `CONFIG_PATH` (Section 1), `USE_GOOGLE_DRIVE` (Colab), `MAX_PROJECTS` and `BATCH` (Section 6), `GITHUB_URL` (Section 16). The defaults are the final run.
2. **Run All.** Stop at the first red error: copy the error text. The notebook's last cells print the warning summary and create `diagnostics_bundle.zip` — send **that file** (plus the error text) to whoever fixes the code.
3. Optional order: quick trial (commented lines) first, then the default final run. Its audit (Section 14) must show no FAIL; A20 (report) stays PENDING until Section 16 has been run with your GitHub URL.

## If one session is too short (free-tier limits) — run in batches
Section 6 of the notebook prints a **time estimate** for the selected projects. If it is longer than your session (Kaggle ~9–12 h, Colab less), split the 39 projects:
1. Session 1: set `BATCH = (0, 10)`, Run All, download `results_upload.zip`. Session 2: `BATCH = (10, 20)`; session 3: `(20, 30)`; session 4: `(30, 39)`. **Keep the same config file in every session.**
2. Unzip each `results_upload.zip` into its own folder (`batch1/`, `batch2/` ...), then merge and rebuild all tables and figures:
   `python -m spsbert merge --sources batch1 batch2 batch3 batch4 --to results`
   (it refuses to merge runs made with different configurations).
3. Too slow even in batches? Use `CONFIG_PATH = "config/balanced.yaml"` (about 2-3x cheaper, same design) for **every** batch and say so in the report.
4. Run `python -m spsbert check`, then build the report. (You may also choose fewer projects with `projects_file` in the config if your instructor allows it; say so in the report.)

## Common problems
| Symptom | Fix |
|---|---|
| `Repository not found` | open the notebook from inside the repo folder, or add the code dataset (Kaggle) |
| `TAWOS data found: FAIL` | put the `.sql/.zip` or `tawos_issues.csv` in `data/` |
| SBERT download fails / `internet FAIL` | Kaggle: Internet ON; Colab/PC: check connection |
| `CUDA out of memory` | `--set embedding.batch_size=16 cnn.batch_size=16`, or restart runtime |
| Requested projects not found (warning) | use the exact names printed in Section 2 (closest names are suggested) |
| Run too slow | read the speed benchmark in PREFLIGHT; use quick config, fewer `cnn.grid` values, `cnn.final_seeds=[42]` |
| Disconnect mid-run | just Run All again: finished projects are cached |

## Team plan (deadline: end of October 2026)
| When | Who | What |
|---|---|---|
| Oct 22–23 | member with Kaggle/GPU | data prep + `MAX_PROJECTS=2` quick run; send `diagnostics_bundle.zip` for fixes |
| Oct 23–24 | 2 members | repeat quick run on own PC/Colab to confirm reproducibility; review dataset table |
| Oct 24–27 | GPU member | full run (`default.yaml`), upload `results_upload.zip` |
| Oct 27–28 | all | review results/plots, write own discussion, build report (`python -m spsbert report`) |
| Oct 29–30 | all | GitHub push, `python -m spsbert check`, viva practice (`docs/VIVA_GUIDE.md`) |
| Oct 31 | — | buffer, no new changes |

## Is running the notebook on Kaggle enough to finish the assignment?
**It produces all the outputs, but the assignment (guideline §3.5–3.7) also needs:**
1. the **GitHub repository** (push this folder; do not commit the raw data; `.gitignore` already excludes it) — you create it;
2. the **final PDF report** with the GitHub link on page 1 — I build it from your uploaded outputs: `python -m spsbert report --github-url <your repo URL>`;
3. your **own interpretation** in the discussion, and everyone able to explain the work (`docs/VIVA_GUIDE.md`).

## What to attach here after the run (so I can check and build the report)
| File | Where it comes from |
|---|---|
| `results_upload.zip` | last notebook cell (tables, figures, logs, metadata) |
| `diagnostics_bundle.zip` | last notebook cell (warnings, preflight, logs) |
| the **executed notebook** (`.ipynb` with outputs) | Kaggle: the version's Output tab, or File ▸ Download Notebook after the run; Jupyter/Colab: File ▸ Save, then download |
| a screenshot or text of any red error | notebook output |
Large raw data and `cache/` do **not** need to be sent.

## Facts learned from the first real Kaggle trial (your executed notebook)
* The real TAWOS dump converted correctly: 458,232 issues, **65,427 with a story point, 39 projects** (smallest: 209 issues).
* The assignment's short project names ("XD", "Nexus", ...) differ from TAWOS's ("Spring XD", "Sonatype Nexus", ...): `config/project_aliases.csv` maps them, so **all 39 projects** are now selected. Two of the 39 mappings (`Evergreen -> The MongoDB Engineering`, `C++ driver (Appcelerator) -> Aptana Studio`) were inferred by elimination: ask your instructor, and mention it in the report.
* Output files appear at the **top level of `/kaggle/working`** (`<results-folder>_upload_<MMDD_HHMM>.zip`, `<results-folder>_diagnostics_bundle_<MMDD_HHMM>.zip`; send only the newest pair, ignore any file with `quick` in the name from earlier runs); click the refresh icon in the Output panel, or open **Save Version ▸ Output**.
* A trial run reports A21 = WARN (reduced config) and A22 = WARN (2 of 39 projects): both are expected for a trial and must disappear in the final run.
* Harmless messages: `HF_TOKEN` (optional; you can add a Hugging Face token as a Kaggle secret for faster downloads).

## Second real run (GPU T4, default config, 6 projects) — what it showed
* GPU worked (Tesla T4, `all-MiniLM`/`all-mpnet-base-v2` on `cuda`), 0 warnings, audit 21 PASS, `verify` REPRODUCIBLE.
* Real speed: ~15-24 s per ~210-issue project with the default grid. Scaling by issue count, **all 39 projects (62,673 issues) should take roughly 1.5-2 hours on a T4**: one session, no batches needed. (The doctor's speed estimate used to include CUDA start-up; it now warms up first.)
* **Outliers:** the data contain story points like 26,710,114 (Spring DataCass), 6,765 (Titanium SDK), 3,030 (Jira Cloud). They produced validation MAE of 763,177 and Mean-baseline MAE of 155,296 on DataCass. The default config now removes story points above 100 and counts them (`data.max_story_point`). State this in the report.
* With only the 6 smallest projects the median baseline was hardest to beat and SBERT+CNN was -1.8 % vs median; this is a small-sample result, not a conclusion.

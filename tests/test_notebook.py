"""The notebook must be a thin, clean, strictly assignment-aligned front-end that actually runs."""
import ast
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "Group7_PS5_SBERT_StoryPoints.ipynb"

REQUIRED_SECTIONS = ["Configuration, seeds and environment", "Dataset description and preprocessing", "Random 80:20 split and five-fold",
                     "SBERT embedding extraction", "three-layer CNN", "Run the experiments", "Five-fold validation results",
                     "Final held-out test results", "Comparative table prescribed by the guideline", "Plots", "Statistical analysis",
                     "Discussion", "Demonstration and reproducibility", "Strict audit against the assignment"]


@pytest.fixture(scope="module")
def nb():
    return nbformat.read(NB, as_version=4)


def test_notebook_is_valid_clean_and_thin(nb):
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code and not any(o.output_type == "error" for c in code for o in c.outputs)   # outputs may be stored (executed copy), errors never
    for c in code:
        ast.parse(c.source)                                                           # no magics, valid Python
        assert not c.source.lstrip().startswith(("%", "!"))
        assert "\ndef " not in "\n" + c.source and "\nclass " not in "\n" + c.source   # no duplicated logic: only package calls


def test_notebook_covers_every_required_section_in_order(nb):
    heads = " \n".join(c.source.splitlines()[0] for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("##"))
    pos = [heads.find(s) for s in REQUIRED_SECTIONS]
    assert all(p >= 0 for p in pos), [s for s, p in zip(REQUIRED_SECTIONS, pos) if p < 0]
    assert pos == sorted(pos)


def test_notebook_contains_group_identity(nb):
    first = nb.cells[0].source
    assert "Group 7" in first and "PS-5" in first and all(r in first for r in
            ["B230857CS", "B230755CS", "B230673CS", "B231035CS", "B231114CS"])


def test_notebook_executes_and_shows_all_required_outputs(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    from jupyter_client.kernelspec import find_kernel_specs
    if "python3" not in find_kernel_specs():
        pytest.skip("no python3 kernelspec installed")
    from spsbert.synthetic import make_synthetic
    csv = make_synthetic(tmp_path / "d.csv", n_projects=2, sizes=(130, 150))
    monkeypatch.setenv("SPSBERT_CONFIG", str(ROOT / "config" / "smoke_test.yaml"))
    monkeypatch.setenv("SPSBERT_OVERRIDES", f"data.csv_path={csv} output_dir={tmp_path / 'res'} "
                                            f"embedding.cache_dir={tmp_path / 'c'} cnn.max_epochs=6")
    ex = nbformat.read(NB, as_version=4)
    nbclient.NotebookClient(ex, timeout=280, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()

    text, n_img, errors = "", 0, 0
    for c in ex.cells:
        for o in c.get("outputs", []):
            errors += o.output_type == "error"
            if o.output_type == "stream":
                text += o.text
            else:
                d = o.get("data", {})
                n_img += "image/png" in d
                text += d.get("text/plain", "") + d.get("text/markdown", "") + d.get("text/html", "")
    assert errors == 0
    assert n_img >= 5                                                                  # all plots displayed
    for needle in ("MAE fold 1", "MdAE fold 5", "mae_ci_lo", "ALL PROJECTS (macro mean)", "SBERT sentence embeddings",
                   "test ∩ development = set()", "predicted story points", "REPRODUCIBLE", "SUMMARY:", "Lowest macro test MAE"):
        assert needle in text, needle
    gh = list(tmp_path.glob("Group7_github_repo_*.zip"))
    assert gh
    import zipfile
    names = zipfile.ZipFile(gh[0]).namelist()
    assert "src/spsbert/cli.py" in names and ".gitignore" in names and not any(n.startswith(("data/", "cache/")) for n in names)
    assert list(tmp_path.glob("res_upload_*.zip")) and list(tmp_path.glob("res_diagnostics_bundle_*.zip"))   # time-stamped bundles produced
    assert not (ROOT / "results_upload.zip").exists()                                 # tests never pollute the repo


def _embedded_zip(nb):
    import base64, io, re, zipfile
    src = next(c.source for c in nb.cells if c.cell_type == "code" and c.source.startswith("EMBEDDED_REPO_B64"))
    b64 = re.search(r'"""(.*?)"""', src, re.S).group(1)
    return zipfile.ZipFile(io.BytesIO(base64.b64decode(b64)))


def test_embedded_code_bundle_matches_the_repository_exactly(nb):
    """If this fails, run `python scripts/build_notebook.py` (the notebook carries a stale copy of the code)."""
    z = _embedded_zip(nb)
    names = set(z.namelist())
    assert {"src/spsbert/cli.py", "src/spsbert/models.py", "config/default.yaml", "README.md", "requirements.txt"} <= names
    stale = [n for n in names if n.startswith(("src/", "config/", "scripts/")) and z.read(n) != (ROOT / n).read_bytes()]
    assert not stale, stale


def test_notebook_works_standalone_without_the_repository_folder(tmp_path, monkeypatch):
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    from jupyter_client.kernelspec import find_kernel_specs
    if "python3" not in find_kernel_specs():
        pytest.skip("no python3 kernelspec installed")
    lone = nbformat.read(NB, as_version=4)
    k = next(i for i, c in enumerate(lone.cells) if c.cell_type == "markdown" and c.source.startswith("## 1."))
    lone.cells = lone.cells[:k]                                        # intro, embedded code, setup only
    monkeypatch.delenv("SPSBERT_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    nbclient.NotebookClient(lone, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()
    out = "".join(o.get("text", "") for c in lone.cells for o in c.get("outputs", []) if o.output_type == "stream")
    assert "using the code embedded in this notebook" in out
    assert (tmp_path / "spsbert_repo" / "src" / "spsbert" / "cli.py").exists()

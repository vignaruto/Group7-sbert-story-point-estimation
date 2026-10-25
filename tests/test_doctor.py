import zipfile

from spsbert.doctor import find_data_file, make_bundle, prepare_data, run_doctor
from tests.test_sqldump import DUMP


def test_prepare_data_converts_found_sql_dump(tmp_path, monkeypatch):
    sub = tmp_path / "kaggle" / "ds" / "x"
    sub.mkdir(parents=True)
    (sub / "TAWOS.sql").write_text(DUMP, encoding="utf-8")
    monkeypatch.setattr("spsbert.doctor.find_data_file.__defaults__", (None,))
    import spsbert.doctor as d
    orig = d.find_data_file
    monkeypatch.setattr(d, "find_data_file", lambda p, roots=None: orig(p, [tmp_path / "kaggle"]))
    out = prepare_data(tmp_path / "data" / "tawos_issues.csv")
    assert out and out.exists() and "Issue_Key" in out.read_text()


def test_find_data_file_prefers_existing_and_returns_none(tmp_path):
    f = tmp_path / "a.csv"; f.write_text("x")
    assert find_data_file(f) == f
    assert find_data_file(tmp_path / "nope.csv", [tmp_path / "empty"]) is None


def test_doctor_runs_and_bundle(tmp_path):
    rows = run_doctor(tmp_path / "none.csv", bench=False, roots=[tmp_path / "empty"])
    assert any(r["check"] == "TAWOS data found" and r["status"] == "FAIL" for r in rows)
    assert any(r["check"] == "package torch" and r["status"] == "PASS" for r in rows)
    z = make_bundle(tmp_path, tmp_path / "b.zip", ["UserWarning: x", "UserWarning: x"], rows)
    assert {"warnings.txt", "doctor.json"} <= set(zipfile.ZipFile(z).namelist())


def test_guideline_matrix_quotes_exist_verbatim_in_the_assignment_pdf():
    import csv, re
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    norm = lambda t: re.sub(r"\s+", " ", t.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')).lower()  # noqa: E731
    text = norm((root / "docs" / "guideline_text.txt").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(root / "docs" / "guideline_matrix.csv", encoding="utf-8")))
    bad = [r["guideline_quote"] for r in rows if norm(r["guideline_quote"]) not in text]
    assert not bad, bad
    assert not [r for r in rows if r["status"].startswith("MISSING")]


def test_resolve_device_falls_back_to_cpu_when_gpu_unusable(monkeypatch):
    import spsbert.utils as u
    monkeypatch.setattr(u, "cuda_ok", lambda: False)
    assert [u.resolve_device(p) for p in ("auto", "cuda", "cpu")] == ["cpu", "cpu", "cpu"]
    monkeypatch.setattr(u, "cuda_ok", lambda: True)
    assert [u.resolve_device(p) for p in ("auto", "cuda", "cpu")] == ["cuda", "cuda", "cpu"]


def test_logging_setup_silences_chatty_libraries():
    import logging
    from spsbert.utils import setup_logging
    setup_logging("INFO")
    assert logging.getLogger("httpx").level == logging.WARNING and logging.getLogger("huggingface_hub").level == logging.WARNING
    import os
    assert os.environ.get("HF_HUB_VERBOSITY") == "error"

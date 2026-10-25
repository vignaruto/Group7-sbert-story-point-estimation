"""The strict checker must (a) pass a sound run and (b) actually catch tampering - mutation tests."""
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")
from spsbert.aggregate import aggregate
from spsbert.check import run_checks
from spsbert.config import load_config
from spsbert.experiment import run_all
from spsbert.synthetic import make_synthetic
from spsbert.utils import read_json, slugify, write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("chk")
    csv = make_synthetic(tmp / "d.csv", n_projects=2, sizes=(130, 150))
    cfg = load_config(ROOT / "config" / "smoke_test.yaml",
                      [f"data.csv_path={csv}", f"output_dir={tmp / 'res'}", f"embedding.cache_dir={tmp / 'c'}", "cnn.max_epochs=6"])
    run_all(cfg)
    aggregate(tmp / "res")
    return tmp / "res"


def status(checks):
    return {c.id: c.status for c in checks}


def test_sound_run_passes_everything_except_the_synthetic_guards(good):
    st = status(run_checks(good, ROOT))
    assert st["A09"] == "FAIL" and st["A15"] == "FAIL"            # smoke run can never pass as a real submission
    assert {k for k, v in st.items() if v == "FAIL"} <= {"A09", "A11", "A15"}   # A11 only if sentence-transformers missing
    assert st["A20"] == "PENDING"
    assert st["A21"] == "WARN" and st["A22"] == "PASS"                 # smoke config differs from default; both projects evaluated
    for k in ("A00", "A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A10", "A12", "A13", "A14", "A16", "A17", "A18", "A19"):
        assert st[k] == "PASS", k


def _mutated(good, tmp_path):
    dst = tmp_path / "res"
    shutil.copytree(good, dst)
    return dst, sorted((dst / "projects").glob("*.json"))[0]


def test_checker_detects_tampered_fold_values(good, tmp_path):
    dst, f = _mutated(good, tmp_path)
    d = read_json(f); d["methods"]["sbert_cnn"]["cv"]["fold_mae"][0] += 1.0; write_json(d, f)
    assert status(run_checks(dst, ROOT))["A03"] == "FAIL"


def test_checker_detects_wrong_model_selection(good, tmp_path):
    dst, f = _mutated(good, tmp_path)
    d = read_json(f); m = d["methods"]["tfidf_ridge"]
    m["best_params"] = max(m["grid"], key=lambda g: g["mae_mean"])["params"]     # deliberately pick the worst setting
    write_json(d, f)
    assert status(run_checks(dst, ROOT))["A07"] == "FAIL"


def test_checker_detects_test_metric_not_matching_predictions(good, tmp_path):
    dst, f = _mutated(good, tmp_path)
    d = read_json(f); d["methods"]["median_baseline"]["test"]["mae"] *= 0.5; write_json(d, f)
    assert status(run_checks(dst, ROOT))["A08"] == "FAIL"


def test_checker_detects_missing_split_and_bad_ratio_and_missing_plots(good, tmp_path):
    dst, f = _mutated(good, tmp_path)
    slug = slugify(read_json(f)["project"])
    (dst / "splits" / f"{slug}.json").unlink()
    assert status(run_checks(dst, ROOT))["A06"] == "FAIL"
    d = read_json(f); d["n_test"] += 5; write_json(d, f)
    assert status(run_checks(dst, ROOT))["A05"] == "FAIL"
    for png in (dst / "figures").glob("*.png"):
        png.unlink()
    assert status(run_checks(dst, ROOT))["A13"] == "FAIL"


def test_checker_detects_readme_gaps(good, tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns("results*", "cache", ".git", "__pycache__", "*.egg-info"))
    (repo / "README.md").write_text("# nothing here", encoding="utf-8")
    assert status(run_checks(good, repo))["A16"] == "FAIL"
    assert status(run_checks(good, repo))["A19"] == "FAIL"


def test_checker_flags_partial_runs_and_reduced_configs(good, tmp_path):
    dst = tmp_path / "res"
    shutil.copytree(good, dst)
    sorted((dst / "projects").glob("*.json"))[0].unlink()                # simulate a batch/partial run
    st = status(run_checks(dst, ROOT))
    assert st["A22"] == "WARN"
    c = next(c for c in run_checks(good, ROOT) if c.id == "A21")
    assert "; seed:" not in "; " + c.detail and "cnn.max_epochs" in c.detail      # names exactly what differs from default.yaml

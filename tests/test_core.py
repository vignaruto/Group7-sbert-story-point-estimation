from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spsbert.config import Config, load_config
from spsbert.data import build_dataset, clean_text, compose_text, load_raw
from spsbert.config import DataConfig
from spsbert.metrics import bootstrap_ci, mae, mdae, paired_wilcoxon
from spsbert.splits import assert_valid_split, make_split
from spsbert.utils import slugify


# ---------------- metrics
def test_mae_mdae_hand_computed():
    y, p = [1, 2, 3, 10], [1, 4, 3, 6]          # abs errs 0,2,0,4
    assert mae(y, p) == pytest.approx(1.5)
    assert mdae(y, p) == pytest.approx(1.0)


def test_metric_shape_mismatch():
    with pytest.raises(ValueError):
        mae([1, 2], [1])


def test_bootstrap_ci_contains_point_estimate_and_is_deterministic():
    errs = np.abs(np.random.default_rng(0).normal(2, 1, size=(2, 200)))
    a, b = bootstrap_ci(errs, 300, 1), bootstrap_ci(errs, 300, 1)
    assert a == b
    assert a["mae_ci95"][0] <= errs.mean() <= a["mae_ci95"][1]


def test_wilcoxon_identical_is_nan_and_different_is_small():
    x = np.abs(np.random.default_rng(1).normal(size=100))
    assert np.isnan(paired_wilcoxon(x, x))
    assert paired_wilcoxon(x + 1.0, x) < 1e-6


# ---------------- splits
@pytest.mark.parametrize("n", [60, 101, 333])
def test_split_properties(n):
    s = make_split(n, 0.2, 5, 42)
    assert_valid_split(s, n)
    assert len(s.test_idx) == int(np.ceil(0.2 * n))
    assert len(s.folds) == 5
    sizes = [len(v) for _, v in s.folds]
    assert max(sizes) - min(sizes) <= 1


def test_split_is_deterministic_and_seed_sensitive():
    a, b, c = make_split(200, 0.2, 5, 1), make_split(200, 0.2, 5, 1), make_split(200, 0.2, 5, 2)
    assert np.array_equal(a.test_idx, b.test_idx)
    assert not np.array_equal(a.test_idx, c.test_idx)


def test_split_too_small():
    with pytest.raises(ValueError):
        make_split(8, 0.2, 5, 0)


def test_split_json_roundtrip_keys():
    s = make_split(50, 0.2, 5, 0)
    j = s.to_json([f"K-{i}" for i in range(50)])
    assert len(j["test_issue_keys"]) + len(j["dev_issue_keys"]) == 50
    assert set(j["validation_fold_of_dev_issue"]) == set(j["dev_issue_keys"])


# ---------------- text / data
def test_clean_text():
    raw = "Fix {code:java}int x=1;{code} see https://a.b/c [~bob] <b>now</b>\n\n  please"
    out = clean_text(raw)
    assert "int x" not in out and "https" not in out and "bob" not in out and "<b>" not in out
    assert out == "Fix see URL now please"
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


def test_compose_text():
    assert compose_text("Title", "Desc") == "Title. Desc"
    assert compose_text("Title.", "Desc") == "Title. Desc"
    assert compose_text("Title", "") == "Title" and compose_text("", "D") == "D"


def _csv(tmp_path, rows, cols=("Project_Name", "Issue_Key", "Title", "Description_Text", "Creation_Date", "Story_Point")):
    p = tmp_path / "d.csv"
    pd.DataFrame(rows, columns=list(cols)).to_csv(p, index=False)
    return p


def test_build_dataset_filters(tmp_path):
    rows = []
    for i in range(30):
        rows.append(["A", f"A-{i}", f"title {i}", "desc", "2020-01-01", 3])
    rows += [["A", "A-x1", "bad sp", "d", "2020", 0], ["A", "A-x2", "nan sp", "d", "2020", None],
             ["A", "A-x3", "", "", "2020", 5], ["A", "A-dup", "title 1", "desc", "2020", 8],
             ["B", "B-1", "small", "d", "2020", 2]]
    df, rep = build_dataset(DataConfig(csv_path=str(_csv(tmp_path, rows)), min_issues=20))
    assert set(df.project) == {"A"} and len(df) == 30
    assert rep["dropped_missing_or_nonpositive_story_point"] == 2
    assert rep["dropped_empty_text"] == 1 and rep["dropped_duplicate_text"] == 1
    assert "B" in rep["skipped_projects_too_small"]
    assert list(df.issue_key) == sorted(df.issue_key)


def test_column_aliases_and_missing(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame({"project": ["A"], "key": ["A-1"], "summary": ["s"], "story_points": [3]}).to_csv(p, index=False)
    assert {"project", "issue_key", "title", "story_point"} <= set(load_raw(p).columns)
    pd.DataFrame({"project": ["A"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required"):
        load_raw(p)
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")


def test_project_whitelist_case_insensitive(tmp_path):
    rows = [["Alpha Proj", f"K-{i}", f"t{i}", "d", "2020", 2] for i in range(30)]
    cfg = DataConfig(csv_path=str(_csv(tmp_path, rows)), projects=["alpha proj", "Missing"], min_issues=20)
    df, rep = build_dataset(cfg)
    assert df.project.nunique() == 1 and rep["requested_projects_not_found"] == ["Missing"]


# ---------------- config / utils
def test_config_validation_and_overrides():
    cfg = load_config(None, ["split.n_folds=4", "cnn.grid.lr=[0.1]"])
    assert cfg.split.n_folds == 4 and cfg.cnn.grid["lr"] == [0.1]
    with pytest.raises(ValueError):
        Config.from_dict({"splitt": {}})
    with pytest.raises(ValueError):
        load_config(None, ["split.test_size=1.5"])
    with pytest.raises(ValueError):
        load_config(None, ["cnn.channels=[1,2]"])


def test_fingerprint_changes_with_hyperparameters():
    assert load_config(None).fingerprint() == load_config(None).fingerprint()
    assert load_config(None).fingerprint() != load_config(None, ["seed=1"]).fingerprint()


def test_default_yaml_loads():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name in ("default", "smoke_test", "quick", "balanced"):
        load_config(root / "config" / f"{name}.yaml")


def test_slugify_unique_for_guideline_names():
    from pathlib import Path
    from spsbert.data import read_project_list
    names = read_project_list(Path(__file__).resolve().parents[1] / "config" / "guideline_projects.txt")
    assert len(names) == 39
    assert len({slugify(n) for n in names}) == 39


def test_problem_statement_scoring_is_consistent_with_guideline_table():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("pss", root / "scripts" / "ps_selection_scores.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert sum(m.WEIGHTS.values()) == 100 and len(m.WEIGHTS) == 8
    S = m.STATEMENTS                                             # numbering must match the guideline's table
    assert S[1] == ("BERT", "within", "random") and S[4] == ("BERT", "cross", "temporal")
    assert S[5] == ("SBERT", "within", "random") and S[6] == ("SBERT", "within", "temporal")
    assert S[9] == ("COMB", "within", "random") and S[12] == ("COMB", "cross", "temporal")
    sc = m.scores()
    assert len(sc) == 12 and all(0 < v["total"] <= 100 for v in sc.values())


def test_outlier_cap_removes_data_entry_errors_and_reports_counts(tmp_path):
    rows = [["P", f"K-{i}", f"title {i}", "d", "2020", 3] for i in range(30)]
    rows += [["P", "K-big1", "huge a", "d", "2020", 26710114], ["P", "K-big2", "huge b", "d", "2020", 101], ["P", "K-100", "edge", "d", "2020", 100]]
    p = tmp_path / "d.csv"
    pd.DataFrame(rows, columns=["Project_Name", "Issue_Key", "Title", "Description_Text", "Creation_Date", "Story_Point"]).to_csv(p, index=False)
    df, rep = build_dataset(DataConfig(csv_path=str(p), min_issues=20, max_story_point=100))
    assert rep["dropped_above_max_story_point"] == 2 and rep["dropped_above_max_by_project"] == {"P": 2}
    assert rep["raw_story_point_max"] == 26710114 and df.story_point.max() == 100          # 100 itself is kept
    assert load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml").data.max_story_point == 100

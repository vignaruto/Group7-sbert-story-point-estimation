"""Adversarial / audit tests: leakage, cache invalidation, degenerate data, CLI, report, demo, verify."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spsbert.aggregate import aggregate, improvement_table
from spsbert.cli import main as cli_main
from spsbert.config import load_config
from spsbert.demo import predict_texts, verify_project
from spsbert.experiment import cross_validate, final_evaluation, run_all
from spsbert.metrics import holm_adjust
from spsbert.models import MethodSpec, TransformedSklearn
from spsbert.splits import make_split
from spsbert.synthetic import make_synthetic
from spsbert.utils import read_json, write_json

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp, csv, out, extra=()):
    return load_config(ROOT / "config" / "smoke_test.yaml",
                       [f"data.csv_path={csv}", f"output_dir={out}", f"embedding.cache_dir={tmp / 'cache'}",
                        "cnn.max_epochs=8", *extra])


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke")
    csv = make_synthetic(tmp / "d.csv", n_projects=2, sizes=(130, 150))
    cfg = _cfg(tmp, csv, tmp / "res")
    run_all(cfg)
    aggregate(tmp / "res")
    return {"tmp": tmp, "csv": csv, "cfg": cfg, "out": tmp / "res"}


# ------------------------------------------------------------------ leakage spy
def test_no_leakage_between_train_validation_and_test():
    """Record every index a model is fitted on / scored on; the test set must never be fitted on or used in CV."""
    n = 120
    split = make_split(n, 0.2, 5, 7)
    y = np.arange(n, dtype=float) + 1
    X = np.arange(n, dtype=float).reshape(-1, 1)          # feature == row index, so indices are observable
    log = {"fit": [], "pred": []}

    class Spy:
        def fit(self, X_, y_):
            log["fit"].append(set(X_[:, 0].astype(int)))
            return self

        def predict(self, X_):
            log["pred"].append(set(X_[:, 0].astype(int)))
            return np.zeros(len(X_))

    spec = MethodSpec("spy", "Spy", "embedding", [{}, {"a": 1}], lambda p, s: Spy(), [0, 1])
    test = set(split.test_idx.tolist())
    cross_validate(spec, X, y, split, 0)
    assert log["fit"] and all(not (f & test) for f in log["fit"]) and all(not (p & test) for p in log["pred"])
    for (tr, va), f, p in zip(split.folds * 2, log["fit"], log["pred"]):
        assert f == set(tr.tolist()) and p == set(va.tolist()) and not (f & p)
    log["fit"].clear(); log["pred"].clear()
    final_evaluation(spec, {}, X, y, split, 50, 0)
    assert all(f == set(split.dev_idx.tolist()) for f in log["fit"])          # fitted on ALL dev data only
    assert all(p == test for p in log["pred"])                                # scored on the test set only


def test_tfidf_vocabulary_comes_from_training_data_only():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    m = TransformedSklearn(lambda: make_pipeline(TfidfVectorizer(), Ridge()), "log1p")
    m.fit(np.array(["alpha beta", "beta gamma", "alpha gamma"], dtype=object), np.array([1.0, 2.0, 3.0]))
    vocab = m.est_.steps[0][1].vocabulary_
    m.predict(np.array(["zzzunseen word"], dtype=object))
    assert "zzzunseen" not in m.est_.steps[0][1].vocabulary_ and set(vocab) == {"alpha", "beta", "gamma"}


# ------------------------------------------------------------------ caching / robustness
def test_cache_is_invalidated_when_data_changes_but_config_does_not(tmp_path):
    csv = make_synthetic(tmp_path / "a.csv", n_projects=1, sizes=(130,))
    cfg = _cfg(tmp_path, csv, tmp_path / "res", ["cnn.max_epochs=3"])
    r1 = run_all(cfg)
    df = pd.read_csv(csv)
    df["Story_Point"] = df["Story_Point"] * 2
    df.to_csv(csv, index=False)                                   # same path, same config, new labels
    r2 = run_all(cfg)
    assert r1[0]["methods"]["median_baseline"]["test"]["mae"] != r2[0]["methods"]["median_baseline"]["test"]["mae"]
    r3 = run_all(cfg)                                              # unchanged -> served from cache
    assert r3[0]["data_hash"] == r2[0]["data_hash"]


def test_constant_target_project_runs_end_to_end(tmp_path):
    csv = make_synthetic(tmp_path / "a.csv", n_projects=2, sizes=(130, 140))
    df = pd.read_csv(csv)
    df.loc[df.Project_Name == "SYNTH-A", "Story_Point"] = 3
    df.to_csv(csv, index=False)
    cfg = _cfg(tmp_path, csv, tmp_path / "res", ["cnn.max_epochs=3"])
    results = run_all(cfg)
    a = next(r for r in results if r["project"] == "SYNTH-A")
    assert all(m["test"]["mae"] == 0 and m["test"]["mdae"] == 0 for m in a["methods"].values())
    agg = aggregate(tmp_path / "res")
    imp = improvement_table(agg["test"])
    assert not np.isinf(imp["mae_improvement_pct"].to_numpy(float)).any()        # no division by zero
    assert agg["test"]["p_vs_median"].dtype.kind == "f"


def test_projects_with_colliding_file_names_are_rejected(tmp_path):
    rows = [[n, f"K-{i}", f"title {n} {i}", "d", "2020", 2] for n in ("A B", "A-B") for i in range(30)]
    csv = tmp_path / "c.csv"
    pd.DataFrame(rows, columns=["Project_Name", "Issue_Key", "Title", "Description_Text", "Creation_Date",
                                "Story_Point"]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="same file name"):
        run_all(_cfg(tmp_path, csv, tmp_path / "res", ["data.min_issues=20"]))


# ------------------------------------------------------------------ statistics / io
def test_holm_adjustment_known_values_and_nan_handling():
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    adj = holm_adjust([0.5, np.nan, 0.001])
    assert np.isnan(adj[1]) and np.allclose(adj[[0, 2]], [0.5, 0.002])
    assert (holm_adjust([0.9, 0.8]) <= 1).all()


def test_write_json_converts_nan_and_numpy(tmp_path):
    write_json({"a": float("nan"), "b": np.float64(1.5), "c": np.arange(2), "d": [float("inf")]}, tmp_path / "x.json")
    assert read_json(tmp_path / "x.json") == {"a": None, "b": 1.5, "c": [0, 1], "d": [None]}


# ------------------------------------------------------------------ outputs required by the guideline
def test_results_contain_everything_the_guideline_requires(smoke):
    out = smoke["out"]
    for f in ("tables/validation_per_fold.csv", "tables/validation_summary.csv", "tables/test_results.csv",
              "tables/macro_summary.csv", "tables/dataset_summary.csv", "figures/test_mae_per_project.png",
              "run_metadata.json", "data_cleaning_report.json"):
        assert (out / f).exists(), f
    folds = pd.read_csv(out / "tables" / "validation_per_fold.csv")
    assert set(folds.fold) == {1, 2, 3, 4, 5}                                        # five folds
    assert (folds.groupby(["project", "method"]).size() == 5).all()
    val = pd.read_csv(out / "tables" / "validation_summary.csv")
    assert {"mae_mean", "mae_std", "mdae_mean", "mdae_std"} <= set(val.columns)       # mean & std over folds
    test = pd.read_csv(out / "tables" / "test_results.csv")
    assert {"mae", "mdae", "mae_ci_lo", "p_vs_median_holm"} <= set(test.columns)
    meta = read_json(out / "run_metadata.json")
    assert meta["config"]["seed"] == 42 and "torch" in meta["environment"]            # seeds + versions recorded
    res = read_json(next((out / "projects").glob("*.json")))
    assert res["methods"]["sbert_cnn"]["test"]["n_trainable_params"] > 0
    assert res["n_dev"] + res["n_test"] == res["n_issues"]
    assert abs(res["n_test"] / res["n_issues"] - 0.2) < 0.02                         # 80:20


def test_validation_and_test_sets_are_reported_separately_and_differ(smoke):
    val = pd.read_csv(smoke["out"] / "tables" / "validation_summary.csv")
    test = pd.read_csv(smoke["out"] / "tables" / "test_results.csv")
    m = val.merge(test, on=["project", "method"])
    assert not np.allclose(m["mae_mean"], m["mae"])


# ------------------------------------------------------------------ CLI, demo, verify, report
def test_cli_synthetic_aggregate_and_report(smoke, tmp_path, capsys):
    assert cli_main(["synthetic", "--out", str(tmp_path / "s.csv")]) == 0
    assert (tmp_path / "s.csv").exists()
    assert cli_main(["aggregate", "--output-dir", str(smoke["out"])]) == 0
    pdf = tmp_path / "r.pdf"
    assert cli_main(["report", "--output-dir", str(smoke["out"]), "--github-url", "https://github.com/x/y",
                     "--out", str(pdf)]) == 0
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 20000
    if shutil.which("pdftotext"):
        import subprocess
        txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
        assert "NOT REAL RESULTS" in txt                                              # synthetic run must be stamped
        assert "https://github.com/x/y" in txt.split("1. Problem statement")[0]       # link on the first page
        assert "B230857CS" in txt and "Group 7" in txt
        ph = tmp_path / "ph.pdf"
        assert cli_main(["report", "--output-dir", str(smoke["out"]), "--github-url", "https://github.com/<your-username>/<repo>", "--out", str(ph)]) == 0
        from spsbert.check import run_checks
        a20 = lambda f: next(c.status for c in run_checks(smoke["out"], ROOT, f) if c.id == "A20")  # noqa: E731
        assert a20(ph) == "FAIL" and a20(pdf) == "PASS"                   # placeholder link must never pass the final audit
        flat = " ".join(txt.split())
        assert "required by guideline 3.4" in flat and "Combined BERT + SBERT" in flat


def test_predict_texts_api_and_cli(smoke, capsys):
    kw = dict(csv_path=str(smoke["csv"]))
    preds = predict_texts(smoke["out"], "SYNTH-A", ["fix typo in label", "refactor security architecture"], **kw)
    assert len(preds) == 2 and all(np.isfinite(preds)) and all(p > 0 for p in preds)
    assert preds == predict_texts(smoke["out"], "SYNTH-A", ["fix typo in label", "refactor security architecture"], **kw)
    assert cli_main(["predict", "--output-dir", str(smoke["out"]), "--project", "SYNTH-A", "--csv", str(smoke["csv"]),
                     "--text", "fix typo"]) == 0
    assert "story points" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        predict_texts(smoke["out"], "NoSuchProject", ["x"], **kw)
    with pytest.raises(ValueError):
        predict_texts(smoke["out"], "SYNTH-A", ["   "], **kw)


def test_verify_reproduces_stored_results_exactly(smoke):
    r = verify_project(smoke["out"], "SYNTH-B", str(smoke["csv"]))
    assert r["all_match"], [row for row in r["rows"] if not row["match"]]
    assert len(r["rows"]) == 2 * len(smoke["cfg"].evaluation.methods)


def test_two_independent_runs_are_identical(tmp_path):
    csv = make_synthetic(tmp_path / "a.csv", n_projects=1, sizes=(130,))
    res = []
    for k in range(2):
        cfg = _cfg(tmp_path, csv, tmp_path / f"r{k}", ["cnn.max_epochs=4", f"embedding.cache_dir={tmp_path / f'c{k}'}"])
        res.append(run_all(cfg)[0])
    for m in res[0]["methods"]:
        assert res[0]["methods"][m]["test"]["mae"] == res[1]["methods"][m]["test"]["mae"]
        assert res[0]["methods"][m]["cv"]["fold_mae"] == res[1]["methods"][m]["cv"]["fold_mae"]


def test_batches_can_be_merged_into_one_valid_result_set(tmp_path):
    from spsbert.merge import merge_results
    csv = make_synthetic(tmp_path / "d.csv", n_projects=2, sizes=(130, 150))
    runs = []
    for i, name in enumerate(("SYNTH-A", "SYNTH-B")):                      # two "sessions", one project each
        cfg = _cfg(tmp_path, csv, tmp_path / f"b{i}", ["cnn.max_epochs=4"])
        run_all(cfg, only=[name]); runs.append(tmp_path / f"b{i}")
    info = merge_results(runs, tmp_path / "merged")
    assert info["projects"] == ["SYNTH-A", "SYNTH-B"]
    agg = aggregate(tmp_path / "merged")
    assert agg["test"].project.nunique() == 2
    from spsbert.check import run_checks
    st = {c.id: c.status for c in run_checks(tmp_path / "merged", ROOT)}
    assert all(st[k] == "PASS" for k in ("A01", "A02", "A03", "A06", "A07", "A08"))
    # different configurations must be refused
    cfg2 = _cfg(tmp_path, csv, tmp_path / "other", ["cnn.max_epochs=5"])
    run_all(cfg2, only=["SYNTH-A"])
    with pytest.raises(ValueError, match="different configurations"):
        merge_results([runs[0], tmp_path / "other"], tmp_path / "bad")

import numpy as np
import pandas as pd

from spsbert.aggregate import ORDER, _fig_grouped, _fig_improvement, extra_analysis, improvement_table, LAB


def _fake_test(n=15, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = rng.uniform(1, 5)
        for k, m in enumerate(ORDER):
            rows.append({"project": f"P{i:02d}", "method": m, "label": LAB[m], "mae": base * (1.3 - 0.1 * k) + rng.normal(0, .05), "mdae": base})
    return pd.DataFrame(rows)


def test_extra_analysis_ranks_wtl_and_spearman(tmp_path):
    t = _fake_test()
    (tmp_path / "tables").mkdir()
    pd.DataFrame({"project": sorted(t.project.unique()), "n_issues": range(100, 115), "mode_share": np.linspace(.2, .5, 15),
                  "n_distinct_sp": range(5, 20), "sp_std": np.linspace(1, 3, 15)}).to_csv(tmp_path / "tables" / "dataset_summary.csv", index=False)
    r = extra_analysis(tmp_path, t)
    assert r["n_projects"] == 15 and set(r["mean_ranks"]) == set(ORDER)
    assert r["mean_ranks"]["sbert_cnn"] < r["mean_ranks"]["mean_baseline"]            # built so lower index = worse
    assert r["friedman"]["p"] < 1e-6 and set(r["primary_vs"]) == set(ORDER) - {"sbert_cnn"}
    assert all(v["wins"] + v["ties"] + v["losses"] == 15 for v in r["primary_vs"].values())
    assert set(r["improvement_spearman"]) == {"n_issues", "mode_share", "n_distinct_sp", "sp_std"}
    assert (tmp_path / "tables" / "analysis_extra.json").exists()


def test_many_project_figures_use_the_readable_dot_layout(tmp_path):
    t = _fake_test(30)
    _fig_grouped(t, "mae", tmp_path / "a.png", "t")
    _fig_improvement(improvement_table(t), tmp_path / "b.png")
    from PIL import Image
    for f in ("a.png", "b.png"):
        w, h = Image.open(tmp_path / f).size
        assert h > 0.8 * w                                          # tall enough to be legible (old layout was ~12:1 wide)


def test_constant_columns_do_not_raise_warnings(tmp_path):
    import warnings
    t = _fake_test(10)
    (tmp_path / "tables").mkdir()
    pd.DataFrame({"project": sorted(t.project.unique()), "n_issues": 100, "mode_share": 0.3, "n_distinct_sp": range(10), "sp_std": 1.0}
                 ).to_csv(tmp_path / "tables" / "dataset_summary.csv", index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = extra_analysis(tmp_path, t)
    assert set(r["improvement_spearman"]) == {"n_distinct_sp"}

"""Collect per-project results into tables (CSV/Markdown) and figures."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from .metrics import holm_adjust  # noqa: E402
from .utils import read_json  # noqa: E402

LOGGER = logging.getLogger("spsbert.aggregate")
PRIMARY = "sbert_cnn"
ORDER = ["mean_baseline", "median_baseline", "tfidf_ridge", "sbert_ridge", "sbert_cnn"]


def load_results(out_dir: str | Path) -> list:
    files = sorted((Path(out_dir) / "projects").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No per-project results in {out_dir}/projects. Run the experiment first.")
    return [read_json(f) for f in files]


def tidy_tables(results: list) -> tuple:
    """Return (validation_per_fold, validation_summary, test) dataframes in long format."""
    fold_rows, val_rows, test_rows = [], [], []
    for r in results:
        for m, d in r["methods"].items():
            cv, t = d["cv"], d["test"]
            for k, (a, b) in enumerate(zip(cv["fold_mae"], cv["fold_mdae"]), 1):
                fold_rows.append({"project": r["project"], "method": m, "fold": k, "mae": a, "mdae": b})
            val_rows.append({"project": r["project"], "method": m, "label": d["label"],
                             "mae_mean": cv["mae_mean"], "mae_std": cv["mae_std"],
                             "mdae_mean": cv["mdae_mean"], "mdae_std": cv["mdae_std"],
                             "best_params": str(d["best_params"])})
            test_rows.append({"project": r["project"], "method": m, "label": d["label"],
                              "n_dev": r["n_dev"], "n_test": r["n_test"],
                              "mae": t["mae"], "mdae": t["mdae"],
                              "mae_std_seeds": t["mae_std_over_seeds"], "mdae_std_seeds": t["mdae_std_over_seeds"],
                              "mae_ci_lo": t["mae_ci95"][0], "mae_ci_hi": t["mae_ci95"][1],
                              "mdae_ci_lo": t["mdae_ci95"][0], "mdae_ci_hi": t["mdae_ci95"][1],
                              "p_vs_median": t.get("wilcoxon_p_vs_median_baseline"),
                              "p_vs_tfidf": t.get("wilcoxon_p_vs_tfidf_ridge")})
    test = pd.DataFrame(test_rows)
    for col in ("p_vs_median", "p_vs_tfidf"):          # all-None columns would otherwise be object dtype
        test[col] = pd.to_numeric(test[col], errors="coerce")
    for col in ("p_vs_median", "p_vs_tfidf"):          # Holm correction across projects, per method
        test[col + "_holm"] = np.nan
        for m, idx in test.groupby("method").groups.items():
            test.loc[idx, col + "_holm"] = holm_adjust(test.loc[idx, col].to_numpy())
    return pd.DataFrame(fold_rows), pd.DataFrame(val_rows), test


def macro_summary(val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Macro-average (every project weighs equally) of validation and test metrics per method."""
    v = val.groupby("method").agg(val_mae=("mae_mean", "mean"), val_mae_fold_std=("mae_std", "mean"),
                                  val_mdae=("mdae_mean", "mean"), val_mdae_fold_std=("mdae_std", "mean"))
    t = test.groupby(["method", "label"]).agg(test_mae=("mae", "mean"), test_mdae=("mdae", "mean")).reset_index().set_index("method")
    out = t.join(v)
    wins_mae = test.loc[test.groupby("project")["mae"].idxmin(), "method"].value_counts()
    wins_mdae = test.loc[test.groupby("project")["mdae"].idxmin(), "method"].value_counts()
    out["projects_lowest_mae"] = wins_mae.reindex(out.index).fillna(0).astype(int)
    out["projects_lowest_mdae"] = wins_mdae.reindex(out.index).fillna(0).astype(int)
    return out.reset_index().sort_values("test_mae").reset_index(drop=True)


def improvement_table(test: pd.DataFrame, ref: str = "median_baseline") -> pd.DataFrame:
    """% reduction of MAE/MdAE relative to a reference method, per project (positive = better than ref)."""
    p_mae = test.pivot(index="project", columns="method", values="mae")
    p_mdae = test.pivot(index="project", columns="method", values="mdae")
    if ref not in p_mae:
        return pd.DataFrame()
    rows = []
    for m in p_mae.columns:
        if m == ref:
            continue
        for proj in p_mae.index:
            r_mae, r_mdae = p_mae.loc[proj, ref], p_mdae.loc[proj, ref]
            rows.append({"project": proj, "method": m,
                         "mae_improvement_pct": 100 * (r_mae - p_mae.loc[proj, m]) / r_mae if r_mae > 0 else np.nan,
                         "mdae_improvement_pct": 100 * (r_mdae - p_mdae.loc[proj, m]) / r_mdae if r_mdae > 0 else np.nan})
    return pd.DataFrame(rows)


def cross_project_wilcoxon(test: pd.DataFrame, a: str = PRIMARY, b: str = "median_baseline") -> dict:
    """Wilcoxon signed-rank across projects on per-project test MAE (needs >= 6 projects)."""
    p = test.pivot(index="project", columns="method", values="mae")
    if a not in p or b not in p or len(p) < 6:
        return {"n_projects": int(len(p)), "p_value": None}
    d = p[a] - p[b]
    if np.allclose(d, 0):
        return {"n_projects": int(len(p)), "p_value": None}
    return {"n_projects": int(len(p)), "p_value": float(stats.wilcoxon(p[a], p[b]).pvalue),
            "projects_a_better": int((d < 0).sum()), "projects_b_better": int((d > 0).sum())}


def extra_analysis(out: Path, test: pd.DataFrame) -> dict:
    """Rank test, win/tie/loss of the primary method, and which project properties go with SBERT+CNN gains (Spearman)."""
    methods = [m for m in ORDER if m in set(test.method)]
    P = test.pivot(index="project", columns="method", values="mae")[methods]
    res: dict = {"n_projects": int(len(P)), "mean_ranks": {m: float(v) for m, v in P.rank(axis=1).mean().items()}}
    if len(P) >= 6 and len(methods) >= 3:
        fr = stats.friedmanchisquare(*[P[c] for c in P])
        res["friedman"] = {"chi2": float(fr.statistic), "p": float(fr.pvalue)}
    if PRIMARY in P:
        wtl = {}
        for b in methods:
            if b == PRIMARY:
                continue
            d = P[PRIMARY] - P[b]
            p = float(stats.wilcoxon(P[PRIMARY], P[b]).pvalue) if len(P) >= 6 and not np.allclose(d, 0) else None
            wtl[b] = {"wins": int((d < 0).sum()), "ties": int((d == 0).sum()), "losses": int((d > 0).sum()), "wilcoxon_p": p}
        res["primary_vs"] = wtl
        sf = out / "tables" / "dataset_summary.csv"
        if sf.exists() and "median_baseline" in P and len(P) >= 6:
            s = pd.read_csv(sf).set_index("project").reindex(P.index)
            imp = 100 * (P["median_baseline"] - P[PRIMARY]) / P["median_baseline"].replace(0, np.nan)
            sp = {}
            for col in ("n_issues", "mode_share", "n_distinct_sp", "sp_std"):
                ok = imp.notna() & s[col].notna()
                if ok.sum() >= 6 and s.loc[ok, col].nunique() > 1 and imp[ok].nunique() > 1:      # correlation undefined for constants
                    r = stats.spearmanr(s.loc[ok, col], imp[ok])
                    sp[col] = {"rho": float(r.statistic), "p": float(r.pvalue)}
            res["improvement_spearman"] = sp
            res["median_improvement_pct"] = float(imp.median())
    from .utils import write_json
    write_json(res, out / "tables" / "analysis_extra.json")
    return res


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ figures
MARK = {"Mean": ("s", "#8c8c8c"), "Median": ("D", "#ff7f0e"), "TF-IDF+Ridge": ("^", "#9467bd"), "SBERT+Ridge": ("v", "#d62728"), "SBERT+CNN": ("o", "#2ca02c")}
LAB = {"mean_baseline": "Mean", "median_baseline": "Median", "tfidf_ridge": "TF-IDF+Ridge", "sbert_ridge": "SBERT+Ridge", "sbert_cnn": "SBERT+CNN"}


def _fig_dot(piv: pd.DataFrame, xlabel: str, title: str, path: Path, zero: bool = False) -> None:
    """Readable for many projects: one row per project, one marker per method."""
    n = len(piv)
    fig, ax = plt.subplots(figsize=(8, 0.22 * n + 2.2))
    y = np.arange(n)
    for lab in piv.columns:
        mk, col = MARK.get(lab, ("o", "k"))
        ax.scatter(piv[lab], y, marker=mk, color=col, s=24, label=lab, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=7); ax.invert_yaxis()
    ax.grid(alpha=0.3); ax.set_xlabel(xlabel); ax.set_title(title, fontsize=10)
    if zero:
        ax.axvline(0, color="k", lw=0.8)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def _fig_grouped(test: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    piv = test.pivot(index="project", columns="label", values=metric)
    if len(piv) > 12:
        order = [l for l in MARK if l in piv.columns]
        _fig_dot(piv[order], f"{metric.upper()} (story points, lower is better)", title, path)
        return
    order = [l for l in ("Mean", "Median", "TF-IDF+Ridge", "SBERT+Ridge", "SBERT+CNN") if l in piv.columns]
    piv = piv[order]
    n_proj, n_m = piv.shape
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * n_proj * n_m / 2 + 3), 4.6))
    piv.plot(kind="bar", ax=ax, width=0.85, edgecolor="none")
    ax.set_ylabel(metric.upper() + " (story points)"); ax.set_xlabel("")
    ax.set_title(title); ax.legend(fontsize=8, ncol=min(5, n_m), loc="upper center"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, float(np.nanmax(piv.to_numpy())) * 1.22)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _fig_improvement(imp: pd.DataFrame, path: Path) -> None:
    if imp.empty:
        return
    piv = imp.pivot(index="project", columns="method", values="mae_improvement_pct")
    if len(piv) > 12:
        d = piv.rename(columns=LAB).clip(-150, 100)
        d = d.sort_values("SBERT+CNN") if "SBERT+CNN" in d else d
        _fig_dot(d[[c for c in MARK if c in d.columns]], "MAE change vs median baseline (%, positive = better; axis clipped to [-150, 100])",
                 "Relative test-MAE improvement over the median baseline", path, zero=True)
        return
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(piv) * piv.shape[1] / 2 + 3), 4.4))
    piv.plot(kind="bar", ax=ax, width=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("MAE reduction vs median baseline (%)"); ax.set_xlabel("")
    ax.set_title("Relative test-MAE improvement over the median baseline (positive = better)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _fig_folds(folds: pd.DataFrame, path: Path) -> None:
    labels = [m for m in ORDER if m in set(folds["method"])]
    data = [folds[folds.method == m]["mae"].to_numpy() for m in labels]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels([LAB[m] for m in labels])
    ax.set_ylabel("Validation MAE (one point per project x fold)")
    ax.set_title("Five-fold validation MAE distribution by method"); ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _fig_scatter(out_dir: Path, path: Path) -> None:
    frames = [pd.read_csv(f) for f in sorted((out_dir / "predictions").glob("*.csv"))]
    if not frames or PRIMARY not in frames[0]:
        return
    d = pd.concat(frames)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(d["y_true"], d[PRIMARY], s=8, alpha=0.35)
    lim = [0, max(d["y_true"].max(), d[PRIMARY].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True story points"); ax.set_ylabel("Predicted (SBERT+CNN)")
    ax.set_title("Held-out test predictions, all projects pooled"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def assignment_table(out: Path, val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """The table prescribed in guideline 3.4: Representation | Model configuration | MAE | MdAE.

    Only the SBERT row belongs to PS-5; the BERT and combined rows are other groups' problem statements.
    """
    meta = read_json(out / "run_metadata.json")
    cfg = meta["config"]
    c = cfg["cnn"]
    first = read_json(next((out / "projects").glob("*.json")))
    encoder = f"{meta.get('encoder', cfg['embedding']['model_name'])} (frozen, dim={first.get('embedding_dim')})"
    desc = (f"{encoder} + 3-layer CNN channels={c['channels']}, kernel={c['kernel_size']}, "
            f"{c['pool_type']}-pool, loss={c['loss']}, Adam; grid={c['grid']} selected by 5-fold CV; final seeds={c['final_seeds']}")
    na = "not part of PS-5 (assigned to other groups)"
    rows = []
    for scope in ["ALL PROJECTS (macro mean)"] + sorted(test.project.unique()):
        t = test[(test.method == PRIMARY) & ((test.project == scope) | (scope.startswith("ALL")))]
        v = val[(val.method == PRIMARY) & ((val.project == scope) | (scope.startswith("ALL")))]
        for ev, mae_s, mdae_s in (
            ("5-fold validation (mean ± std over folds)",
             f"{v.mae_mean.mean():.3f} ± {v.mae_std.mean():.3f}", f"{v.mdae_mean.mean():.3f} ± {v.mdae_std.mean():.3f}"),
            ("Final held-out test", f"{t.mae.mean():.3f}", f"{t.mdae.mean():.3f}"),
        ):
            rows.append({"Scope": scope, "Evaluation": ev, "Representation": "BERT word embeddings", "Model configuration": na, "MAE": "-", "MdAE": "-"})
            rows.append({"Scope": scope, "Evaluation": ev, "Representation": "SBERT sentence embeddings", "Model configuration": desc, "MAE": mae_s, "MdAE": mdae_s})
            rows.append({"Scope": scope, "Evaluation": ev, "Representation": "Combined BERT + SBERT embeddings", "Model configuration": na, "MAE": "-", "MdAE": "-"})
    return pd.DataFrame(rows)


def aggregate(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    (out / "tables").mkdir(exist_ok=True); (out / "figures").mkdir(exist_ok=True)
    results = load_results(out)
    folds, val, test = tidy_tables(results)
    macro = macro_summary(val, test)
    imp = improvement_table(test)
    xp = cross_project_wilcoxon(test)

    folds.to_csv(out / "tables" / "validation_per_fold.csv", index=False)
    val.to_csv(out / "tables" / "validation_summary.csv", index=False)
    test.to_csv(out / "tables" / "test_results.csv", index=False)
    macro.to_csv(out / "tables" / "macro_summary.csv", index=False)
    imp.to_csv(out / "tables" / "improvement_vs_median.csv", index=False)
    (out / "tables" / "macro_summary.md").write_text(to_markdown(macro), encoding="utf-8")
    at = assignment_table(out, val, test)
    at.to_csv(out / "tables" / "assignment_table.csv", index=False)
    (out / "tables" / "assignment_table.md").write_text(to_markdown(at[at.Scope.str.startswith("ALL")]), encoding="utf-8")

    _fig_grouped(test, "mae", out / "figures" / "test_mae_per_project.png", "Held-out test MAE per project")
    _fig_grouped(test, "mdae", out / "figures" / "test_mdae_per_project.png", "Held-out test MdAE per project")
    _fig_improvement(imp, out / "figures" / "improvement_vs_median.png")
    _fig_folds(folds, out / "figures" / "validation_fold_mae_boxplot.png")
    _fig_scatter(out, out / "figures" / "pred_vs_true_pooled.png")
    extra = extra_analysis(out, test)
    LOGGER.info("Wrote tables and figures to %s", out)
    return {"extra": extra, "results": results, "folds": folds, "val": val, "test": test, "macro": macro,
            "improvement": imp, "cross_project_wilcoxon": xp}

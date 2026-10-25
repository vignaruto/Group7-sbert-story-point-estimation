"""Evaluation metrics (MAE, MdAE), bootstrap confidence intervals and paired significance tests."""
from __future__ import annotations

import numpy as np
from scipy import stats


def abs_errors(y_true, y_pred) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_pred.shape}")
    return np.abs(y_true - y_pred)


def mae(y_true, y_pred) -> float:
    return float(np.mean(abs_errors(y_true, y_pred)))


def mdae(y_true, y_pred) -> float:
    return float(np.median(abs_errors(y_true, y_pred)))


def regression_metrics(y_true, y_pred) -> dict:
    return {"mae": mae(y_true, y_pred), "mdae": mdae(y_true, y_pred)}


def bootstrap_ci(errs_by_seed: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap over test issues.

    `errs_by_seed` has shape (n_seeds, n_test) of absolute errors. For every resample the metric is
    computed per seed and averaged over seeds (same estimand as the reported headline numbers).
    """
    errs = np.atleast_2d(np.asarray(errs_by_seed, float))
    n = errs.shape[1]
    rng = np.random.default_rng(seed)
    maes, mdaes = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        e = errs[:, rng.integers(0, n, n)]
        maes[b] = e.mean()
        mdaes[b] = np.median(e, axis=1).mean()
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "mae_ci95": [float(np.percentile(maes, lo)), float(np.percentile(maes, hi))],
        "mdae_ci95": [float(np.percentile(mdaes, lo)), float(np.percentile(mdaes, hi))],
    }


def paired_wilcoxon(err_a: np.ndarray, err_b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value on paired per-issue absolute errors (nan if undefined)."""
    a, b = np.asarray(err_a, float), np.asarray(err_b, float)
    if a.shape != b.shape or np.allclose(a, b):
        return float("nan")
    try:
        return float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def holm_adjust(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error control). NaNs are ignored and returned as NaN."""
    p = np.asarray(pvalues, float)
    out = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    if ok.size == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out

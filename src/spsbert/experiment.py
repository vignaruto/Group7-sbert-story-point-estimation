"""Experiment engine: per project -> split, 5-fold CV model selection (dev only), final test evaluation."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .data import build_dataset, project_summary
from .embeddings import EmbeddingStore, build_encoder
from .metrics import abs_errors, bootstrap_ci, paired_wilcoxon, regression_metrics
from .models import MethodSpec, build_method_specs
from .splits import ProjectSplit, make_split
from .utils import environment_info, read_json, sha1_text, slugify, write_json

LOGGER = logging.getLogger("spsbert.experiment")


def project_data_hash(dfp: pd.DataFrame) -> str:
    """Fingerprint of a project's usable data; cached results are reused only if this matches."""
    d = dfp.sort_values("issue_key", kind="mergesort")
    return sha1_text(*[f"{k}|{sp}|{t}" for k, sp, t in zip(d["issue_key"], d["story_point"], d["text"])])


def _ms(vals: list) -> tuple:
    a = np.asarray(vals, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def cross_validate(spec: MethodSpec, X: Any, y: np.ndarray, split: ProjectSplit, seed: int) -> dict:
    """Grid search with k-fold CV on the DEVELOPMENT data only. Selects by mean validation MAE (tie: MdAE)."""
    results = []
    for params in spec.grid:
        f_mae, f_mdae, epochs = [], [], []
        for k, (tr, va) in enumerate(split.folds):
            model = spec.factory(params, seed + k).fit(X[tr], y[tr])
            m = regression_metrics(y[va], model.predict(X[va]))
            f_mae.append(m["mae"]); f_mdae.append(m["mdae"])
            if getattr(model, "best_epoch_", None):
                epochs.append(model.best_epoch_)
        mae_m, mae_s = _ms(f_mae)
        mdae_m, mdae_s = _ms(f_mdae)
        results.append({"params": params, "fold_mae": f_mae, "fold_mdae": f_mdae,
                        "mae_mean": mae_m, "mae_std": mae_s, "mdae_mean": mdae_m, "mdae_std": mdae_s,
                        "mean_best_epoch": float(np.mean(epochs)) if epochs else None})
    best = min(range(len(results)), key=lambda i: (results[i]["mae_mean"], results[i]["mdae_mean"], i))
    return {"best_index": best, "best": results[best], "grid": results}


def final_evaluation(spec: MethodSpec, params: dict, X: Any, y: np.ndarray, split: ProjectSplit,
                     n_boot: int, boot_seed: int) -> tuple:
    """Refit the selected configuration on all development data and score it once on the held-out test set."""
    dev, test = split.dev_idx, split.test_idx
    models = [spec.factory(params, s).fit(X[dev], y[dev]) for s in spec.seeds]
    preds = np.vstack([m.predict(X[test]) for m in models])
    errs = np.vstack([abs_errors(y[test], p) for p in preds])             # (n_seeds, n_test)
    per_seed = [regression_metrics(y[test], p) for p in preds]
    mae_m, mae_s = _ms([m["mae"] for m in per_seed])
    mdae_m, mdae_s = _ms([m["mdae"] for m in per_seed])
    out = {"mae": mae_m, "mdae": mdae_m, "mae_std_over_seeds": mae_s, "mdae_std_over_seeds": mdae_s,
           "n_seeds": len(spec.seeds), "per_seed_mae": [m["mae"] for m in per_seed],
           "per_seed_mdae": [m["mdae"] for m in per_seed],
           "n_trainable_params": getattr(models[0], "n_params_", None),
           "final_best_epochs": [getattr(m, "best_epoch_", None) for m in models]}
    out.update(bootstrap_ci(errs, n_boot, boot_seed))
    return out, preds.mean(0), errs.mean(0)


def run_project(project: str, dfp: pd.DataFrame, cfg: Config, store: EmbeddingStore, specs: dict, out: Path) -> dict:
    t0 = time.time()
    dfp = dfp.sort_values("issue_key", kind="mergesort").reset_index(drop=True)
    y = dfp["story_point"].to_numpy(float)
    slug = slugify(project)
    split = make_split(len(dfp), cfg.split.test_size, cfg.split.n_folds, cfg.seed)
    write_json(split.to_json(dfp["issue_key"].tolist()), out / "splits" / f"{slug}.json")

    emb = store.get(dfp["text"].tolist())
    texts = np.array(dfp["text"].tolist(), dtype=object)
    res: dict[str, Any] = {
        "project": project, "n_issues": len(dfp), "n_dev": len(split.dev_idx), "n_test": len(split.test_idx),
        "embedding_dim": int(emb.shape[1]), "config_hash": cfg.fingerprint(), "methods": {},
        "test_sp_median": float(np.median(y[split.test_idx])), "data_hash": project_data_hash(dfp),
    }
    dev_texts = set(dfp["text"].iloc[split.dev_idx])
    res["n_test_texts_also_in_dev"] = int(sum(t in dev_texts for t in dfp["text"].iloc[split.test_idx]))
    if res["n_test_texts_also_in_dev"]:
        LOGGER.warning("  %d test issue(s) have a text identical to a development issue (data.deduplicate=false?)",
                       res["n_test_texts_also_in_dev"])
    pred_cols = {"issue_key": dfp["issue_key"].to_numpy()[split.test_idx], "y_true": y[split.test_idx]}
    err_cols: dict[str, np.ndarray] = {}

    for name, spec in specs.items():
        ts = time.time()
        X = emb if spec.feature == "embedding" else texts
        cv = cross_validate(spec, X, y[:], split, cfg.seed)
        test, pred, err = final_evaluation(spec, cv["best"]["params"], X, y, split,
                                           cfg.evaluation.bootstrap_samples, cfg.evaluation.bootstrap_seed)
        res["methods"][name] = {"label": spec.label, "best_params": cv["best"]["params"],
                                "cv": {k: v for k, v in cv["best"].items() if k != "params"},
                                "grid": [{"params": g["params"], "mae_mean": g["mae_mean"], "mdae_mean": g["mdae_mean"]}
                                         for g in cv["grid"]],
                                "test": test, "runtime_s": round(time.time() - ts, 2)}
        pred_cols[name] = pred
        err_cols[name] = err
        LOGGER.info("  %-16s val MAE %.3f±%.3f | test MAE %.3f MdAE %.3f", spec.label,
                    cv["best"]["mae_mean"], cv["best"]["mae_std"], test["mae"], test["mdae"])

    # paired significance tests against baselines (same test issues)
    for ref in ("median_baseline", "tfidf_ridge"):
        if ref in err_cols:
            for name in err_cols:
                if name != ref:
                    res["methods"][name]["test"][f"wilcoxon_p_vs_{ref}"] = paired_wilcoxon(err_cols[name], err_cols[ref])

    pd.DataFrame(pred_cols).to_csv(out / "predictions" / f"{slug}.csv", index=False)
    res["runtime_s"] = round(time.time() - t0, 2)
    write_json(res, out / "projects" / f"{slug}.json")
    return res


def run_all(cfg: Config, only: list | None = None, force: bool = False) -> list:
    out = Path(cfg.output_dir)
    for sub in ("projects", "predictions", "splits", "tables", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    df, report = build_dataset(cfg.data)
    write_json(report, out / "data_cleaning_report.json")
    project_summary(df).to_csv(out / "tables" / "dataset_summary.csv", index=False)
    projects = sorted(df["project"].unique())
    slugs: dict = {}
    for p in projects:  # distinct projects must never share an output file name
        if slugify(p) in slugs:
            raise ValueError(f"Projects {slugs[slugify(p)]!r} and {p!r} map to the same file name '{slugify(p)}'")
        slugs[slugify(p)] = p
    if only:
        wanted = {p.casefold() for p in only}
        projects = [p for p in projects if p.casefold() in wanted]
        if not projects:
            raise ValueError(f"None of {only} found among usable projects")
    LOGGER.info("Usable projects (%d): %s", len(projects), projects)

    encoder = build_encoder(cfg.embedding)
    write_json({"config": cfg.to_dict(), "config_hash": cfg.fingerprint(), "environment": environment_info(),
                "encoder": encoder.name, "encoder_backend": cfg.embedding.backend,
                "synthetic_data": all(str(p).startswith("SYNTH") for p in projects),
                "started": pd.Timestamp.now().isoformat(timespec="seconds")}, out / "run_metadata.json")
    store = EmbeddingStore(encoder, cfg.embedding)
    specs = build_method_specs(cfg)

    results = []
    for i, p in enumerate(projects, 1):
        f = out / "projects" / f"{slugify(p)}.json"
        if f.exists() and not force:
            old = read_json(f)
            dfp = df[df.project == p]
            if old.get("config_hash") == cfg.fingerprint() and old.get("data_hash") == project_data_hash(dfp):
                LOGGER.info("[%d/%d] %s: cached result reused (use --force to recompute)", i, len(projects), p)
                results.append(old)
                continue
            LOGGER.info("Config or data changed since last run of %s: recomputing", p)
        LOGGER.info("[%d/%d] %s (n=%d)", i, len(projects), p, int((df.project == p).sum()))
        results.append(run_project(p, df[df.project == p], cfg, store, specs, out))
    return results

"""Demonstration and reproducibility helpers.

* predict_texts - retrain the selected configuration of a finished project and predict new issue texts.
* verify_project - recompute a project from scratch and compare with the stored results.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from .config import Config
from .data import build_dataset, clean_text
from .embeddings import EmbeddingStore, build_encoder
from .experiment import run_project
from .models import build_method_specs
from .splits import make_split
from .utils import read_json, slugify

LOGGER = logging.getLogger("spsbert.demo")


def _load_context(output_dir: str | Path, project: str, csv_path: str | None):
    out = Path(output_dir)
    meta = read_json(out / "run_metadata.json")
    cfg = Config.from_dict(meta["config"]).validate()
    if csv_path:
        cfg.data.csv_path = csv_path
    f = out / "projects" / f"{slugify(project)}.json"
    if not f.exists():
        done = sorted(p.stem for p in (out / "projects").glob("*.json"))
        raise FileNotFoundError(f"No stored result for '{project}' ({f.name}). Available: {done}")
    res = read_json(f)
    df, _ = build_dataset(cfg.data)
    dfp = df[df["project"].str.casefold() == res["project"].casefold()]
    if dfp.empty:
        raise ValueError(f"Project '{res['project']}' not found in {cfg.data.csv_path}")
    return cfg, res, dfp.sort_values("issue_key", kind="mergesort").reset_index(drop=True)


def predict_texts(output_dir: str | Path, project: str, texts: list, method: str = "sbert_cnn",
                  csv_path: str | None = None) -> list:
    """Fit `method` (with the hyper-parameters selected by 5-fold CV) on the project's development data,
    then predict story points for new free-text issues. Mean over the final seeds for stochastic methods."""
    cfg, res, dfp = _load_context(output_dir, project, csv_path)
    if method not in res["methods"]:
        raise ValueError(f"Method '{method}' not in results; available: {list(res['methods'])}")
    spec = build_method_specs(cfg)[method]
    params = res["methods"][method]["best_params"]
    split = make_split(len(dfp), cfg.split.test_size, cfg.split.n_folds, cfg.seed)
    y = dfp["story_point"].to_numpy(float)
    store = EmbeddingStore(build_encoder(cfg.embedding), cfg.embedding)
    clean = [clean_text(t) for t in texts]
    if any(not c for c in clean):
        raise ValueError("Every text must contain some non-empty content")
    if spec.feature == "embedding":
        X_all, X_new = store.get(dfp["text"].tolist()), store.encoder.encode(clean)
    else:
        X_all, X_new = np.array(dfp["text"].tolist(), dtype=object), np.array(clean, dtype=object)
    dev = split.dev_idx
    preds = [spec.factory(params, s).fit(X_all[dev], y[dev]).predict(X_new) for s in spec.seeds]
    return [float(v) for v in np.mean(preds, axis=0)]


def verify_project(output_dir: str | Path, project: str, csv_path: str | None = None, tol: float = 1e-6) -> dict:
    """Recompute `project` in a temporary directory and compare test MAE/MdAE with the stored result."""
    cfg, res, dfp = _load_context(output_dir, project, csv_path)
    store = EmbeddingStore(build_encoder(cfg.embedding), cfg.embedding)
    specs = build_method_specs(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for sub in ("projects", "predictions", "splits"):
            (tmp_path / sub).mkdir()
        new = run_project(res["project"], dfp, cfg, store, specs, tmp_path)
    rows, ok = [], True
    for m, d in res["methods"].items():
        for metric in ("mae", "mdae"):
            old_v, new_v = d["test"][metric], new["methods"][m]["test"][metric]
            diff = abs(old_v - new_v)
            good = diff <= tol
            ok &= good
            rows.append({"method": m, "metric": metric, "stored": old_v, "recomputed": new_v, "abs_diff": diff, "match": good})
    return {"project": res["project"], "all_match": bool(ok), "tolerance": tol, "rows": rows}

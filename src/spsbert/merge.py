"""Merge result folders from several sessions/batches (e.g. several Kaggle runs) into one folder."""
from __future__ import annotations

import shutil
from pathlib import Path

from .utils import read_json, slugify


def merge_results(sources: list, dest: str | Path) -> dict:
    """Copy per-project results of `sources` into `dest`. All results must share one config hash (same experiment)."""
    dest = Path(dest)
    for sub in ("projects", "predictions", "splits", "tables"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    seen: dict = {}
    hashes = set()
    for src in map(Path, sources):
        for f in sorted((src / "projects").glob("*.json")):
            r = read_json(f)
            hashes.add(r["config_hash"])
            old = seen.get(r["project"])
            if old and old[1]["data_hash"] != r["data_hash"]:
                raise ValueError(f"Project {r['project']} appears in two sources with different data")
            if old is None or f.stat().st_mtime > old[0].stat().st_mtime:
                seen[r["project"]] = (f, r, src)
    if not seen:
        raise FileNotFoundError("No per-project results found in the given sources")
    if len(hashes) != 1:
        raise ValueError(f"Sources come from different configurations (config hashes {sorted(hashes)}); re-run with one config")
    for name, (f, r, src) in seen.items():
        slug = slugify(name)
        shutil.copy(f, dest / "projects" / f.name)
        for sub, ext in (("predictions", "csv"), ("splits", "json")):
            shutil.copy(src / sub / f"{slug}.{ext}", dest / sub / f"{slug}.{ext}")
    first = Path(sources[0])
    for rel in ("run_metadata.json", "data_cleaning_report.json", "tables/dataset_summary.csv"):
        if (first / rel).exists():
            shutil.copy(first / rel, dest / rel)
    return {"projects": sorted(seen), "n": len(seen), "config_hash": hashes.pop()}

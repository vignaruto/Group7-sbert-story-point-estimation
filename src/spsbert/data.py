"""Loading, validating and cleaning the TAWOS issue table.

Expected CSV columns (case/spacing-insensitive, aliases accepted):
    Project_Name, Issue_Key, Title, Description_Text, Creation_Date, Story_Point
"""
from __future__ import annotations

import csv
import difflib
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DataConfig

LOGGER = logging.getLogger("spsbert.data")
_LOGGED: set = set()      # notices already shown in this process (data is reloaded by several steps)


def _log_once(level: int, msg: str, *args) -> None:
    key = (level, msg % args if args else msg)
    if key not in _LOGGED:
        _LOGGED.add(key)
        LOGGER.log(level, msg, *args)

COLUMN_ALIASES: dict[str, list[str]] = {
    "project": ["project_name", "project", "project_key"],
    "issue_key": ["issue_key", "key", "issue_id", "id"],
    "title": ["title", "summary"],
    "description": ["description_text", "description", "desc"],
    "created": ["creation_date", "created", "created_at"],
    "story_point": ["story_point", "story_points", "storypoint", "sp"],
}
REQUIRED = ["project", "issue_key", "title", "story_point"]

_CODE_BLOCK = re.compile(r"\{(code|noformat)[^}]*\}.*?\{\1\}", re.DOTALL | re.IGNORECASE)
_JIRA_MACRO = re.compile(
    r"\{(?:color|panel|quote|noformat|code|anchor|toc|html|info|warning|note|tip)(?::[^}]*)?\}", re.IGNORECASE
)
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"\[~[^\]]+\]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")


def _norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def clean_text(s: Any) -> str:
    """Light, label-free normalisation (SBERT is trained on natural text, so we keep it natural).

    Removes code/noformat blocks and Jira macros, HTML tags, user mentions; URLs -> 'URL';
    collapses whitespace. Returns '' for NaN/None.
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = _CODE_BLOCK.sub(" ", s)
    s = _JIRA_MACRO.sub(" ", s)
    s = _HTML_TAG.sub(" ", s)
    s = _MENTION.sub(" ", s)
    s = _URL.sub(" URL ", s)
    s = _CTRL.sub(" ", s)
    return _WS.sub(" ", s).strip()


def compose_text(title: str, description: str) -> str:
    if title and description:
        sep = "" if title.endswith((".", "!", "?", ":")) else "."
        return f"{title}{sep} {description}"
    return title or description


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Read the CSV and map columns to canonical names; fail loudly if required ones are missing."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create it with scripts/export_tawos_to_csv.py (see README, section 'Data')."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    norm = {_norm_col(c): c for c in df.columns}
    mapping: dict[str, str] = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in norm:
                mapping[canon] = norm[a]
                break
    missing = [c for c in REQUIRED if c not in mapping]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s) {missing}. Found columns: {list(df.columns)}. "
            f"Accepted aliases: { {k: COLUMN_ALIASES[k] for k in missing} }"
        )
    out = pd.DataFrame({canon: df[src] for canon, src in mapping.items()})
    for opt in ("description", "created"):
        if opt not in out:
            LOGGER.warning("Optional column '%s' not found; continuing without it.", opt)
            out[opt] = np.nan
    return out


def read_project_list(path: str | Path) -> list[str]:
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def _key(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().casefold()


def read_aliases(path: str | Path) -> dict:
    """{casefolded guideline name: (TAWOS name, basis)} from config/project_aliases.csv ('#' lines ignored)."""
    out: dict = {}
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    for row in csv.DictReader(lines):
        out[_key(row["guideline_name"])] = (row["tawos_name"].strip(), row.get("basis", "").strip())
    return out


def build_dataset(cfg: DataConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (clean dataframe, cleaning report).

    Columns: project, issue_key, text, created, story_point. Sorted by (project, issue_key)
    so that the downstream random splits are independent of the CSV row order.
    """
    raw = load_raw(cfg.csv_path)
    report: dict[str, Any] = {"rows_raw": int(len(raw))}

    df = raw.copy()
    df["story_point"] = pd.to_numeric(df["story_point"], errors="coerce")
    ok = np.isfinite(df["story_point"]) & (df["story_point"] > 0)
    report["dropped_missing_or_nonpositive_story_point"] = int((~ok).sum())
    df = df[ok]

    report["raw_story_point_max"] = float(df["story_point"].max()) if len(df) else None
    if cfg.max_story_point is not None:
        keep = df["story_point"] <= cfg.max_story_point
        report["dropped_above_max_story_point"] = int((~keep).sum())
        report["dropped_above_max_by_project"] = {str(k): int(v) for k, v in df[~keep].groupby("project").size().items()}
        _log_once(logging.INFO, "Outlier policy: %d issues with story points > %s removed (raw maximum %s)", int((~keep).sum()),
                  cfg.max_story_point, report["raw_story_point_max"])
        df = df[keep]

    df["project"] = df["project"].astype(str).str.strip()
    df["issue_key"] = df["issue_key"].astype(str).str.strip()
    title = df["title"].map(clean_text)
    desc = df["description"].map(clean_text)
    df["text"] = [compose_text(t, d) for t, d in zip(title, desc)]
    empty = df["text"].str.len() == 0
    report["dropped_empty_text"] = int(empty.sum())
    df = df[~empty]

    df = df.sort_values(["project", "issue_key"], kind="mergesort")
    before = len(df)
    df = df.drop_duplicates(subset=["project", "issue_key"], keep="first")
    report["dropped_duplicate_issue_key"] = int(before - len(df))
    if cfg.deduplicate:
        before = len(df)
        df = df.drop_duplicates(subset=["project", "text"], keep="first")
        report["dropped_duplicate_text"] = int(before - len(df))

    # project selection
    wanted: list[str] | None = cfg.projects
    if wanted is None and cfg.projects_file:
        wanted = read_project_list(cfg.projects_file)
    if wanted is not None:
        by_key = {_key(p): p for p in df["project"].unique()}
        aliases = read_aliases(cfg.aliases_file) if cfg.aliases_file else {}
        matched, unmatched, mapping, inferred = [], [], {}, []
        for w in wanted:
            k = _key(w)
            if k in by_key:
                matched.append(w)
            elif k in aliases and _key(aliases[k][0]) in by_key:
                matched.append(aliases[k][0])
                mapping[w] = aliases[k][0]
                if aliases[k][1] == "inferred":
                    inferred.append(f"{w} -> {aliases[k][0]}")
            else:
                unmatched.append(w)
        report["project_name_mapping"] = mapping
        report["inferred_name_mappings"] = inferred
        if inferred:
            _log_once(logging.WARNING, "Project names matched by ELIMINATION (confirm with your instructor; the numbers are unaffected, only the label): %s", inferred)
        if unmatched:
            known = list(by_key)
            sugg = {u: [by_key[k] for k in difflib.get_close_matches(_key(u), known, n=2, cutoff=0.6)] for u in unmatched}
            sugg = {u: v for u, v in sugg.items() if v}
            _log_once(logging.WARNING, "Requested projects not found in data: %s. Closest names present: %s", unmatched, sugg)
            report["requested_projects_suggestions"] = sugg
        report["requested_projects_not_found"] = unmatched
        df = df[df["project"].map(_key).isin({_key(m) for m in matched})]

    counts = df.groupby("project").size()
    small = counts[counts < cfg.min_issues]
    report["skipped_projects_too_small"] = {k: int(v) for k, v in small.items()}
    df = df[df["project"].isin(counts[counts >= cfg.min_issues].index)]

    df = df[["project", "issue_key", "text", "created", "story_point"]].reset_index(drop=True)
    report["rows_final"] = int(len(df))
    report["projects_final"] = int(df["project"].nunique())
    if df.empty:
        raise ValueError(f"No project has >= {cfg.min_issues} usable issues. Cleaning report: {report}")
    return df, report


def project_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-project descriptive statistics of the story-point target."""
    rows = []
    for p, g in df.groupby("project", sort=True):
        sp = g["story_point"].to_numpy(float)
        vals, cnts = np.unique(sp, return_counts=True)
        rows.append(
            {
                "project": p,
                "n_issues": len(g),
                "sp_mean": sp.mean(),
                "sp_median": float(np.median(sp)),
                "sp_std": sp.std(ddof=1) if len(sp) > 1 else 0.0,
                "sp_min": sp.min(),
                "sp_max": sp.max(),
                "n_distinct_sp": len(vals),
                "mode_sp": float(vals[cnts.argmax()]),
                "mode_share": float(cnts.max() / len(sp)),
                "mean_text_words": float(g["text"].str.split().str.len().mean()),
            }
        )
    return pd.DataFrame(rows)

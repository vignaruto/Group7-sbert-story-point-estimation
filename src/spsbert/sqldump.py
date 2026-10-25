"""Convert a mysqldump-style TAWOS dump (.sql / .sql.gz / .zip) to the issues CSV WITHOUT needing MySQL.

Streams the file, parses only `CREATE TABLE` / `INSERT INTO` of the Issue and Project tables, keeps issues that
have a story point, and joins project names. Memory use is small (only story-point issues are kept).
Tested on mysqldump-format fixtures (quotes, escapes, NULLs, multi-row inserts); not verified against the real TAWOS file.
"""
from __future__ import annotations

import csv
import gzip
import io
import re
import zipfile
from pathlib import Path
from typing import Iterator

ISSUE_ALIASES = {
    "project_id": ["project_id"], "issue_key": ["issue_key", "key"], "title": ["title", "summary"],
    "description": ["description_text", "description"], "created": ["creation_date", "created"],
    "story_point": ["story_point", "story_points"],
}
PROJECT_ALIASES = {"id": ["id"], "name": ["name", "project_name"]}
OUT_COLUMNS = ["Project_Name", "Issue_Key", "Title", "Description_Text", "Creation_Date", "Story_Point"]

_STR = re.compile(r"(?:_[A-Za-z0-9]+\s*)?'((?:[^'\\]|\\.|'')*)'", re.S)
_HEX = re.compile(r"0[xX]([0-9A-Fa-f]*)")
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_NULL = re.compile(r"NULL\b", re.I)
_WS = re.compile(r"\s*")
_ESC = {"0": "\0", "n": "\n", "r": "\r", "t": "\t", "b": "\b", "Z": "\x1a", "'": "'", '"': '"', "\\": "\\"}
_UNESC = re.compile(r"\\(.)|''", re.S)


class _Incomplete(Exception):
    """The statement continues on the next line."""


def _unescape(s: str) -> str:
    return _UNESC.sub(lambda m: "'" if m.group(0) == "''" else _ESC.get(m.group(1), m.group(1)), s)


def _value(text: str, pos: int) -> tuple:
    if pos >= len(text):
        raise _Incomplete
    m = _STR.match(text, pos)
    if m:
        return _unescape(m.group(1)), m.end()
    if text[pos] == "'" or text[pos] == "_":
        raise _Incomplete                          # opening quote without a closing one on this line
    m = _NULL.match(text, pos)
    if m:
        return None, m.end()
    m = _HEX.match(text, pos)
    if m:
        raw = bytes.fromhex(m.group(1) if len(m.group(1)) % 2 == 0 else "0" + m.group(1))
        return raw.decode("utf-8", "replace"), m.end()
    m = _NUM.match(text, pos)
    if m:
        return m.group(0), m.end()
    raise ValueError(f"Cannot parse SQL value near: {text[pos:pos + 60]!r}")


def parse_values(text: str, pos: int) -> list:
    """Parse `(v,v,..),(v,..);` starting at `pos`. Raises _Incomplete if the text ends mid-statement."""
    rows: list = []
    while True:
        pos = _WS.match(text, pos).end()
        if pos >= len(text):
            raise _Incomplete
        if text[pos] == ";":
            return rows
        if text[pos] == ",":
            pos += 1
            continue
        if text[pos] != "(":
            raise ValueError(f"Expected '(' near: {text[pos:pos + 60]!r}")
        pos += 1
        row: list = []
        while True:
            pos = _WS.match(text, pos).end()
            val, pos = _value(text, pos)
            row.append(val)
            pos = _WS.match(text, pos).end()
            if pos >= len(text):
                raise _Incomplete
            if text[pos] == ",":
                pos += 1
            elif text[pos] == ")":
                pos += 1
                break
            else:
                raise ValueError(f"Expected ',' or ')' near: {text[pos:pos + 60]!r}")
        rows.append(row)


def _open(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace", newline="")
    if path.suffix == ".zip":
        zf = zipfile.ZipFile(path)
        name = next(n for n in zf.namelist() if n.lower().endswith(".sql"))
        return io.TextIOWrapper(zf.open(name), encoding="utf-8", errors="replace", newline="")
    return open(path, "r", encoding="utf-8", errors="replace", newline="")


def _read_columns(lines: Iterator[str]) -> list:
    cols = []
    for line in lines:
        if line.startswith(")"):
            break
        m = re.match(r"\s*`([^`]+)`", line)
        if m:
            cols.append(m.group(1))
    return cols


def _index(names: list, aliases: dict, table: str, need: tuple) -> dict:
    low = {n.lower(): i for i, n in enumerate(names)}
    idx = {}
    for canon, al in aliases.items():
        idx[canon] = next((low[a] for a in al if a in low), None)
    missing = [k for k in need if idx[k] is None]
    if missing:
        raise ValueError(f"Table {table}: cannot find column(s) for {missing}. Columns present: {names}")
    return idx


def convert(dump_path: str | Path, out_csv: str | Path) -> dict:
    """Write `out_csv` (Project_Name, Issue_Key, Title, Description_Text, Creation_Date, Story_Point); return statistics."""
    dump_path = Path(dump_path)
    cols: dict = {}
    issues: list = []
    projects: dict = {}
    stats: dict = {"issue_rows": 0, "issues_with_story_point": 0, "issues_unknown_project": 0, "written": 0}
    ix_i = ix_p = None
    wanted = {"issue", "project"}
    with _open(dump_path) as f:
        lines = iter(f)
        for line in lines:
            if line.startswith("CREATE TABLE"):
                m = re.match(r"CREATE TABLE (?:IF NOT EXISTS )?`([^`]+)`", line)
                if m and m.group(1).lower() in wanted:
                    cols[m.group(1).lower()] = _read_columns(lines)
                continue
            if not line.startswith("INSERT INTO"):
                continue
            m = re.match(r"INSERT\s+(?:IGNORE\s+)?INTO\s+`([^`]+)`\s*(\(([^)]*)\))?\s*VALUES\s*", line, re.I)
            if not m or m.group(1).lower() not in wanted:
                continue
            table = m.group(1).lower()
            names = [c.strip().strip("`") for c in m.group(3).split(",")] if m.group(3) else cols.get(table)
            if not names:
                raise ValueError(f"INSERT into {table} seen before its CREATE TABLE and without a column list")
            text, start = line, m.end()
            while True:
                try:
                    rows = parse_values(text, start)
                    break
                except _Incomplete:
                    try:
                        text += next(lines)
                    except StopIteration:
                        raise ValueError("Dump ends in the middle of an INSERT statement (truncated file?)") from None
            if table == "issue":
                ix_i = _index(names, ISSUE_ALIASES, "Issue", ("project_id", "issue_key", "title", "story_point"))
                for r in rows:
                    stats["issue_rows"] += 1
                    if len(r) != len(names):
                        raise ValueError(f"Issue row has {len(r)} values but {len(names)} columns")
                    sp = r[ix_i["story_point"]]
                    if sp is None or sp == "":
                        continue
                    stats["issues_with_story_point"] += 1
                    g = lambda k: r[ix_i[k]] if ix_i.get(k) is not None else None  # noqa: E731
                    issues.append((g("project_id"), g("issue_key"), g("title"), g("description"), g("created"), sp))
            else:
                ix_p = _index(names, PROJECT_ALIASES, "Project", ("id", "name"))
                for r in rows:
                    projects[r[ix_p["id"]]] = r[ix_p["name"]]
    if ix_i is None:
        raise ValueError("No INSERT INTO `Issue` found - is this the TAWOS dump?")
    out = []
    for pid, key, title, desc, created, sp in issues:
        name = projects.get(pid)
        if name is None:
            stats["issues_unknown_project"] += 1
            continue
        out.append([name, key, title, desc, created, sp])
    out.sort(key=lambda r: (str(r[0]), str(r[1])))
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(OUT_COLUMNS)
        w.writerows(out)
    stats["written"] = len(out)
    stats["projects"] = len({r[0] for r in out})
    return stats

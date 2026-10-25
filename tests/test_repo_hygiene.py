"""Repository hygiene: docs must not lie (paths, commands), imports must be installable, no leftovers."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def test_every_repo_path_mentioned_in_docs_exists():
    pat = re.compile(r"(?<![\w/.-])((?:config|docs|scripts|src/spsbert|tests|notebooks)/[A-Za-z0-9_./-]+\.(?:md|yaml|yml|py|csv|ipynb|txt|pdf))")
    bad = []
    for f in DOCS:
        for m in pat.findall(f.read_text(encoding="utf-8")):
            if "*" not in m and not (ROOT / m).exists():
                bad.append((f.name, m))
    assert not bad, bad


def test_every_cli_command_shown_in_docs_exists():
    cmds = set(re.findall(r'add_parser\("(\w+)"', (ROOT / "src" / "spsbert" / "cli.py").read_text(encoding="utf-8")))
    texts = [f.read_text(encoding="utf-8") for f in DOCS] + [(ROOT / "notebooks" / "Group7_PS5_SBERT_StoryPoints.ipynb").read_text(encoding="utf-8")]
    used = {m for t in texts for m in re.findall(r"python -m spsbert (\w+)", t)}
    assert used and used <= cmds, used - cmds


def test_third_party_imports_are_listed_in_requirements():
    names = {"sklearn": "scikit-learn", "yaml": "pyyaml", "sentence_transformers": "sentence-transformers"}
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    missing = set()
    for f in [*(ROOT / "src" / "spsbert").glob("*.py"), ROOT / "scripts" / "export_tawos_to_csv.py", ROOT / "scripts" / "ps_selection_scores.py"]:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            for m in mods:
                root = m.split(".")[0]
                if root and root not in sys.stdlib_module_names and root != "spsbert":
                    if names.get(root, root).lower() not in req:
                        missing.add((f.name, root))
    assert not missing, missing


def test_no_leftover_markers_and_deliverable_files_are_not_git_ignored():
    for f in [*DOCS, *(ROOT / "src" / "spsbert").glob("*.py"), *(ROOT / "config").glob("*.yaml")]:
        assert not re.search(r"\b(TODO|FIXME|XXX)\b", f.read_text(encoding="utf-8")), f.name
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for must_commit in ("results/", "notebooks/", "docs/", "config/", "src/", "README.md"):
        assert must_commit not in ignore

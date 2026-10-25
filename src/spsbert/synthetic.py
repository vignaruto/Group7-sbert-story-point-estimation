"""Synthetic Jira-like data for OFFLINE smoke tests and unit tests ONLY. Never use for reported results."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EASY = ["typo", "label", "tooltip", "rename", "color", "spacing", "icon", "text", "wording", "link"]
HARD = ["refactor", "migration", "concurrency", "architecture", "security", "performance", "cluster", "protocol"]
FILL = ["user", "page", "button", "config", "login", "service", "module", "update", "report", "export",
        "list", "field", "error", "request", "cache", "query", "build", "test", "api", "view"]
SP_SCALE = [1, 2, 3, 5, 8, 13]


def make_synthetic(out_csv: str | Path, n_projects: int = 3, sizes: tuple = (160, 220, 300), seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_projects):
        name = f"SYNTH-{chr(65 + p)}"
        n = sizes[p % len(sizes)]
        for i in range(n):
            n_hard = int(rng.integers(0, 4))
            words_t = list(rng.choice(FILL, 3)) + list(rng.choice(EASY, 1)) + list(rng.choice(HARD, n_hard))
            words_d = list(rng.choice(FILL, int(rng.integers(8, 30)))) + list(rng.choice(HARD, n_hard))
            rng.shuffle(words_t); rng.shuffle(words_d)
            level = min(len(SP_SCALE) - 1, n_hard * 2 + int(rng.integers(0, 2)))
            sp = SP_SCALE[level]
            rows.append({"Project_Name": name, "Issue_Key": f"{name[-1]}-{i + 1}",
                         "Title": " ".join(words_t).capitalize(),
                         "Description_Text": " ".join(words_d) + ".",
                         "Creation_Date": f"2020-{1 + i % 12:02d}-{1 + i % 27:02d}", "Story_Point": sp})
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv

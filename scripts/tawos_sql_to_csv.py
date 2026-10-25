#!/usr/bin/env python
"""TAWOS .sql / .sql.gz / .zip dump  ->  data/tawos_issues.csv  (no MySQL needed).

    python scripts/tawos_sql_to_csv.py path/to/TAWOS.sql --out data/tawos_issues.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spsbert.sqldump import convert  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("--out", default="data/tawos_issues.csv")
    a = ap.parse_args()
    s = convert(a.dump, a.out)
    print(f"Wrote {s['written']} issues with story points from {s['projects']} projects to {a.out}")
    print("Details:", s)

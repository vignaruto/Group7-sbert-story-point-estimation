#!/usr/bin/env python
"""Export Story-Point issues from a MySQL import of the TAWOS dump to data/tawos_issues.csv.

Steps: download TAWOS (https://doi.org/10.5522/04/21308124), import the SQL file into MySQL, then
    python scripts/export_tawos_to_csv.py --host localhost --user root --database tawos --out data/tawos_issues.csv
(password via --password or env MYSQL_PWD). Needs: pip install pymysql
Table/column names follow the TAWOS schema (Issue, Project); if your dump differs, adjust QUERY.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

QUERY = """
SELECT p.Name AS Project_Name, i.Issue_Key AS Issue_Key, i.Title AS Title,
       i.Description_Text AS Description_Text, i.Creation_Date AS Creation_Date, i.Story_Point AS Story_Point
FROM Issue i JOIN Project p ON i.Project_ID = p.ID
WHERE i.Story_Point IS NOT NULL
ORDER BY p.Name, i.Issue_Key
"""


def export_issues(conn, out_csv: str | Path, query: str = QUERY) -> int:
    """Write the query result using any DB-API connection; returns the number of rows."""
    cur = conn.cursor()
    cur.execute(query)
    cols = [d[0] for d in cur.description]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in cur:
            w.writerow(row)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=3306)
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", default=os.environ.get("MYSQL_PWD", ""))
    ap.add_argument("--database", default="tawos")
    ap.add_argument("--out", default="data/tawos_issues.csv")
    a = ap.parse_args()
    import pymysql  # type: ignore

    conn = pymysql.connect(host=a.host, port=a.port, user=a.user, password=a.password, database=a.database,
                           charset="utf8mb4")
    try:
        print(f"Exported {export_issues(conn, a.out)} rows to {a.out}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

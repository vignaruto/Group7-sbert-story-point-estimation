"""Command-line interface:  python -m spsbert <run|aggregate|report|synthetic> ..."""
from __future__ import annotations

import argparse
import sys

from .config import load_config
from .utils import setup_logging


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="spsbert", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run experiments for all (or selected) projects")
    r.add_argument("--config", default="config/default.yaml")
    r.add_argument("--projects", nargs="*", help="only these project names")
    r.add_argument("--force", action="store_true", help="recompute even if cached results exist")
    r.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="key=value",
                   help="override config values, e.g. --set embedding.batch_size=32")

    a = sub.add_parser("aggregate", help="build tables and figures from results/")
    a.add_argument("--output-dir", default="results")

    p = sub.add_parser("report", help="build the final PDF report from results/")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--github-url", required=True)
    p.add_argument("--out", default="results/Group7_Final_Report.pdf")

    d = sub.add_parser("predict", help="demo: predict story points for new issue text(s) with a finished project's model")
    d.add_argument("--output-dir", default="results")
    d.add_argument("--project", required=True)
    d.add_argument("--text", nargs="+", required=True, help="one or more issue texts (title + description)")
    d.add_argument("--method", default="sbert_cnn")
    d.add_argument("--csv", default=None, help="override the data CSV path stored in the run metadata")

    v = sub.add_parser("verify", help="recompute a project and check it reproduces the stored results")
    v.add_argument("--output-dir", default="results")
    v.add_argument("--project", required=True)
    v.add_argument("--csv", default=None)
    v.add_argument("--tol", type=float, default=1e-6)

    k = sub.add_parser("check", help="strict audit of results/ and the repo against the assignment (exit 1 on FAIL)")
    k.add_argument("--output-dir", default="results")
    k.add_argument("--repo-root", default=".")
    k.add_argument("--report", default=None, help="path of the final report PDF (default results/Group7_Final_Report.pdf)")

    q = sub.add_parser("sql2csv", help="convert the TAWOS .sql/.sql.gz/.zip dump to data/tawos_issues.csv (no MySQL needed)")
    q.add_argument("dump")
    q.add_argument("--out", default="data/tawos_issues.csv")

    m = sub.add_parser("merge", help="merge result folders of several batches/sessions, then rebuild tables and figures")
    m.add_argument("--sources", nargs="+", required=True)
    m.add_argument("--to", default="results")

    sub.add_parser("doctor", help="preflight: check packages, GPU, RAM, disk, internet, data, speed")

    s = sub.add_parser("synthetic", help="write a SYNTHETIC csv for offline smoke tests")
    s.add_argument("--out", default="data/synthetic_issues.csv")

    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):      # Windows consoles default to cp1252: never crash on printing
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    setup_logging("INFO")

    if args.cmd == "run":
        from .experiment import run_all

        cfg = load_config(args.config, args.overrides)
        setup_logging("INFO", f"{cfg.output_dir}/run.log")
        run_all(cfg, only=args.projects, force=args.force)
        from .aggregate import aggregate

        aggregate(cfg.output_dir)
    elif args.cmd == "aggregate":
        from .aggregate import aggregate

        aggregate(args.output_dir)
    elif args.cmd == "report":
        from .report import build_report

        build_report(args.output_dir, args.out, args.github_url)
    elif args.cmd == "predict":
        from .demo import predict_texts

        for t, v in zip(args.text, predict_texts(args.output_dir, args.project, args.text, args.method, args.csv)):
            print(f"{v:6.2f} story points  <-  {t[:90]!r}")
    elif args.cmd == "verify":
        from .demo import verify_project

        r = verify_project(args.output_dir, args.project, args.csv, args.tol)
        for row in r["rows"]:
            print(f"{row['method']:16s} {row['metric']:5s} stored={row['stored']:.6f} recomputed={row['recomputed']:.6f} "
                  f"{'OK' if row['match'] else 'MISMATCH'}")
        print("REPRODUCIBLE" if r["all_match"] else "NOT REPRODUCED (see mismatches; different hardware can cause tiny differences)")
        return 0 if r["all_match"] else 1
    elif args.cmd == "sql2csv":
        from .sqldump import convert

        st = convert(args.dump, args.out)
        print(f"Wrote {st['written']} issues with story points from {st['projects']} projects to {args.out}\nDetails: {st}")
    elif args.cmd == "merge":
        from .aggregate import aggregate
        from .merge import merge_results

        info = merge_results(args.sources, args.to)
        aggregate(args.to)
        print(f"Merged {info['n']} projects into {args.to}: {info['projects']}")
    elif args.cmd == "doctor":
        from .doctor import run_doctor

        rows = run_doctor()
        for r in rows:
            print(f"[{r['status']:4s}] {r['check']}: {r['detail']}")
        return 1 if any(r["status"] == "FAIL" for r in rows) else 0
    elif args.cmd == "check":
        from .check import format_checks, run_checks

        checks = run_checks(args.output_dir, args.repo_root, args.report)
        print(format_checks(checks))
        return 1 if any(c.status == "FAIL" for c in checks) else 0
    elif args.cmd == "synthetic":
        from .synthetic import make_synthetic

        print("wrote", make_synthetic(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

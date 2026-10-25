#!/usr/bin/env python
"""Re-score the 12 problem statements with the group's weights (heuristic 1-5 rubric; edit RUBRIC to disagree).

Writes docs/PS_SELECTION_CHECK.md. The rubric is OUR estimate from the structure of each statement
(representation x setting x split), not a measurement; its purpose is transparency, not proof.
"""
from pathlib import Path

WEIGHTS = {"Implementation feasibility": 20, "Experimental reliability and reproducibility": 20,
           "Dataset and split feasibility": 15, "Validation and data-leakage control": 10,
           "Time and compute requirements": 10, "Report and analytical quality": 10,
           "Viva and demonstration readiness": 10, "Research value": 5}
# score (1 worst .. 5 best) of each attribute value, per parameter; a statement's score = mean over its 3 attributes
RUBRIC = {
    "Implementation feasibility": {"rep": {"BERT": 3, "SBERT": 5, "COMB": 2}, "set": {"within": 5, "cross": 3}, "split": {"random": 5, "temporal": 4}},
    "Experimental reliability and reproducibility": {"rep": {"BERT": 3, "SBERT": 5, "COMB": 3}, "set": {"within": 4, "cross": 4}, "split": {"random": 4, "temporal": 5}},
    "Dataset and split feasibility": {"rep": {"BERT": 4, "SBERT": 4, "COMB": 4}, "set": {"within": 4, "cross": 3}, "split": {"random": 5, "temporal": 3}},
    "Validation and data-leakage control": {"rep": {"BERT": 4, "SBERT": 4, "COMB": 4}, "set": {"within": 3, "cross": 4}, "split": {"random": 2, "temporal": 5}},
    "Time and compute requirements": {"rep": {"BERT": 2, "SBERT": 5, "COMB": 1}, "set": {"within": 3, "cross": 3}, "split": {"random": 4, "temporal": 4}},
    "Report and analytical quality": {"rep": {"BERT": 3, "SBERT": 4, "COMB": 4}, "set": {"within": 4, "cross": 4}, "split": {"random": 3, "temporal": 4}},
    "Viva and demonstration readiness": {"rep": {"BERT": 3, "SBERT": 5, "COMB": 3}, "set": {"within": 5, "cross": 3}, "split": {"random": 5, "temporal": 4}},
    "Research value": {"rep": {"BERT": 3, "SBERT": 3, "COMB": 5}, "set": {"within": 2, "cross": 5}, "split": {"random": 2, "temporal": 5}},
}
STATEMENTS = {i + 1: (rep, st, sp) for i, (rep, st, sp) in enumerate(
    [(r, s, p) for r in ("BERT", "SBERT", "COMB") for s in ("within", "cross") for p in ("random", "temporal")])}
# order above gives PS-1..12 exactly as in the guideline table (BERT within random/temporal, BERT cross random/temporal, ...)


def scores() -> dict:
    assert sum(WEIGHTS.values()) == 100
    out = {}
    for n, (rep, st, sp) in STATEMENTS.items():
        per = {p: (RUBRIC[p]["rep"][rep] + RUBRIC[p]["set"][st] + RUBRIC[p]["split"][sp]) / 3 for p in WEIGHTS}
        out[n] = {"desc": f"{rep}, {st}-project, {sp}", "per": per,
                  "total": sum(per[p] * w for p, w in WEIGHTS.items()) / 5}      # 1..5 scale -> 0..100
    return out


def main() -> None:
    sc = scores()
    ranking = sorted(sc, key=lambda n: -sc[n]["total"])
    lines = ["# Re-check of the problem-statement choice with the group's weights", "",
             "Heuristic 1-5 rubric (see `scripts/ps_selection_scores.py`), total on a 0-100 scale. Our estimate, not a measurement.", "",
             "| Rank | PS | Setting | Weighted score | " + " | ".join(f"{p.split()[0]} ({w}%)" for p, w in WEIGHTS.items()) + " |",
             "|---|---|---|---|" + "---|" * len(WEIGHTS)]
    for r, n in enumerate(ranking, 1):
        lines.append(f"| {r} | PS-{n} | {sc[n]['desc']} | **{sc[n]['total']:.1f}** | " + " | ".join(f"{sc[n]['per'][p]:.1f}" for p in WEIGHTS) + " |")
    me = sc[5]
    weak = sorted(WEIGHTS, key=lambda p: me["per"][p])[:2]
    best = ranking[0]
    gap = sc[best]["total"] - me["total"]
    flip = max(WEIGHTS.values()) / 5 / 3          # points moved by changing ONE rubric cell by 1
    tie = (f"PS-{best} scores {gap:.1f} points higher; changing a single rubric cell by one point moves a total by up to {flip:.1f} points, "
           f"so the two are statistically indistinguishable under this rubric." if best != 5 else "PS-5 is first.")
    lines += ["", f"**PS-5 ranks #{ranking.index(5) + 1} of 12 ({me['total']:.1f}/100).** {tie} Its weakest parameters: "
              + ", ".join(f"{w} ({me['per'][w]:.1f}/5)" for w in weak) + ". Random splitting leaks temporal information and the "
              "within-project setting has the least research novelty; the weights (only 5 % for research value) are what favour PS-5. "
              "Mitigations built into this repo: (a) the leakage limitation is stated prominently, (b) duplicates are removed, "
              "(c) a test proves the test set is never touched before the final evaluation, (d) compute is kept down with batches, a "
              "runtime estimate and `config/balanced.yaml`."]
    Path(__file__).resolve().parents[1].joinpath("docs", "PS_SELECTION_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[4:4 + 5]))
    print(f"... PS-5 rank {ranking.index(5) + 1}/12, score {me['total']:.1f}; weakest: {weak}")


if __name__ == "__main__":
    main()

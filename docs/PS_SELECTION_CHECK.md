# Re-check of the problem-statement choice with the group's weights

Heuristic 1-5 rubric (see `scripts/ps_selection_scores.py`), total on a 0-100 scale. Our estimate, not a measurement.

| Rank | PS | Setting | Weighted score | Implementation (20%) | Experimental (20%) | Dataset (15%) | Validation (10%) | Time (10%) | Report (10%) | Viva (10%) | Research (5%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PS-6 | SBERT, within-project, temporal | **85.0** | 4.7 | 4.7 | 3.7 | 4.0 | 4.0 | 4.0 | 4.7 | 3.3 |
| 2 | PS-5 | SBERT, within-project, random | **84.0** | 5.0 | 4.3 | 4.3 | 3.0 | 4.0 | 3.7 | 5.0 | 2.3 |
| 3 | PS-8 | SBERT, cross-project, temporal | **81.7** | 4.0 | 4.7 | 3.3 | 4.3 | 4.0 | 4.0 | 4.0 | 4.3 |
| 4 | PS-7 | SBERT, cross-project, random | **80.7** | 4.3 | 4.3 | 4.0 | 3.3 | 4.0 | 3.7 | 4.3 | 3.3 |
| 5 | PS-2 | BERT, within-project, temporal | **75.7** | 4.0 | 4.0 | 3.7 | 4.0 | 3.0 | 3.7 | 4.0 | 3.3 |
| 6 | PS-10 | COMB, within-project, temporal | **75.0** | 3.7 | 4.0 | 3.7 | 4.0 | 2.7 | 4.0 | 4.0 | 4.0 |
| 7 | PS-1 | BERT, within-project, random | **74.7** | 4.3 | 3.7 | 4.3 | 3.0 | 3.0 | 3.3 | 4.3 | 2.3 |
| 8 | PS-9 | COMB, within-project, random | **74.0** | 4.0 | 3.7 | 4.3 | 3.0 | 2.7 | 3.7 | 4.3 | 3.0 |
| 9 | PS-4 | BERT, cross-project, temporal | **72.3** | 3.3 | 4.0 | 3.3 | 4.3 | 3.0 | 3.7 | 3.3 | 4.3 |
| 10 | PS-12 | COMB, cross-project, temporal | **71.7** | 3.0 | 4.0 | 3.3 | 4.3 | 2.7 | 4.0 | 3.3 | 5.0 |
| 11 | PS-3 | BERT, cross-project, random | **71.3** | 3.7 | 3.7 | 4.0 | 3.3 | 3.0 | 3.3 | 3.7 | 3.3 |
| 12 | PS-11 | COMB, cross-project, random | **70.7** | 3.3 | 3.7 | 4.0 | 3.3 | 2.7 | 3.7 | 3.7 | 4.0 |

**PS-5 ranks #2 of 12 (84.0/100).** PS-6 scores 1.0 points higher; changing a single rubric cell by one point moves a total by up to 1.3 points, so the two are statistically indistinguishable under this rubric. Its weakest parameters: Research value (2.3/5), Validation and data-leakage control (3.0/5). Random splitting leaks temporal information and the within-project setting has the least research novelty; the weights (only 5 % for research value) are what favour PS-5. Mitigations built into this repo: (a) the leakage limitation is stated prominently, (b) duplicates are removed, (c) a test proves the test set is never touched before the final evaluation, (d) compute is kept down with batches, a runtime estimate and `config/balanced.yaml`.

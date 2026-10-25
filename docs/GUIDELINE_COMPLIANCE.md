# Traceability: Project_Guidelines.pdf  ->  code / tests

| Guideline requirement | Where it is implemented | Where it is verified |
|---|---|---|
| §1 Objective: SBERT sentence embeddings for story points, MAE + MdAE | `embeddings.SBERTEncoder`, `metrics.py`, `experiment.py` | `test_core.py::test_mae_mdae_hand_computed`, `test_sbert_encoder_wiring_with_fake_library` |
| PS-5 setting: **within-project**, **random split** | per-project loop in `experiment.run_all`; `splits.make_split` (shuffled `train_test_split`) | `test_split_properties`, `test_end_to_end_smoke_pipeline` |
| §3.1 SBERT for sentence-level representation | `embeddings.py` (`all-mpnet-base-v2`, configurable) | wiring test with stand-in library |
| §3.1 three-layer CNN as prediction component | `models.build_cnn_module` / `CNNRegressor` | `test_cnn_architecture_has_three_conv_layers_and_right_shape`, `test_cnn_learns_dense_signal_and_is_reproducible` |
| §3.1 document configuration, CNN architecture, hyper-parameters, seeds | `config/*.yaml`, `results/run_metadata.json`, `results/projects/*.json` (best params, #parameters, epochs) | `test_results_contain_everything_the_guideline_requires` |
| §3.2 80:20 development/test split | `split.test_size: 0.2`; ceil rounding of the test part | `test_split_properties`, results test (`n_test/n_issues ~ 0.2`) |
| §3.2 five folds over the 80 %, each fold validates once | `splits.make_split` + `assert_valid_split` | `test_split_properties`, `test_no_leakage_between_train_validation_and_test`, fold-count test |
| §3.2 select configuration on development data only; test set held out | `experiment.cross_validate` (dev only) then `final_evaluation` (test once); early stopping uses an inner hold-out of the training part | leakage-spy test (records every fitted/scored index) |
| §3.2 same partitions for fair paired comparison | deterministic split from sorted issue keys + seed; `results/splits/*.json`; all methods share one split | `test_split_is_deterministic_and_seed_sensitive`, spy test |
| §3.4 MAE, MdAE; 5-fold results reported **separately** from held-out test | `tables/validation_*.csv` vs `tables/test_results.csv` | `test_validation_and_test_sets_are_reported_separately_and_differ` |
| §3.4 mean and std of validation metrics across folds | `cv.mae_mean/std`, `mdae_mean/std` (sample std, ddof=1) | audit results test |
| §3.4 table: representation / model configuration / MAE / MdAE | `aggregate.macro_summary`, `report.py` Table A + Table 4 | report test |
| §3.4 discuss lowest MAE/MdAE, limitations, threats | auto-generated Discussion/Limitations in `report.py` (reads measured numbers only) | report test |
| §3.5 README, code, configs, results with metrics, comparative tables, plots | `README.md`, `src/`, `config/`, `results/tables`, `results/figures` | `test_cli_synthetic_aggregate_and_report` |
| §3.6 PDF report with GitHub link on page 1 | `python -m spsbert report --github-url ...` | report test (link before Section 1) |
| §3.6 demonstration + how to reproduce principal results | `python -m spsbert predict ...` (demo), `python -m spsbert verify ...` (recompute and compare) | `test_predict_texts_api_and_cli`, `test_verify_reproduces_stored_results_exactly` |
| §3.7 record seeds, split procedure, model versions, configs | `set_seed`, `environment_info`, config fingerprint, split JSON | `test_two_independent_runs_are_identical` |
| §3.7 do not fabricate results | report text is generated from result files; synthetic/non-SBERT runs are stamped "NOT REAL RESULTS" | report test |
| Guideline data: TAWOS | `scripts/export_tawos_to_csv.py`, `data.py` | `test_export_script_with_sqlite`, loader tests |

## Automated verification
`python -m spsbert check` (see `src/spsbert/check.py`) encodes the rows above as checks A00-A20 and is exercised by `tests/test_check.py`
(including mutation tests). The notebook ends with the same audit (`tests/test_notebook.py`).

## Not applicable to PS-5 (documented, not implemented)
BERT word embeddings, combined/fusion embeddings, temporal splitting and cross-project pairs belong to other problem statements (PS-1..4, 6..12).

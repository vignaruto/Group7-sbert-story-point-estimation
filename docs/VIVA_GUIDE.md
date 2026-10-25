# Viva guide — every member should be able to answer these (guideline §3.7)

## The 60-second explanation
We predict an issue's story points from its text. Each issue is turned into one vector by a frozen Sentence-BERT model; a
three-layer CNN maps the vector to a number. For each project separately we hold out a random 20 % for the final test, use
5-fold cross-validation on the other 80 % to pick hyper-parameters, refit, and score the test set once with MAE and MdAE.
Mean, median, TF-IDF+Ridge and SBERT+Ridge are evaluated on the *same* splits so "effective" means "better than something".

## Likely questions and where the answer lives
| Question | Short answer | Code |
|---|---|---|
| Why a random split, and what is wrong with it? | PS-5 asks for it; it leaks future information (temporal leakage), so scores can be optimistic vs deployment. | `splits.py`, report §7 |
| How do you avoid data leakage? | Test indices are never fitted on or used for selection; scaling/TF-IDF/early stopping use training data only; exact duplicate texts are removed. | `experiment.py`, `tests/test_audit.py::test_no_leakage...` |
| Why 5-fold CV *and* a test set? | CV on the 80 % chooses settings; the 20 % test gives one unbiased final estimate. | `experiment.cross_validate / final_evaluation` |
| Why is early stopping not a leak? | It uses a 10 % hold-out carved from the four *training* folds, never the validation fold. | `models.CNNRegressor.fit` |
| What is SBERT vs BERT? | SBERT is fine-tuned (siamese) so a whole sentence maps to one meaningful vector; BERT gives per-token vectors. | `embeddings.py` |
| Why freeze the encoder? | Small projects; avoids overfitting and label leakage; cheap — embeddings are computed once and cached. | `EmbeddingStore` |
| Why a CNN on a sentence vector? | The guideline allows it; the embedding is treated as a 1-D signal. Ridge on the same vector tests whether the CNN's extra capacity helps. | `models.build_cnn_module` |
| Why MAE *and* MdAE? | MAE is sensitive to large errors; MdAE shows the typical error. They can disagree. | `metrics.py` |
| Why log1p on the target? | Story points are skewed; the transform stabilises training and favours median-like predictions. | `models.py` |
| Why a median baseline? | Story points concentrate on few values; if the CNN cannot beat the median it has learned nothing useful. | `models.ConstantRegressor` |
| How do you know results are significant? | Paired Wilcoxon on per-issue errors, Holm-corrected across projects, bootstrap CIs; treated as descriptive. | `metrics.py`, `aggregate.py` |
| How reproducible is it? | Seeds, config hash, data hash, versions are stored; `python -m spsbert verify` recomputes a project and compares. | `demo.verify_project` |
| What if SBERT does not beat the baselines? | Report it honestly; that is a valid finding and the report text adapts to the numbers. | `report._discussion` |
| Limitations? | Random split, single split per project, near-duplicates, noisy team-specific labels, frozen/truncating encoder. | report §7 |

## Be able to demo live
1. Open the notebook, show Section 3 (split sizes, test ∩ dev = ∅) and Section 7 (per-fold validation).
2. Run `python -m spsbert predict --project "<name>" --text "..."`.
3. Run `python -m spsbert verify --project "<name>"` -> REPRODUCIBLE.
4. Run `python -m spsbert check` and explain each line.

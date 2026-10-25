| method | label | test_mae | test_mdae | val_mae | val_mae_fold_std | val_mdae | val_mdae_fold_std | projects_lowest_mae | projects_lowest_mdae |
|---|---|---|---|---|---|---|---|---|---|
| tfidf_ridge | TF-IDF+Ridge | 1.962 | 1.158 | 1.999 | 0.360 | 1.173 | 0.161 | 21 | 13 |
| sbert_ridge | SBERT+Ridge | 2.017 | 1.227 | 2.028 | 0.361 | 1.180 | 0.148 | 10 | 2 |
| sbert_cnn | SBERT+CNN | 2.063 | 1.199 | 2.058 | 0.364 | 1.177 | 0.158 | 2 | 10 |
| median_baseline | Median | 2.165 | 1.372 | 2.181 | 0.384 | 1.354 | 0.213 | 6 | 11 |
| mean_baseline | Mean | 2.442 | 1.913 | 2.484 | 0.313 | 1.919 | 0.232 | 0 | 3 |
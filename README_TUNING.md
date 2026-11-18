# Phase 2: Bayesian hyperparameter optimisation

## Completed run

The retained Optuna/TPE study completed **20 LightGBM trials** using the 80-trip training partition and five-fold `GroupKFold` by trip. The sampler used seed 42 and 10 startup trials. The search objective was mean fold Macro F1 at a fixed probability threshold of 0.50. Neither student-test nor evaluator-test labels were used to choose the search space or parameters.

Trial 14 was the best completed trial, with fixed-threshold CV Macro F1 of approximately 0.8492. The tuned model did not materially improve the student-test results, so the untuned Phase 1 LightGBM remains the deployed baseline. The student-test comparison below is report-only and was not used to choose the final model.

| Student-test metric | Untuned LightGBM | Tuned LightGBM | Difference (tuned minus baseline) |
|---|---:|---:|---:|
| Top-1 accuracy | 0.815142 | 0.815037 | -0.000105 |
| Candidate Macro F1 | 0.870394 | 0.870463 | +0.000070 |
| Candidate-row accuracy | 0.916503 | 0.915971 | -0.000532 |
| Log loss | 0.191658 | 0.195218 | +0.003560 (worse) |
| Route continuity | 0.897441 | 0.896919 | -0.000522 |

The fixed-threshold tuning objective is not strictly comparable with the Phase 1 CV Macro F1 where the threshold was selected from out-of-fold predictions. The student-test rows above use the same 10 trips and metrics for both LightGBM versions.

## Reproducibility artifacts

| Path | Use |
|---|---|
| `optimisation/studies/lgbm_fixed050.db` | SQLite study with 20 completed trials |
| `optimisation/studies/study_history_lgbm.json` | Serialized Optuna history |
| `optimisation/results/trials_lgbm.csv` | Trial-level metrics |
| `optimisation/results/trial_fold_metrics_lgbm.csv` | Per-trial, per-fold metrics |
| `optimisation/results/best_params_lgbm.json` | Best parameters and fixed-threshold CV summary |
| `optimisation/results/cv_folds_tuned_lgbm.csv` | Tuned fold metrics |
| `optimisation/results/oof_scores_tuned_lgbm.parquet` | Tuned out-of-fold scores |
| `optimisation/results/student_test_once.json` | One-shot, report-only student-test comparison |
| `optimisation/results/baseline_reproduction_lgbm.csv` | Reproduction of the untuned baseline through the tuning harness |
| `optimisation/results/fold_assignment.csv` and `baseline_hashes.json` | Fold record and baseline integrity snapshot |
| `optimisation/artifacts/model_tuned_lgbm.joblib` | Tuned model retained for comparison; not the selected deployment model |
| `artifacts/final_model.joblib` | Selected untuned Phase 1 deployment model |

The `best_params_lgbm.json` objective field is normalized by this cleanup patch to the executed fixed-0.50 objective. The two-trial local smoke database `lgbm.db` is not the authoritative study; use `lgbm_fixed050.db` with the history and trial CSVs. Do not rerun the search for submission. Any future tuning/debugging must use training and student-test data only; evaluator data remains a final holdout.

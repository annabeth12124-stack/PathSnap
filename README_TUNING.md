# PathSnap Phase 2 - Bayesian tuning runbook

**Status: this code has NOT been executed by its author.** Run the smoke test first (step 2) and fix anything it exposes.

Copy the contents of this folder into the repo root (branch `feature/pathsnap-bayesian-tuning`, based on `abd3bb9`).
No existing file is modified. `predict.py` is unchanged: `predict.load_model(art_dir)` and
`predict_trip(trip_dir, model, cfg)` already accept a model from any directory.

## Run (PowerShell, repo root, venv active)

```powershell
pip install -r requirements-ml.txt          # needs lightgbm, optuna, pyarrow, matplotlib
git status                                   # must be clean before the first run (baseline hashes are snapshotted)

# 1) baseline integrity + existing tests
python tests/test_phase1_integration.py

# 2) smoke test (2 trials, one model) - checks the whole pipeline end to end
python -m pathsnap_ml.tuning.run_tuning --models lgbm --trials 2
Remove-Item optimisation\studies\lgbm.db     # discard the smoke study before the real run

# 3) real run: 3 models x 40 trials (resumable: re-run the same command to continue)
python -m pathsnap_ml.tuning.run_tuning

# 4) refit best params on all 80 trips, save tuned artifacts + OOF + fold metrics
python -m pathsnap_ml.tuning.refit

# 5) baseline-vs-tuned comparison, selection, plots, final tuned artifact
python -m pathsnap_ml.tuning.report

# 6) tests
python -m pytest tests/test_tuned_model.py
python tests/test_phase1_integration.py

# 7) ONE-SHOT, report-only student_test scoring (refuses to run twice)
python -m pathsnap_ml.tuning.student_once

git status                                   # only optimisation/, pathsnap_ml/tuning/, tests/, README_TUNING.md new
```

## What it does
* Same 80 train trips, same row policy, same 5-fold `GroupKFold` by trip as Phase 1. `verify_folds_match_baseline`
  proves identity by reproducing the Phase-1 fold Macro F1 from the committed OOF files; it aborts otherwise.
* Median imputer is inside the per-fold `Pipeline` (fitted on the training fold only).
* Optuna `TPESampler(seed=42, multivariate=True, n_startup_trials=10)`, maximise mean fold Macro F1 (threshold from pooled
  OOF, exactly as Phase 1). Trial 0 is an (almost) baseline anchor. 40 trials per model (`--trials`).
* Every trial records Macro F1, Macro-F1@0.5, Top-1, log loss, observation-level NLL of `prob_obs`, continuity,
  threshold, wall time, and the fold-level metrics.
* The exact Phase-1 configs are re-run through the same harness (`baseline_reproduction_*.csv`) for a like-for-like check.
* student_test and evaluator data are never used for tuning or selection. `student_once.py` scores the final choice once.
* `artifacts/` and `results/` are sha256-snapshotted before the run and verified after every script.

## Outputs (all under `optimisation/`)
| Path | Content |
|---|---|
| `studies/<key>.db`, `studies/study_history_<key>.json` | Optuna SQLite study + full trial history |
| `results/trials_<key>.csv`, `trial_fold_metrics_<key>.csv` | per-trial metrics; per-trial per-fold metrics |
| `results/best_params_<key>.json` | best hyperparameters + best grouped-CV scores |
| `results/cv_folds_tuned_<key>.csv`, `oof_scores_tuned_<key>.parquet` | tuned fold-level metrics, OOF scores |
| `results/comparison_baseline_vs_tuned.{csv,json}` | deltas, paired differences, folds improved |
| `results/model_selection.json` | scorecard, tolerances, regression guard, noise flags |
| `results/fold_assignment.csv`, `baseline_hashes.json` | folds used; baseline integrity snapshot |
| `plots/` | progress, trade-offs, fold-level, importance, summary |
| `artifacts/model_tuned_<key>.joblib` + `_config.json` | tuned models |
| `artifacts/final_model.joblib`, `final_model_config.json` | selected tuned model (loadable by `predict.load_model`) |

Loading the tuned model:
```python
from pathsnap_ml import predict
model, cfg = predict.load_model("optimisation/artifacts")
out = predict.predict_trip(trip_dir, model=model, cfg=cfg)   # same 10-column contract, prob_obs sums to 1 per observation
```

## Selection rule (declared before results; see `report.py`)
Regression guard vs own baseline, then a tolerance-point scorecard over Macro F1, Top-1, log loss, observation-level NLL
and fold-std of Macro F1; ties -> Macro F1, then log loss. The report also says whether the winner's lead and each
model's gain over its baseline exceed the fold standard deviation. Not "highest single score".

## Compute
Not measured. Phase-1 CV wall times were about 20 s (LightGBM), 15 s (HGB), 53 s (RF) per 5-fold CV on the original machine;
larger search spaces may make a trial 2-5x slower. Per-trial wall time is printed and stored; if the first trials show the
full run is too expensive, stop and decide before reducing trials.

"""Select the final Phase-1 baseline by the PRE-DECLARED primary metric and export artifacts.
Primary metric: 5-fold trip-grouped CV observation-level Top-1 on the 80 training trips (not student-test, not train accuracy).
Student-test is reported as a development check only. No evaluator data, no Optuna."""
import json, shutil, pandas as pd
from .config import *
R = json.load(open("results/model_comparison.json"))
rows = sorted(R.values(), key=lambda r: -r["cv_top1_mean"]); best = rows[0]; second = rows[1]
within_noise = (best["cv_top1_mean"] - second["cv_top1_mean"]) < best["cv_top1_std"]
shutil.copy(f"artifacts/model_{best['key']}.joblib", "artifacts/final_model.joblib")
cfg = dict(phase="Phase 1 baseline (no Bayesian optimisation)", selected_model=best["model"], selected_key=best["key"],
           selection_metric="5-fold trip-grouped CV observation-level Top-1 (80 training trips only)",
           selection_ranking=[(r["model"], round(r["cv_top1_mean"], 4), round(r["cv_top1_std"], 4)) for r in rows],
           top_two_within_one_cv_std=bool(within_noise), features=FEATURES, identifier_columns_excluded=ID_COLS,
           decision_threshold=best["threshold"], hyperparameters=best["config"], seed=SEED,
           candidate_generation=dict(source="Sakshi commit 441f431 preprocessing/candidate_generation.ipynb", projection="local equirectangular, centre = mean track lat/lon, R=6371000",
                                     kdtree="arc midpoints", k_nearby=K_NEARBY, top_candidates=TOP_CANDIDATES, distance_cutoff=None),
           training=dict(trips=80, rows=best["n_train_rows"], row_policy=f"reference-reliable (aligned route arc within {REF_OK_M} m), every {STRIDE}th observation"),
           student_test_metrics={k: best[k] for k in ["test_top1", "test_macro_f1", "test_log_loss", "test_continuity", "inference_us_per_candidate"]})
json.dump(cfg, open("artifacts/final_model_config.json", "w"), indent=1, default=float); print(json.dumps(cfg, indent=1, default=float))

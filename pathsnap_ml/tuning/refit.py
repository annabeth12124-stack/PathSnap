"""Refit each model with its best Optuna parameters.

For every model: (1) re-run grouped CV with the best params on the SAME folds to save fold metrics + OOF scores,
(2) fit on all 80 training trips, (3) save the tuned Pipeline and its config under optimisation/artifacts/.
Writes only inside optimisation/. student_test / evaluator data are not used.

Usage: python -m pathsnap_ml.tuning.refit [--models lgbm,rf,hgb]
"""
import argparse
import json
import time
import warnings

import joblib
import pandas as pd

from ..config import FEATURES, ID_COLS, REF_OK_M, SEED, STRIDE
from . import common as C

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(C.MODEL_KEYS))
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()

    C.make_dirs()
    trs = C.load_training_table()
    X, y, g, splits = C.make_splits(trs)
    C.verify_folds_match_baseline(trs, splits)
    arcs = C.load_arcs_for(trs)

    for key in a.models.split(","):
        best = json.load(open(C.RESULTS / f"best_params_{key}.json"))
        params = best["params"]
        df, thr, wall, oof = C.cross_validate(key, params, trs, X, y, splits, arcs, a.n_jobs)
        rec = float(df.macro_f1_thr050.mean())
        if abs(rec - best["best_value"]) > 2e-3:
            print(f"[{key}] WARNING: recomputed fixed-threshold CV Macro F1 {rec:.4f} != study value {best['best_value']:.4f}")
        df.to_csv(C.RESULTS / f"cv_folds_tuned_{key}.csv", index=False)
        pd.DataFrame(dict(trip_id=trs.trip_id.values, observation_id=trs.observation_id.values,
                          arc_id=trs.arc_id.values, label=y, score=oof)
                     ).to_parquet(C.RESULTS / f"oof_scores_tuned_{key}.parquet")

        t0 = time.time()
        model = C.build_model(key, params, a.n_jobs)
        model.fit(X, y)
        train_time = time.time() - t0
        joblib.dump(model, C.ARTIFACTS / f"model_tuned_{key}.joblib")

        cfg = dict(phase="Phase 2 Bayesian tuning (Optuna TPE)", model=best["model_name"], key=key,
                   hyperparameters=params, decision_threshold=thr, features=FEATURES,
                   identifier_columns_excluded=ID_COLS, seed=SEED, sampler=best["sampler"],
                   best_trial=best["best_trial"], n_trials=best["n_trials_total"],
                   cv_summary=C.summarize(df), cv_protocol="5-fold GroupKFold by trip, 80 train trips, "
                   "median imputer fitted per training fold",
                   training=dict(trips=int(trs.trip_id.nunique()), rows=int(len(trs)),
                                 row_policy=f"reference-reliable (<= {REF_OK_M} m), every {STRIDE}th observation"),
                   train_time_s=train_time)
        json.dump(cfg, open(C.ARTIFACTS / f"model_tuned_{key}_config.json", "w"), indent=1, default=float)
        print(f"[{key}] tuned CV macroF1={rec:.4f} top1={df.top1.mean():.4f} logloss={df.log_loss.mean():.4f} "
              f"-> model_tuned_{key}.joblib", flush=True)


if __name__ == "__main__":
    main()


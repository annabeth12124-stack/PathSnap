"""Bayesian hyperparameter optimisation (Optuna TPE, fixed seed) for LightGBM, Random Forest, HistGradientBoosting.

Objective  : mean fold Macro F1 over 5-fold GroupKFold-by-trip (maximise). Top-1, log loss, obs-level NLL, continuity,
             Macro-F1@0.5 and the pooled threshold are recorded for EVERY trial (+ fold-level metrics).
Never used : student_test, evaluator data.
Resumable  : studies persist in optimisation/studies/<key>.db; re-running continues until --trials are finished.

Usage (repo root):
    python -m pathsnap_ml.tuning.run_tuning                       # 3 models x 40 trials
    python -m pathsnap_ml.tuning.run_tuning --models lgbm --trials 2   # quick smoke run
"""
import argparse
import json
import time
import warnings
from types import SimpleNamespace

import optuna
import pandas as pd
from optuna.samplers import TPESampler

from . import common as C

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_objective(key, ctx):
    def objective(trial):
        params = C.suggest(key, trial)
        df, thr, wall, _ = C.cross_validate(key, params, ctx.trs, ctx.X, ctx.y, ctx.splits, ctx.arcs, ctx.n_jobs)
        s = C.summarize(df)
        trial.set_user_attr("summary", s)
        trial.set_user_attr("threshold", thr)
        trial.set_user_attr("wall_s", wall)
        trial.set_user_attr("fold_metrics", json.loads(df.to_json(orient="records")))
        return s["macro_f1_thr050_mean"]
    return objective


def export(study, key):
    """Write the full trial history (CSV + JSON) and fold-level metrics. Called after every trial."""
    rows, folds, hist = [], [], []
    for t in study.trials:
        r = dict(trial=t.number, state=t.state.name, value=t.value, **{f"param_{k}": v for k, v in t.params.items()})
        r.update(t.user_attrs.get("summary", {}))
        r["threshold"] = t.user_attrs.get("threshold")
        r["wall_s"] = t.user_attrs.get("wall_s")
        rows.append(r)
        for f in t.user_attrs.get("fold_metrics", []):
            folds.append(dict(trial=t.number, **f))
        hist.append(dict(number=t.number, state=t.state.name, value=t.value, params=t.params,
                         user_attrs=t.user_attrs,
                         start=str(t.datetime_start), end=str(t.datetime_complete)))
    pd.DataFrame(rows).to_csv(C.RESULTS / f"trials_{key}.csv", index=False)
    pd.DataFrame(folds).to_csv(C.RESULTS / f"trial_fold_metrics_{key}.csv", index=False)
    json.dump(hist, open(C.STUDIES / f"study_history_{key}.json", "w"), indent=1, default=str)


def write_best(study, key, seed):
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise RuntimeError(f"[{key}] no completed trials - every trial failed; inspect trials_{key}.csv / the log.")
    best = study.best_trial
    out = dict(model=key, model_name=C.MODEL_NAMES[key], best_trial=best.number, best_value=best.value,
               objective="mean fold Macro F1 (5-fold GroupKFold by trip, pooled-OOF threshold)",
               params=best.params, cv_summary=best.user_attrs["summary"], threshold=best.user_attrs["threshold"],
               n_trials_completed=len(done), n_trials_total=len(study.trials),
               sampler=C.SAMPLER_DESC.replace("SEED", str(seed)), seed=seed)
    json.dump(out, open(C.RESULTS / f"best_params_{key}.json", "w"), indent=1, default=float)
    return out


def baseline_reproduction(key, ctx):
    """Re-run the EXACT Phase-1 config through this harness on the same folds and compare to the stored Phase-1 CV."""
    df, thr, wall, _ = C.cross_validate(key, C.BASELINE[key], ctx.trs, ctx.X, ctx.y, ctx.splits, ctx.arcs, ctx.n_jobs)
    stored = pd.read_csv(C.BASE_RESULTS / f"cv_folds_{key}.csv")
    diff = max(abs(df.macro_f1 - stored.macro_f1).max(), abs(df.top1 - stored.top1).max())
    out = df.copy()
    out["stored_macro_f1"], out["stored_top1"] = stored.macro_f1, stored.top1
    out.to_csv(C.RESULTS / f"baseline_reproduction_{key}.csv", index=False)
    flag = "OK" if diff < 1e-3 else "WARNING: differs from stored Phase-1 CV (thread-count / library-version effects?)"
    print(f"[{key}] baseline reproduction max |diff| = {diff:.2e}  {flag}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(C.MODEL_KEYS))
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--skip-baseline-check", action="store_true")
    a = ap.parse_args()

    C.make_dirs()
    C.snapshot_baseline()
    trs = C.load_training_table()
    X, y, g, splits = C.make_splits(trs)
    C.verify_folds_match_baseline(trs, splits)
    ctx = SimpleNamespace(trs=trs, X=X, y=y, splits=splits, arcs=C.load_arcs_for(trs), n_jobs=a.n_jobs)
    print(f"train rows={len(trs)} trips={trs.trip_id.nunique()} folds verified identical to Phase 1", flush=True)

    for key in a.models.split(","):
        if not a.skip_baseline_check:
            baseline_reproduction(key, ctx)
        db = (C.STUDIES / f"{key}_fixed050.db").as_posix()
        study = optuna.create_study(
            study_name=f"pathsnap_{key}_fixed050", storage=f"sqlite:///{db}", direction="maximize", load_if_exists=True,
            sampler=TPESampler(seed=a.seed, multivariate=True, n_startup_trials=10))
        if len(study.trials) == 0:
            study.enqueue_trial(C.ANCHOR[key])
        finished = sum(t.state.is_finished() for t in study.trials)
        remaining = max(0, a.trials - finished)
        print(f"=== {key}: {finished} finished, running {remaining} more (target {a.trials})", flush=True)

        def cb(st, tr, key=key):
            export(st, key)
            v = tr.value if tr.value is not None else float("nan")
            print(f"  [{key}] trial {tr.number:>3} {tr.state.name:<8} macroF1={v:.4f} "
                  f"wall={tr.user_attrs.get('wall_s', float('nan')):.0f}s", flush=True)

        t0 = time.time()
        study.optimize(make_objective(key, ctx), n_trials=remaining, callbacks=[cb], catch=(Exception,))
        export(study, key)
        best = write_best(study, key, a.seed)
        print(f"=== {key} done in {time.time() - t0:.0f}s best trial {best['best_trial']} "
              f"macroF1={best['best_value']:.4f}", flush=True)

    C.verify_baseline_untouched()
    print("baseline artifacts/results verified byte-identical (sha256).")


if __name__ == "__main__":
    main()

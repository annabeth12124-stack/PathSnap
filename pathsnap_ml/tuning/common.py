"""Shared utilities for PathSnap Phase-2 Bayesian tuning (Optuna).

Design rules (all enforced here):
  * Training data = the same 80 train trips / same row policy as Phase 1 (ref_ok, every STRIDE-th observation).
  * Folds = the same 5-fold GroupKFold BY TRIP as Phase 1; verified against the committed baseline OOF files.
  * Preprocessing (median imputer) lives INSIDE the per-fold Pipeline, so it is fitted on the training fold only.
  * Evaluator data is never read (assert_not_evaluator guards every data path); student_test is not used for tuning.
  * Baseline artifacts/ and results/ are never written to (sha256 snapshot taken before, verified after).
"""
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from .. import evaluation as ev
from ..config import CACHE, DATA_ROOT, FEATURES, PROJECT_ROOT, SEED, STRIDE

# ----------------------------------------------------------------------------------------------- paths
OPT = PROJECT_ROOT / "optimisation"
STUDIES = OPT / "studies"
RESULTS = OPT / "results"
PLOTS = OPT / "plots"
ARTIFACTS = OPT / "artifacts"
BASE_RESULTS = PROJECT_ROOT / "results"      # READ ONLY
BASE_ART = PROJECT_ROOT / "artifacts"        # READ ONLY

MODEL_KEYS = ["lgbm", "rf", "hgb"]
MODEL_NAMES = {"lgbm": "LightGBM", "rf": "Random Forest", "hgb": "HistGradientBoosting"}
N_SPLITS = 5
SAMPLER_DESC = "TPESampler(seed=SEED, multivariate=True, n_startup_trials=10)"


def make_dirs():
    for d in (STUDIES, RESULTS, PLOTS, ARTIFACTS):
        d.mkdir(parents=True, exist_ok=True)


def assert_not_evaluator(path):
    """Hard guard: this package must never touch evaluator test data."""
    if "evaluator" in str(path).lower():
        raise RuntimeError(f"Refusing to read evaluator data: {path}")


# ----------------------------------------------------------------------------------------------- baseline configs
# Exact Phase-1 hyperparameters (pathsnap_ml/run_experiments.py), expressed as build_model() params.
BASELINE = {
    "lgbm": dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=0.0),
    "rf": dict(n_estimators=60, max_depth=14, min_samples_leaf=20, max_features="sqrt", max_samples=0.5),
    "hgb": dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0),
}
# First trial enqueued into each study: (almost) the baseline, so TPE starts from a known-good point.
# reg_lambda / l2_regularization cannot be exactly 0 in a log-scale space, so 1e-3 is used. The EXACT baseline is
# evaluated separately by the baseline-reproduction step in run_tuning.py.
ANCHOR = {
    "lgbm": dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1e-3),
    "rf": dict(n_estimators=60, max_depth=14, min_samples_leaf=20, max_features="sqrt", max_samples=0.5),
    "hgb": dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1e-3),
}


def suggest(key, trial):
    """Search spaces. Returns the full parameter dict for build_model()."""
    if key == "lgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    if key == "rf":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 60, 200, step=20),
            max_depth=trial.suggest_int("max_depth", 8, 24),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 60),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            max_samples=trial.suggest_float("max_samples", 0.3, 0.8),
        )
    if key == "hgb":
        return dict(
            max_iter=trial.suggest_int("max_iter", 150, 400, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 100),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        )
    raise KeyError(key)


def build_model(key, params, n_jobs=-1):
    """sklearn Pipeline: median imputer (fit per training fold) + estimator. Same structure as Phase 1."""
    imp = SimpleImputer(strategy="median")
    if key == "lgbm":
        est = lgb.LGBMClassifier(**params, subsample_freq=1, random_state=SEED, n_jobs=n_jobs, verbose=-1)
    elif key == "rf":
        est = RandomForestClassifier(**params, n_jobs=n_jobs, random_state=SEED)
    elif key == "hgb":
        # Phase-1 settings retained: early stopping on an internal 10% validation split.
        est = HistGradientBoostingClassifier(**params, early_stopping=True, validation_fraction=0.1,
                                             random_state=SEED)
    else:
        raise KeyError(key)
    return make_pipeline(imp, est)


# ----------------------------------------------------------------------------------------------- data + folds
def load_training_table():
    """Same row policy as run_experiments.py: ref_ok==1 and observation_id % STRIDE == 0, index reset."""
    p = Path(CACHE) / "train.parquet"
    assert_not_evaluator(p)
    tr = pd.read_parquet(p)
    return tr[(tr.ref_ok == 1) & (tr.observation_id % STRIDE == 0)].reset_index(drop=True)


def load_arcs_for(trs):
    assert_not_evaluator(DATA_ROOT)
    return ev.load_arcs(DATA_ROOT, "train", trs.trip_id.unique())


def make_splits(trs):
    X = trs[FEATURES].values.astype(np.float64)
    y = trs.label.values
    g = trs.trip_id.values
    splits = list(GroupKFold(n_splits=N_SPLITS).split(X, y, g))
    return X, y, g, splits


def verify_folds_match_baseline(trs, splits, keys=MODEL_KEYS):
    """Proves these folds are the Phase-1 folds: row order must match the committed baseline OOF files AND the
    baseline fold Macro F1 must be reproduced exactly from those OOF scores using these splits."""
    thr_all = json.load(open(BASE_RESULTS / "model_comparison.json"))
    for key in keys:
        o = pd.read_parquet(BASE_RESULTS / f"oof_scores_{key}.parquet")
        same_rows = (len(o) == len(trs)
                     and (o.trip_id.values == trs.trip_id.values).all()
                     and (o.observation_id.values == trs.observation_id.values).all()
                     and (o.arc_id.values == trs.arc_id.values).all())
        if not same_rows:
            raise RuntimeError(f"[{key}] training rows differ from the Phase-1 OOF rows - data/cache mismatch.")
        cv = pd.read_csv(BASE_RESULTS / f"cv_folds_{key}.csv")
        thr = thr_all[key]["threshold"]
        yy, ss = o.label.values, o.score.values
        for k, (_, b) in enumerate(splits):
            f1 = f1_score(yy[b], ss[b] >= thr, average="macro")
            if abs(f1 - cv.macro_f1[k]) > 1e-9:
                raise RuntimeError(f"[{key}] fold {k} differs from Phase-1 folds (macro-F1 {f1} vs {cv.macro_f1[k]}).")
    fa = np.empty(len(trs), dtype=int)
    for k, (_, b) in enumerate(splits):
        fa[b] = k
    make_dirs()
    pd.DataFrame({"trip_id": trs.trip_id.values, "fold": fa}).drop_duplicates("trip_id") \
        .sort_values("trip_id").to_csv(RESULTS / "fold_assignment.csv", index=False)


# ----------------------------------------------------------------------------------------------- metrics
def obs_nll(df_fold, score):
    """Mean negative log of the per-observation NORMALISED probability (the prob_obs that predict.py emits)
    assigned to the reference arc, over observations whose reference arc is among the 5 candidates.
    This measures the quality of the quantity the HMM consumes as emission probability."""
    d = df_fold[["trip_id", "observation_id", "label"]].copy()
    d["s"] = np.asarray(score, dtype=float)
    tot = d.groupby(["trip_id", "observation_id"])["s"].transform("sum").clip(lower=1e-12)
    d["p"] = d["s"] / tot
    pos = d[d.label == 1]
    return float(-np.log(np.clip(pos["p"].values, 1e-12, 1.0)).mean())


def cross_validate(key, params, trs, X, y, splits, arcs, n_jobs=-1):
    """Grouped CV exactly as Phase 1 (threshold from pooled OOF; fold metrics at that threshold).
    Returns (fold_metrics_df, pooled_threshold, wall_seconds, oof_scores)."""
    oof = np.zeros(len(y))
    t0 = time.time()
    for a, b in splits:
        m = build_model(key, params, n_jobs)
        m.fit(X[a], y[a])                      # imputer fitted on the training fold only
        oof[b] = m.predict_proba(X[b])[:, 1]
    wall = time.time() - t0
    thr = ev.best_threshold(y, oof)
    rows = []
    for k, (_, b) in enumerate(splits):
        sub = trs.iloc[b]
        cm = ev.cand_metrics(y[b], oof[b], thr)
        t1, _ = ev.top1_accuracy(sub, oof[b])
        cont = ev.route_continuity(ev.top1_table(sub, oof[b]), arcs)
        rows.append(dict(fold=k, macro_f1=cm["macro_f1"],
                         macro_f1_thr050=f1_score(y[b], oof[b] >= 0.5, average="macro"),
                         top1=t1, log_loss=cm["log_loss"], obs_nll=obs_nll(sub, oof[b]), continuity=cont))
    return pd.DataFrame(rows), float(thr), float(wall), oof


def summarize(df):
    out = {}
    for c in df.columns:
        if c == "fold":
            continue
        out[f"{c}_mean"] = float(df[c].mean())
        out[f"{c}_std"] = float(df[c].std())
    return out


def baseline_fold_metrics(key):
    """Phase-1 fold metrics (stored CSV) + obs_nll and Macro-F1@0.5 recomputed from the stored OOF scores."""
    cv = pd.read_csv(BASE_RESULTS / f"cv_folds_{key}.csv")
    o = pd.read_parquet(BASE_RESULTS / f"oof_scores_{key}.parquet")
    thr = json.load(open(BASE_RESULTS / "model_comparison.json"))[key]["threshold"]
    splits = list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(len(o)), o.label.values, o.trip_id.values))
    cv["obs_nll"] = [obs_nll(o.iloc[b], o.score.values[b]) for _, b in splits]
    cv["macro_f1_thr050"] = [f1_score(o.label.values[b], o.score.values[b] >= 0.5, average="macro") for _, b in splits]
    cv.attrs["threshold"] = thr
    return cv


# ----------------------------------------------------------------------------------------------- baseline guard
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_baseline():
    out = {}
    for d in (BASE_ART, BASE_RESULTS):
        for p in sorted(Path(d).rglob("*")):
            if p.is_file():
                out[str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")] = _sha(p)
    return out


def snapshot_baseline():
    """Hash every baseline artifact/result once (never overwritten). Run on a clean checkout of abd3bb9."""
    make_dirs()
    p = RESULTS / "baseline_hashes.json"
    if not p.exists():
        json.dump(_hash_baseline(), open(p, "w"), indent=1)


def verify_baseline_untouched():
    p = RESULTS / "baseline_hashes.json"
    if not p.exists():
        raise RuntimeError("No baseline snapshot found; run run_tuning.py first (on a clean checkout).")
    before, now = json.load(open(p)), _hash_baseline()
    changed = [k for k in before if now.get(k) != before[k]]
    added = [k for k in now if k not in before]
    if changed or added:
        raise RuntimeError(f"Baseline modified! changed={changed} added={added}")
    return True

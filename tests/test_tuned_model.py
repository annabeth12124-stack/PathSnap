"""Phase-2 tests: tuned model + prediction contract. Run from the repo root AFTER run_tuning, refit and report:
    python -m pytest tests/test_tuned_model.py        (or: python tests/test_tuned_model.py)
Also run the existing suite: python tests/test_phase1_integration.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap_ml import predict  # noqa: E402
from pathsnap_ml.config import DATA_ROOT, FEATURES  # noqa: E402
from pathsnap_ml.tuning import common as C  # noqa: E402
from pathsnap_ml.tuning import report  # noqa: E402

CONTRACT_COLS = {"trip_id", "observation_id", "timestamp", "arc_id", "from_node_id", "to_node_id",
                 "score", "prob_obs", "rank", "is_top1"}


def _trip_copy_without_route():
    t = sorted(os.listdir(f"{DATA_ROOT}/student_test"))[0]
    tmp = tempfile.mkdtemp()
    d = f"{tmp}/{t}"
    os.makedirs(d)
    for ext in ["track", "nodes", "arcs"]:                      # NO .route copied
        shutil.copy(f"{DATA_ROOT}/student_test/{t}/{t}.{ext}", d)
    return tmp, d


# ---------------------------------------------------------------------------------- unit tests (no data needed)
def test_evaluator_guard():
    for bad in ["data/evaluator_test/x", "D:/EVALUATOR_testing"]:
        try:
            C.assert_not_evaluator(bad)
        except RuntimeError:
            continue
        raise AssertionError(f"guard did not block {bad}")
    assert "evaluator" not in str(C.CACHE).lower() and "evaluator" not in str(C.DATA_ROOT).lower()


def test_search_spaces_contain_anchor_and_build():
    from optuna.trial import FixedTrial
    for key in C.MODEL_KEYS:
        params = C.suggest(key, FixedTrial(C.ANCHOR[key]))
        assert params == C.ANCHOR[key], (key, params)
        model = C.build_model(key, params, n_jobs=1)
        assert list(model.named_steps)[0] == "simpleimputer"        # preprocessing is inside the per-fold pipeline


def test_selection_is_not_highest_single_score():
    base = {k: dict(macro_f1=0.85, macro_f1_std=0.03, top1=0.77, log_loss=0.22, obs_nll=0.5) for k in C.MODEL_KEYS}
    tuned = {
        "lgbm": dict(macro_f1=0.8560, macro_f1_std=0.060, top1=0.771, log_loss=0.2220, obs_nll=0.60),  # highest F1, unstable/worse probs
        "rf":   dict(macro_f1=0.8540, macro_f1_std=0.030, top1=0.775, log_loss=0.2200, obs_nll=0.50),
        "hgb":  dict(macro_f1=0.8500, macro_f1_std=0.032, top1=0.770, log_loss=0.2250, obs_nll=0.52),
    }
    s = report.select(tuned, base)
    assert s["selected"] == "rf", s


def test_selection_regression_guard():
    base = {k: dict(macro_f1=0.85, macro_f1_std=0.03, top1=0.77, log_loss=0.22, obs_nll=0.5) for k in C.MODEL_KEYS}
    tuned = {k: dict(base[k]) for k in base}
    tuned["lgbm"].update(macro_f1=0.90, top1=0.70)               # big F1 gain but Top-1 collapses -> ineligible
    s = report.select(tuned, base)
    assert s["regressed"]["lgbm"] and s["selected"] != "lgbm"


# ---------------------------------------------------------------------------------- integration tests (need data/cache)
def test_baseline_untouched():
    assert C.verify_baseline_untouched()


def test_folds_identical_to_baseline():
    trs = C.load_training_table()
    _, _, _, splits = C.make_splits(trs)
    C.verify_folds_match_baseline(trs, splits)                    # raises if not identical
    assert len(splits) == 5
    trips = [set(trs.trip_id.values[b]) for _, b in splits]
    assert all(not (trips[i] & trips[j]) for i in range(5) for j in range(i + 1, 5))      # grouped by trip
    for a, b in splits:
        assert not (set(trs.trip_id.values[a]) & set(trs.trip_id.values[b]))              # no trip in train and test


def test_final_tuned_config_and_artifact():
    cfg = json.load(open(C.ARTIFACTS / "final_model_config.json"))
    assert cfg["selected_key"] in C.MODEL_KEYS and list(cfg["features"]) == FEATURES
    assert 0.05 <= cfg["decision_threshold"] <= 0.95 and cfg["baseline_artifacts_modified"] is False
    model, cfg2 = predict.load_model(str(C.ARTIFACTS))
    assert hasattr(model, "predict_proba") and cfg2["selected_key"] == cfg["selected_key"]


def test_tuned_predict_contract():
    model, cfg = predict.load_model(str(C.ARTIFACTS))
    tmp, d = _trip_copy_without_route()
    out = predict.predict_trip(d, model=model, cfg=cfg)
    shutil.rmtree(tmp)
    assert set(out.columns) == CONTRACT_COLS
    g = out.groupby("observation_id")
    assert (g.size() == 5).all()                                                           # 5 candidates per observation
    assert np.allclose(g["prob_obs"].sum(), 1.0)                                           # normalised per observation
    assert out["prob_obs"].between(0, 1).all() and out["score"].between(0, 1).all()
    assert (g["is_top1"].sum() == 1).all()
    assert (g["rank"].apply(lambda r: sorted(r) == [1, 2, 3, 4, 5])).all()
    assert not out[["score", "prob_obs"]].isna().any().any()


def test_tuned_output_schema_matches_baseline():
    tmp, d = _trip_copy_without_route()
    base = predict.predict_trip(d)                                                         # untouched baseline artifacts
    model, cfg = predict.load_model(str(C.ARTIFACTS))
    tuned = predict.predict_trip(d, model=model, cfg=cfg)
    shutil.rmtree(tmp)
    assert list(base.columns) == list(tuned.columns) and (base.dtypes == tuned.dtypes).all()
    key = ["observation_id", "arc_id"]                                                     # candidate generation unchanged
    assert set(map(tuple, base[key].values)) == set(map(tuple, tuned[key].values))
    assert len(base) == len(tuned)


def test_tuned_oof_scores_use_baseline_rows():
    cfg = json.load(open(C.ARTIFACTS / "final_model_config.json"))
    o = pd.read_parquet(C.RESULTS / f"oof_scores_tuned_{cfg['selected_key']}.parquet")
    b = pd.read_parquet(C.BASE_RESULTS / f"oof_scores_{cfg['selected_key']}.parquet")
    assert (o.trip_id.values == b.trip_id.values).all() and (o.arc_id.values == b.arc_id.values).all()


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            n += 1
            print("PASS", k)
    print(f"{n} tests passed")

"""ONE-SHOT report-only scoring of the selected tuned model and its untuned counterpart on student_test.

Run this ONLY after report.py has fixed the selection. It refuses to run a second time (so student-test results cannot
be used to iterate on hyperparameters) unless --force is given. Evaluator data is never read.

Usage: python -m pathsnap_ml.tuning.student_once
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from .. import evaluation as ev
from ..config import CACHE, DATA_ROOT, FEATURES
from . import common as C

OUT = C.RESULTS / "student_test_once.json"


def score(model, te, thr, arcs):
    p = model.predict_proba(te[FEATURES].values.astype(np.float64))[:, 1]
    cm = ev.cand_metrics(te.label.values, p, thr)
    t1, n = ev.top1_accuracy(te, p)
    return dict(macro_f1=cm["macro_f1"], log_loss=cm["log_loss"], accuracy=cm["accuracy"], top1=t1,
                continuity=ev.route_continuity(ev.top1_table(te, p), arcs), threshold=thr, n_obs_ref_ok=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if OUT.exists() and not a.force:
        raise SystemExit(f"{OUT} already exists: student_test was already scored once. Not running again.")
    p = CACHE / "student_test.parquet"
    C.assert_not_evaluator(p)
    C.assert_not_evaluator(DATA_ROOT)
    te = pd.read_parquet(p)
    arcs = ev.load_arcs(DATA_ROOT, "student_test", te.trip_id.unique())
    cfg = json.load(open(C.ARTIFACTS / "final_model_config.json"))
    key = cfg["selected_key"]
    base_thr = json.load(open(C.BASE_RESULTS / "model_comparison.json"))[key]["threshold"]
    out = dict(selected_key=key,
               tuned=score(joblib.load(C.ARTIFACTS / "final_model.joblib"), te, cfg["decision_threshold"], arcs),
               untuned_baseline=score(joblib.load(C.BASE_ART / f"model_{key}.joblib"), te, base_thr, arcs),
               note="report-only; not used for any selection or tuning decision")
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    C.verify_baseline_untouched()


if __name__ == "__main__":
    main()

"""Phase-1 integration tests. Run: python tests/test_phase1_integration.py (after build_tables, run_experiments, export_final)."""
import os, sys, shutil, tempfile, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap_ml.config import *
from pathsnap_ml import predict, evaluation as ev
from pathsnap_ml.candidates import read_trip, generate_candidates

def test_feature_contract():
    assert len(FEATURES) == 7 and not set(FEATURES) & set(ID_COLS) and len(FEATURES) + len(ID_COLS) == 13
def test_candidate_table_shape():
    te = pd.read_parquet(f"{CACHE}/student_test.parquet")
    assert pd.api.types.is_string_dtype(te.trip_id) and te.trip_id.str.len().eq(8).all()       # zero-padded strings
    assert (te.groupby(["trip_id", "observation_id"]).size() == TOP_CANDIDATES).all()
    nan_rows = te[te[FEATURES].isna().any(axis=1)]; assert (nan_rows.observation_id == 0).all() and len(nan_rows) == 5 * te.trip_id.nunique()
def test_predict_never_reads_route_and_contract():
    t = sorted(os.listdir(f"{DATA_ROOT}/student_test"))[0]; tmp = tempfile.mkdtemp(); d = f"{tmp}/{t}"; os.makedirs(d)
    for ext in ["track", "nodes", "arcs"]: shutil.copy(f"{DATA_ROOT}/student_test/{t}/{t}.{ext}", d)   # NO .route copied
    out = predict.predict_trip(d)
    g = out.groupby("observation_id")
    assert np.allclose(g["prob_obs"].sum(), 1.0) and (g["is_top1"].sum() == 1).all() and out["rank"].between(1, 5).all()
    assert set(out.columns) == {"trip_id", "observation_id", "timestamp", "arc_id", "from_node_id", "to_node_id", "score", "prob_obs", "rank", "is_top1"}
    shutil.rmtree(tmp)
def test_predict_matches_experiment_scores():
    import json; cfg = json.load(open("artifacts/final_model_config.json")); key = cfg["selected_key"]
    sc = pd.read_parquet(f"results/test_scores_{key}.parquet"); t = sc.trip_id.iloc[0]
    out = predict.predict_trip(f"{DATA_ROOT}/student_test/{t}")
    m = out.merge(sc[sc.trip_id == t], on=["trip_id", "observation_id", "arc_id"], suffixes=("", "_exp"))
    assert len(m) == len(out) and np.allclose(m.score, m.score_exp, atol=1e-9)
def test_top1_beats_nearest_heuristic_on_student_test():
    import json; cfg = json.load(open("artifacts/final_model_config.json")); key = cfg["selected_key"]
    te = pd.read_parquet(f"{CACHE}/student_test.parquet"); sc = pd.read_parquet(f"results/test_scores_{key}.parquet")
    a, _ = ev.top1_accuracy(te, sc.score.values); b, _ = ev.top1_accuracy(te, -te.distance_m.values); assert a > b
def test_mock_viterbi_consumes_output():
    t = sorted(os.listdir(f"{DATA_ROOT}/student_test"))[0]; out = predict.predict_trip(f"{DATA_ROOT}/student_test/{t}").sort_values(["observation_id", "rank"])
    obs = out.observation_id.unique()[:200]; sub = out[out.observation_id.isin(obs)]
    prev = None; path = []
    for o, g in sub.groupby("observation_id"):
        e = np.log(g.prob_obs.values + 1e-9); a = g.arc_id.values; fr = g.from_node_id.values; to = g.to_node_id.values
        if prev is None: sc = e
        else:
            pa, pt, psc = prev; trans = np.where((pa[:, None] == a[None]) | (pt[:, None] == fr[None]), 0.0, -3.0)
            sc = e + (psc[:, None] + trans).max(0)
        prev = (a, to, sc); path.append(a[int(np.argmax(sc))])
    assert len(path) == len(obs)
if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); n += 1; print("PASS", k)
    print(f"{n} tests passed")

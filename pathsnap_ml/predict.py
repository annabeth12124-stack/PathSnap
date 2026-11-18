"""Inference interface for Divyanshu's HMM/API. Reads ONLY .track/.nodes/.arcs - never .route.
Output (one row per candidate): trip_id, observation_id, timestamp, arc_id, from_node_id, to_node_id,
                                 score, prob_obs (sums to 1 per observation: use as HMM emission), rank (1=best), is_top1."""
import os, json, argparse, numpy as np, pandas as pd, joblib
from .config import FEATURES
from .candidates import read_trip, generate_candidates
HERE = os.path.dirname(os.path.abspath(__file__)); ART = os.path.join(HERE, "..", "artifacts")

def load_model(art_dir=ART):
    cfg = json.load(open(os.path.join(art_dir, "final_model_config.json")))
    return joblib.load(os.path.join(art_dir, "final_model.joblib")), cfg

def predict_trip(trip_dir, model=None, cfg=None):
    if model is None: model, cfg = load_model()
    t = os.path.basename(os.path.normpath(trip_dir))
    track, nodes, arcs = read_trip(trip_dir, t)
    df, *_ = generate_candidates(t, track, nodes, arcs)
    assert list(cfg["features"]) == FEATURES
    df["score"] = model.predict_proba(df[FEATURES].values.astype(np.float64))[:, 1]
    df["prob_obs"] = df["score"] / df.groupby("observation_id")["score"].transform("sum").clip(lower=1e-12)
    df = df.sort_values(["observation_id", "score"], ascending=[True, False], kind="stable").reset_index(drop=True)
    df["rank"] = df.groupby("observation_id").cumcount() + 1; df["is_top1"] = df["rank"] == 1
    return df[["trip_id", "observation_id", "timestamp", "arc_id", "from_node_id", "to_node_id",
           "projection_fraction", "segment_length_m", "score", "prob_obs", "rank", "is_top1"]]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trip_dir", required=True); ap.add_argument("--out", default="pred.csv")
    a = ap.parse_args(); d = predict_trip(a.trip_dir); d.to_csv(a.out, index=False); print(f"{len(d)} rows -> {a.out}")

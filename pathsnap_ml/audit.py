"""Audit of the Phase-1 candidate tables vs Sakshi's reported findings. Writes results/candidate_audit.json"""
import json, numpy as np, pandas as pd
from .config import *
def audit():
    out = {}
    for part in ["train", "student_test"]:
        df = pd.read_parquet(f"{CACHE}/{part}.parquet"); meta = pd.read_csv(f"{CACHE}/{part}_meta.csv", dtype={"trip_id": str})
        g = df.groupby(["trip_id", "observation_id"])
        nearest = g["distance_m"].min()
        obs = g["ref_ok"].first(); lab = g["label"].max()
        ok = obs == 1
        sakshi_cols = ID_COLS + FEATURES
        r = dict(
            candidate_rows=len(df), gps_observations=len(nearest), trips=df.trip_id.nunique(),
            candidates_per_obs_unique=sorted(g.size().unique().tolist()),
            sakshi_columns=len(sakshi_cols), columns_present=all(c in df.columns for c in sakshi_cols),
            nan_rows_per_feature={c: int(df[c].isna().sum()) for c in FEATURES},
            nan_rows_any_feature=int(df[FEATURES].isna().any(axis=1).sum()),
            inf_rows_any_feature=int(np.isinf(df[FEATURES].to_numpy(dtype=float)).any(axis=1).sum()),
            nearest_gt_50m_pct=float((nearest > 50).mean() * 100), nearest_gt_100m_pct=float((nearest > 100).mean() * 100),
            nearest_max_m=float(nearest.max()), nearest_median_m=float(nearest.median()),
            ref_ok_obs=int(ok.sum()), ref_unreliable_obs=int((~ok).sum()), ref_unreliable_pct=float((~ok).mean() * 100),
            OBS_LEVEL_candidate_miss_rate_all_obs_pct=float((1 - lab).mean() * 100),
            OBS_LEVEL_candidate_miss_rate_ref_ok_obs_pct=float((1 - lab[ok]).mean() * 100),
            nearest_candidate_is_ref_top1_ref_ok_pct=None,
            positive_rate_candidates_pct=float(df.label.mean() * 100),
            trip_id_dtype=str(df.trip_id.dtype))
        # nearest-distance heuristic upper bound / baseline
        top = df.sort_values(["trip_id", "observation_id", "distance_m"], kind="stable").drop_duplicates(["trip_id", "observation_id"])
        r["nearest_candidate_is_ref_top1_ref_ok_pct"] = float(top[top.ref_ok == 1].label.mean() * 100)
        out[part] = r
    tr = pd.read_parquet(f"{CACHE}/train.parquet", columns=["trip_id", "observation_id", "timestamp"])
    dt = tr.drop_duplicates(["trip_id", "observation_id"]).groupby("trip_id")["timestamp"].diff().dropna()
    out["train"]["timestamp_dt_nonpositive_steps"] = int((dt <= 0).sum()); out["train"]["timestamp_dt_gt_1s_pct"] = float((dt > 1).mean() * 100)
    json.dump(out, open("results/candidate_audit.json", "w"), indent=1)
    return out
if __name__ == "__main__":
    print(json.dumps(audit(), indent=1))

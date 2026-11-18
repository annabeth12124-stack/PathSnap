"""Build Phase-1 candidate tables for train (80) and student_test (10). Evaluator data is never read.
Output: cache/{part}.parquet (13 Sakshi columns + label, ref_arc, ref_dist, ref_ok), cache/{part}_meta.csv"""
import os, glob, sys, numpy as np, pandas as pd
from .config import *
from .candidates import read_trip, generate_candidates
from .labels import align_reference

def build_partition(part, root=DATA_ROOT, cache=CACHE):
    out, metas = [], []
    for d in sorted(glob.glob(f"{root}/{part}/*")):
        t = os.path.basename(d)
        track, nodes, arcs = read_trip(d, t)
        df, gxy, a0, a1 = generate_candidates(t, track, nodes, arcs)
        route = np.loadtxt(f"{d}/{t}.route", dtype=np.int64, ndmin=1)
        ref, rd = align_reference(gxy, a0, a1, route)
        o = df["observation_id"].to_numpy()
        df["label"] = (df["arc_id"].to_numpy() == ref[o]).astype(np.int8)
        df["ref_arc"] = ref[o]; df["ref_dist"] = rd[o]; df["ref_ok"] = (rd[o] <= REF_OK_M).astype(np.int8)
        has = df.groupby("observation_id")["label"].max().reindex(range(len(track))).values
        ok = rd <= REF_OK_M
        metas.append(dict(trip_id=t, n_obs=len(track), n_arcs=len(arcs), n_route=len(route), frac_ref_unreliable=1 - ok.mean(),
                          miss_rate_all=1 - has.mean(), miss_rate_ref_ok=(1 - has[ok].mean()) if ok.any() else np.nan))
        out.append(df); print(part, t, len(track), flush=True)
    df = pd.concat(out, ignore_index=True)
    os.makedirs(cache, exist_ok=True)
    df.to_parquet(f"{cache}/{part}.parquet"); pd.DataFrame(metas).to_csv(f"{cache}/{part}_meta.csv", index=False)
    return df

if __name__ == "__main__":
    for part in ["train", "student_test"]:
        build_partition(part)

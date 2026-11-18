"""Metrics for candidate classification + observation-level map-matching quality (Phase 1)."""
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix

def best_threshold(y, p):
    ts = np.linspace(0.05, 0.95, 91)
    return float(ts[int(np.argmax([f1_score(y, p >= t, average="macro") for t in ts]))])

def cand_metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, yh), precision_macro=precision_score(y, yh, average="macro", zero_division=0),
                recall_macro=recall_score(y, yh, average="macro", zero_division=0), macro_f1=f1_score(y, yh, average="macro"),
                precision_pos=precision_score(y, yh, zero_division=0), recall_pos=recall_score(y, yh, zero_division=0),
                f1_pos=f1_score(y, yh, zero_division=0), log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]), threshold=thr,
                confusion=confusion_matrix(y, yh, labels=[0, 1]).tolist())

def top1_table(df, score):
    d = df[["trip_id", "observation_id", "arc_id", "label", "ref_ok"]].copy(); d["score"] = np.asarray(score)
    d = d.sort_values(["trip_id", "observation_id", "score"], ascending=[True, True, False], kind="stable")
    return d.drop_duplicates(["trip_id", "observation_id"])

def top1_accuracy(df, score):
    """Observation-level Top-1 over reference-reliable observations. A miss (reference arc not among the 5 candidates) counts as an error."""
    top = top1_table(df, score); top = top[top.ref_ok == 1]
    return float(top.label.mean()), len(top)

def candidate_miss_rate(df):
    """Observation-level candidate miss rate over reference-reliable observations (NOT Sakshi's trip-level arc-set coverage)."""
    g = df[df.ref_ok == 1].groupby(["trip_id", "observation_id"])["label"].max()
    return float(1 - g.mean())

def route_continuity(top, arcs_by_trip):
    """Fraction of consecutive Top-1 predictions that are the same arc or head-to-tail adjacent."""
    ok = tot = 0
    for t, g in top.groupby("trip_id"):
        a = arcs_by_trip[t]; p = g.sort_values("observation_id").arc_id.values
        if len(p) < 2: continue
        ok += int(((p[1:] == p[:-1]) | (a[p[:-1], 1] == a[p[1:], 0])).sum()); tot += len(p) - 1
    return ok / max(tot, 1)

def load_arcs(root, part, trips):
    return {t: np.loadtxt(f"{root}/{part}/{t}/{t}.arcs", dtype=np.int64, ndmin=2) for t in trips}

"""Proves pathsnap_ml.candidates == Sakshi's notebook `process_trip` on real trips by executing the notebook's own code."""
import json, sys, glob, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from pathsnap_ml.config import DATA_ROOT
from pathsnap_ml.candidates import read_trip, generate_candidates
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "preprocessing" / "candidate_generation.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
src = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}
ns = {"TRAIN_DIR": Path(DATA_ROOT) / "train", "EARTH_RADIUS": 6371000}
exec(src[0].split("# Project paths")[0], ns)
exec(src[21], ns)
cols = ["trip_id", "observation_id", "timestamp", "arc_id", "from_node_id", "to_node_id", "distance_m", "projection_fraction",
        "segment_length_m", "arc_heading", "gps_heading", "gps_speed_mps", "heading_difference"]
sizes = sorted((sum(1 for _ in open(f)), os.path.basename(os.path.dirname(f))) for f in glob.glob(f"{DATA_ROOT}/train/*/*.track"))
for n, t in sizes[:3] + sizes[40:41]:
    ref = ns["process_trip"](t)
    track, nodes, arcs = read_trip(f"{DATA_ROOT}/train/{t}", t)
    new, *_ = generate_candidates(t, track, nodes, arcs)
    assert len(ref) == len(new) == 5 * n
    # 1) per-observation sorted distances identical (float noise only)
    a = ref.sort_values(["observation_id", "distance_m"], kind="stable").reset_index(drop=True)
    b = new.sort_values(["observation_id", "distance_m"], kind="stable").reset_index(drop=True)
    assert np.allclose(a.distance_m, b.distance_m, atol=1e-9), f"{t}: distance vectors differ"
    # 2) arc sets identical, except exact distance ties at the 5th-rank cut-off (notebook uses unstable sort -> arbitrary pick)
    ra = a.groupby("observation_id")["arc_id"].apply(lambda s: set(s)); rb = b.groupby("observation_id")["arc_id"].apply(lambda s: set(s))
    diff_obs = ra.index[[x != y for x, y in zip(ra, rb)]]
    for o in diff_obs:
        da = a[a.observation_id == o]; db = b[b.observation_id == o]
        only_a = da[~da.arc_id.isin(db.arc_id)]; only_b = db[~db.arc_id.isin(da.arc_id)]
        assert np.allclose(sorted(only_a.distance_m), sorted(only_b.distance_m), atol=1e-9), f"{t} obs {o}: non-tie arc difference"
    # 3) all per-row features identical for arcs present in both
    m = a.merge(b, on=["observation_id", "arc_id"], suffixes=("_nb", "_new"))
    for c in cols[6:]:
        assert np.allclose(m[c + "_nb"].astype(float), m[c + "_new"].astype(float), equal_nan=True, atol=1e-9), f"{t}: {c} differs"
    print(f"OK {t}: {n} obs, {len(new)} rows; distance vectors identical; {len(diff_obs)} obs differ only by exact rank-5 ties; features identical on shared rows")

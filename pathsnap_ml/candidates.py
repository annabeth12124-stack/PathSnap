"""Sakshi's candidate generation, reproduced exactly (vectorised).
Source of truth: preprocessing/candidate_generation.ipynb, cell `process_trip` (commit 441f431).
  projection : local equirectangular, centre = mean lat/lon of the GPS TRACK, R = 6371000
  index      : cKDTree over arc MIDPOINTS; query K=100 nearest midpoints per GPS fix
  distance   : exact Euclidean point-to-segment in projected metres
  retained   : 5 closest segments, no distance cutoff
Vectorisation changes speed only; tests/test_candidates_match_notebook.py proves equality with the notebook code.
"""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from .config import EARTH_RADIUS, K_NEARBY, TOP_CANDIDATES

def read_trip(trip_dir, trip_id):
    p = f"{trip_dir}/{trip_id}"
    track = pd.read_csv(p + ".track", sep=r"\s+", header=None, names=["longitude", "latitude", "timestamp"])
    nodes = pd.read_csv(p + ".nodes", sep=r"\s+", header=None, names=["longitude", "latitude"])
    arcs = pd.read_csv(p + ".arcs", sep=r"\s+", header=None, names=["from_node_id", "to_node_id"])
    return track, nodes, arcs

def make_projector(track):
    ref_lat = track["latitude"].mean(); ref_lon = track["longitude"].mean()
    def project(lon, lat):
        x = EARTH_RADIUS * np.radians(np.asarray(lon) - ref_lon) * np.cos(np.radians(ref_lat))
        y = EARTH_RADIUS * np.radians(np.asarray(lat) - ref_lat)
        return np.column_stack((x, y))
    return project

def trip_geometry(track, nodes, arcs):
    project = make_projector(track)
    node_xy = project(nodes["longitude"], nodes["latitude"]); gps_xy = project(track["longitude"], track["latitude"])
    from_ids = arcs["from_node_id"].to_numpy(); to_ids = arcs["to_node_id"].to_numpy()
    return gps_xy, node_xy[from_ids], node_xy[to_ids], from_ids, to_ids

def generate_candidates(trip_id, track, nodes, arcs):
    track = track.copy()
    gps_xy, arc_start, arc_end, from_ids, to_ids = trip_geometry(track, nodes, arcs)
    n_obs = len(track); n_arcs = len(arcs)
    tree = cKDTree((arc_start + arc_end) / 2)
    k = K_NEARBY
    if n_arcs < k:   # notebook would crash on such a trip; none exist in train/student_test (asserted in audit)
        raise ValueError(f"trip {trip_id}: only {n_arcs} arcs < K={k}")
    _, nearby = tree.query(gps_xy, k=k)                       # (N,100)
    S = arc_start[nearby]; E = arc_end[nearby]; seg = E - S    # (N,100,2)
    L2 = (seg ** 2).sum(-1)
    P = gps_xy[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(L2 == 0, 0.0, np.clip(((P - S) * seg).sum(-1) / np.where(L2 == 0, 1, L2), 0, 1))
    closest = S + frac[..., None] * seg
    dist = np.linalg.norm(P - closest, axis=-1)
    order = np.argsort(dist, axis=1, kind="stable")[:, :TOP_CANDIDATES]
    take = lambda a: np.take_along_axis(a, order, axis=1).ravel()
    arc_id = take(nearby); obs = np.repeat(np.arange(n_obs), TOP_CANDIDATES)
    df = pd.DataFrame({
        "trip_id": trip_id, "observation_id": obs,
        "timestamp": track["timestamp"].to_numpy()[obs],
        "arc_id": arc_id.astype(int), "from_node_id": from_ids[arc_id].astype(int), "to_node_id": to_ids[arc_id].astype(int),
        "distance_m": take(dist), "projection_fraction": take(frac), "segment_length_m": np.sqrt(take(L2))})
    gdx = np.diff(gps_xy[:, 0], prepend=gps_xy[0, 0]); gdy = np.diff(gps_xy[:, 1], prepend=gps_xy[0, 1])
    dt = track["timestamp"].diff().fillna(1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        gps_speed = np.sqrt(gdx ** 2 + gdy ** 2) / dt
    gps_heading = (np.degrees(np.arctan2(gdx, gdy)) + 360) % 360
    arc_heading = (np.degrees(np.arctan2(arc_end[:, 0] - arc_start[:, 0], arc_end[:, 1] - arc_start[:, 1])) + 360) % 360
    df["arc_heading"] = arc_heading[df["arc_id"].to_numpy()]
    df["gps_heading"] = gps_heading[obs]; df["gps_speed_mps"] = gps_speed[obs]
    diff = df["gps_heading"] - df["arc_heading"]
    df["heading_difference"] = (((diff + 180) % 360) - 180).abs()
    first = df["observation_id"] == 0
    df.loc[first, ["gps_speed_mps", "gps_heading", "heading_difference"]] = np.nan
    return df, gps_xy, arc_start, arc_end

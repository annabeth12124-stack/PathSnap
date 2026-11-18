"""FastAPI service for classifier and HMM map matching."""
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathsnap_ml.predict import predict_trip
from pathsnap_ml.candidates import read_trip, make_projector
from route_reconstruction.transition_model import TransitionModel
from route_reconstruction.viterbi import viterbi

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" 
FRONTEND_DIR = ROOT / "frontend"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API", description="GPS map matching and route reconstruction", version="1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "healthy", "service": "PathSnap API"}


@app.get("/trips")
def list_trips(split: str = Query("student_test", pattern="^(train|student_test)$")):
    folder = DATA_DIR / split
    if not folder.is_dir():
        raise HTTPException(404, f"Dataset split not found: {split}")
    return {"split": split, "trip_ids": sorted(p.name for p in folder.iterdir() if p.is_dir())}


@app.post("/predict/{trip_id}")
def predict_route(trip_id: str, split: str = Query("student_test", pattern="^(train|student_test)$")):
    # Restrict IDs to one path component to prevent path traversal.
    if not trip_id or Path(trip_id).name != trip_id or trip_id in {".", ".."}:
        raise HTTPException(400, "Invalid trip ID")
    trip_dir = DATA_DIR / split / trip_id
    if not trip_dir.is_dir():
        raise HTTPException(404, "Trip not found")
    try:
        track, nodes, arcs = read_trip(str(trip_dir), trip_id)
        predictions = predict_trip(str(trip_dir))
        project = make_projector(track)
        gps_xy = project(track["longitude"], track["latitude"])
        distances = {i: float(np.linalg.norm(gps_xy[i] - gps_xy[i-1])) for i in range(1, len(gps_xy))}
        transition_model = TransitionModel(nodes, arcs)
        route = viterbi(predictions, transition_model, distances)
        # Classifier-only baseline: highest emission probability at each fix.
        baseline = (predictions.sort_values(["observation_id", "prob_obs"], ascending=[True, False], kind="stable")
                    .drop_duplicates("observation_id"))
        route_df = pd.DataFrame(route)
        valid = sum(bool(row["transition_valid"]) for row in route[1:])
        invalid = max(0, len(route) - 1 - valid)
        arc_ids = arcs.index.to_numpy()
        coords = nodes[["longitude", "latitude"]].to_numpy()
        arc_coords = {int(i): [[float(coords[int(r.from_node_id)][0]), float(coords[int(r.from_node_id)][1])],
                                [float(coords[int(r.to_node_id)][0]), float(coords[int(r.to_node_id)][1])]]
                      for i, r in arcs.iterrows()}
        log.info("trip=%s split=%s observations=%d transitions_valid=%d transitions_broken=%d", trip_id, split, len(route), valid, invalid)
        return {
            "trip_id": trip_id, "split": split, "observations": len(route),
            "valid_transitions": valid, "invalid_transitions": invalid,
            "classifier_only": [{"observation_id": int(r.observation_id), "arc_id": int(r.arc_id), "prob_obs": float(r.prob_obs)} for r in baseline.itertuples()],
            "predicted_route": route,
            "gps_track": [{"longitude": float(r.longitude), "latitude": float(r.latitude)} for r in track.itertuples()],
            "arc_coordinates": arc_coords,
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Route prediction failed for trip %s", trip_id)
        raise HTTPException(500, f"Route prediction failed: {exc}") from exc


if FRONTEND_DIR.is_dir() and (FRONTEND_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

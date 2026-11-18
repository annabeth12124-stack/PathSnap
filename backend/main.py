"""FastAPI service for classifier and HMM map matching."""
from pathlib import Path
import json
import logging
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathsnap_ml.predict import predict_trip
from pathsnap_ml.candidates import read_trip, make_projector
from pathsnap_ml.labels import align_reference
from pathsnap_ml.config import REF_OK_M
from route_reconstruction.transition_model import TransitionModel
from route_reconstruction.viterbi import viterbi

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FRONTEND_DIR = ROOT / "frontend"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API", description="GPS map matching and route reconstruction", version="1.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    config_path = ROOT / "artifacts" / "final_model_config.json"
    model_name = "Classifier"
    try:
        model_name = json.loads(config_path.read_text(encoding="utf-8")).get("selected_model", model_name)
    except (OSError, ValueError):
        pass
    return {"status": "healthy", "service": "PathSnap API", "model": model_name}


@app.get("/trips")
def list_trips(split: str = Query("student_test", pattern="^(train|student_test|evaluator_test)$")):
    folder = DATA_DIR / split
    if not folder.is_dir():
        raise HTTPException(404, f"Dataset split not found: {split}")
    return {"split": split, "trip_ids": sorted(p.name for p in folder.iterdir() if p.is_dir())}


@app.post("/predict/{trip_id}")
def predict_route(trip_id: str, split: str = Query("student_test", pattern="^(train|student_test|evaluator_test)$")):
    # Restrict IDs to one path component to prevent path traversal.
    if not trip_id or Path(trip_id).name != trip_id or trip_id in {".", ".."}:
        raise HTTPException(400, "Invalid trip ID")
    trip_dir = DATA_DIR / split / trip_id
    if not trip_dir.is_dir():
        raise HTTPException(404, "Trip not found")
    started = time.perf_counter()
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
        valid = sum(bool(row["transition_valid"]) for row in route[1:])
        invalid = max(0, len(route) - 1 - valid)

        # Read reference labels only on the server. The browser receives the
        # aggregate scores below, never the reference route or aligned labels.
        route_file = trip_dir / f"{trip_id}.route"
        metrics = {
            "available": False,
            "hmm_top1_accuracy": None,
            "classifier_top1_accuracy": None,
            "route_f1": None,
            "route_precision": None,
            "route_recall": None,
            "reference_observations": 0,
            "reference_reliability_m": REF_OK_M,
        }
        if route_file.is_file() and route_file.stat().st_size:
            reference_route = np.loadtxt(route_file, dtype=np.int64, ndmin=1)
            node_xy = project(nodes["longitude"], nodes["latitude"])
            arc_start = node_xy[arcs["from_node_id"].to_numpy(dtype=int)]
            arc_end = node_xy[arcs["to_node_id"].to_numpy(dtype=int)]
            reference_arcs, reference_distance = align_reference(
                gps_xy, arc_start, arc_end, reference_route
            )
            reliable = reference_distance <= REF_OK_M
            reliable_count = int(reliable.sum())
            predicted_arcs = np.asarray([row["arc_id"] for row in route], dtype=int)
            classifier_arcs = (
                baseline.sort_values("observation_id", kind="stable")["arc_id"]
                .to_numpy(dtype=int)
            )

            if reliable_count:
                metrics["available"] = True
                metrics["reference_observations"] = reliable_count
                metrics["hmm_top1_accuracy"] = float(
                    np.mean(predicted_arcs[reliable] == reference_arcs[reliable])
                )
                metrics["classifier_top1_accuracy"] = float(
                    np.mean(classifier_arcs[reliable] == reference_arcs[reliable])
                )

                reference_set = set(map(int, reference_arcs[reliable]))
                predicted_set = set(map(int, predicted_arcs[reliable]))
                overlap = len(reference_set & predicted_set)
                precision = overlap / len(predicted_set) if predicted_set else 0.0
                recall = overlap / len(reference_set) if reference_set else 0.0
                metrics["route_precision"] = precision
                metrics["route_recall"] = recall
                metrics["route_f1"] = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                )

        coords = nodes[["longitude", "latitude"]].to_numpy()
        arc_coords = {int(i): [[float(coords[int(r.from_node_id)][0]), float(coords[int(r.from_node_id)][1])],
                                [float(coords[int(r.to_node_id)][0]), float(coords[int(r.to_node_id)][1])]]
                      for i, r in arcs.iterrows()}
        log.info("trip=%s split=%s observations=%d transitions_valid=%d transitions_broken=%d", trip_id, split, len(route), valid, invalid)
        processing_time = time.perf_counter() - started
        return {
            "trip_id": trip_id, "split": split, "observations": len(route),
            "valid_transitions": valid, "invalid_transitions": invalid,
            "route_continuity": valid / max(1, len(route) - 1),
            "processing_time_seconds": processing_time,
            "metrics": metrics,
            "classifier_only": [{
                "observation_id": int(r.observation_id),
                "arc_id": int(r.arc_id),
                "from_node_id": int(r.from_node_id),
                "to_node_id": int(r.to_node_id),
                "projection_fraction": float(r.projection_fraction),
                "prob_obs": float(r.prob_obs),
            } for r in baseline.itertuples()],
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

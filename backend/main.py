
from pathlib import Path
from functools import lru_cache
import math
import time

import pandas as pd
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from route_reconstruction.graph_builder import build_road_graph
from hmm.route_predictor import predict_route


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_CSV = (
    PROJECT_ROOT / "data" / "processed"
    / "all_training_candidate_features.csv"
)
TRAIN_FOLDER = PROJECT_ROOT / "Student folder" / "train"

app = FastAPI(title="PathSnap API", version="1.0.0")

# Allow the local React development server to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    trip_id: str
    method: str = "classifier_hmm"


@lru_cache(maxsize=4)
def load_candidates():
    if not CANDIDATE_CSV.is_file():
        raise FileNotFoundError(
            f"Candidate CSV not found: {CANDIDATE_CSV}"
        )

    df = pd.read_csv(CANDIDATE_CSV)

    required = {
        "trip_id", "observation_id", "arc_id", "distance_m"
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Candidate CSV is missing columns: {sorted(missing)}"
        )

    df["trip_id"] = df["trip_id"].astype(str).str.zfill(8)
    return df


@lru_cache(maxsize=4)
def load_graph(trip_id):
    trip_folder = TRAIN_FOLDER / trip_id

    nodes_file = trip_folder / f"{trip_id}.nodes"
    arcs_file = trip_folder / f"{trip_id}.arcs"

    if not nodes_file.is_file() or not arcs_file.is_file():
        raise FileNotFoundError(
            f"Road network files not found for trip {trip_id}"
        )

    return build_road_graph(nodes_file, arcs_file)


def read_gps_observations(trip_id):
    track_file = TRAIN_FOLDER / trip_id / f"{trip_id}.track"

    if not track_file.is_file():
        raise FileNotFoundError(
            f"GPS track file not found: {track_file}"
        )

    track = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"],
    )

    if track.empty or track[["longitude", "latitude"]].isna().any().any():
        raise ValueError("GPS track is empty or contains invalid coordinates")

    return [
        {
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
        }
        for row in track.itertuples()
    ]


def build_route_coordinates(graph, arc_ids):
    coordinates = []

    for arc_id in arc_ids:
        matching_edges = [
            (source, target)
            for source, target, key, data in graph.edges(
                keys=True, data=True
            )
            if int(data.get("arc_id", key)) == int(arc_id)
        ]

        if not matching_edges:
            raise ValueError(f"Arc {arc_id} was not found in the graph")

        source, target = matching_edges[0]

        for node_id in (source, target):
            node = graph.nodes[node_id]
            point = {
                "latitude": float(node["latitude"]),
                "longitude": float(node["longitude"]),
            }

            if not coordinates or coordinates[-1] != point:
                coordinates.append(point)

    return coordinates


def calculate_route_length_km(coordinates):
    total = 0.0

    for first, second in zip(coordinates, coordinates[1:]):
        lat1 = math.radians(first["latitude"])
        lon1 = math.radians(first["longitude"])
        lat2 = math.radians(second["latitude"])
        lon2 = math.radians(second["longitude"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2)
            * math.sin(dlon / 2) ** 2
        )

        total += 6371.0 * 2 * math.atan2(
            math.sqrt(a), math.sqrt(1 - a)
        )

    return total


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "PathSnap API",
        "candidate_csv_exists": CANDIDATE_CSV.is_file(),
        "training_folder_exists": TRAIN_FOLDER.is_dir(),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    trip_id = str(request.trip_id).strip().zfill(8)

    if not trip_id.isdigit():
        raise HTTPException(status_code=400, detail="Invalid Trip ID")

    if request.method == "classifier":
        raise HTTPException(
            status_code=501,
            detail=(
                "Classifier-only prediction is not connected yet. "
                "The trained classifier interface is still required."
            ),
        )

    if request.method != "classifier_hmm":
        raise HTTPException(
            status_code=400,
            detail="Unknown prediction method",
        )

    started = time.perf_counter()

    try:
        df = load_candidates()
        trip_df = df[df["trip_id"] == trip_id].copy()

        if trip_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No candidate data found for trip {trip_id}",
            )

        observations = []

        for _, group in trip_df.groupby("observation_id", sort=True):
            candidates = [
                (int(row.arc_id), float(row.distance_m))
                for row in group.itertuples()
            ]
            observations.append(candidates)

        graph = load_graph(trip_id)
        gps_points = read_gps_observations(trip_id)

        if len(gps_points) != len(observations):
            raise ValueError(
                "GPS observation count does not match candidate data"
            )

        predicted_arcs = predict_route(graph, observations)

        route_coordinates = build_route_coordinates(
            graph, predicted_arcs
        )

        elapsed_ms = (time.perf_counter() - started) * 1000

        return {
            "trip_id": trip_id,
            "method": "classifier_hmm_distance_emission",
            "num_observations": len(gps_points),
            "num_arcs": len(predicted_arcs),
            "route_length_km": round(
                calculate_route_length_km(route_coordinates), 3
            ),
            "processing_time_ms": round(elapsed_ms, 2),
            "observations": gps_points,
            "route": route_coordinates,
            "predicted_arc_ids": predicted_arcs,
            "candidates": [],
        }

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, nx.NetworkXException) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Route reconstruction failed: {exc}",
        )


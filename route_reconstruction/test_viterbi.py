
from pathlib import Path
import numpy as np
import pandas as pd

from pathsnap_ml.predict import predict_trip
from pathsnap_ml.candidates import read_trip, make_projector
from route_reconstruction.transition_model import TransitionModel
from route_reconstruction.viterbi import viterbi


# Project paths
ROOT = Path(__file__).resolve().parent.parent
TRIP_ID = "00000001"
TRIP_DIR = ROOT / "data" / "raw" / "train" / TRIP_ID

# Load original trip data
track, nodes, arcs = read_trip(
    str(TRIP_DIR),
    TRIP_ID
)

# Generate classifier predictions
print("Generating candidate predictions...")
predictions = predict_trip(str(TRIP_DIR))

print("Prediction rows:", len(predictions))

# Limit initial test to 100 observations
N_OBSERVATIONS = 100

predictions = predictions[
    predictions["observation_id"] < N_OBSERVATIONS
].copy()

track = track.iloc[:N_OBSERVATIONS].copy()

# Calculate GPS displacement using the same projection
project = make_projector(track)
gps_xy = project(track["longitude"], track["latitude"])

gps_distances = {
    i: float(np.linalg.norm(gps_xy[i] - gps_xy[i - 1]))
    for i in range(1, len(gps_xy))
}

# Build transition model
print("Building transition model...")
transition_model = TransitionModel(nodes, arcs)

# Run Viterbi
print("Running Viterbi...")
route = viterbi(
    predictions,
    transition_model,
    gps_distances
)

# Save output
result = pd.DataFrame(route)

# Validate consecutive predicted transitions
valid_transitions = 0
invalid_transitions = 0
transition_distances = []

for i in range(1, len(result)):
    previous = result.iloc[i - 1]
    current = result.iloc[i]

    distance = transition_model.network_distance(
        previous, current
    )

    if np.isfinite(distance):
        valid_transitions += 1
        transition_distances.append(distance)
    else:
        invalid_transitions += 1

print("\nROUTE VALIDATION")
print("Valid transitions:", valid_transitions)
print("Invalid transitions:", invalid_transitions)

if transition_distances:
    print(
        "Mean network transition distance:",
        round(np.mean(transition_distances), 2), "m"
    )

print(
    "Directly connected transitions:",
    sum(
        transition_model.graph.has_edge(
            int(result.iloc[i - 1]["to_node_id"]),
            int(result.iloc[i]["from_node_id"])
        )
        for i in range(1, len(result))
    )
)


output_path = ROOT / "results" / "viterbi_trip_00000001.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

result.to_csv(output_path, index=False)

print("\nViterbi completed!")
print("Observations reconstructed:", len(result))
print("Unique predicted arcs:", result["arc_id"].nunique())
print("Output:", output_path)

print("\nFirst 10 predictions:")
print(result.head(10).to_string(index=False))

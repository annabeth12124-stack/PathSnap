
"""PathSnap Phase 1 configuration.

Candidate-generation constants mirror Sakshi's notebook (commit 441f431).
"""
import os
from pathlib import Path

# Project root: parent of the pathsnap_ml package
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset root contains train/ and student_test/
DATA_ROOT = Path(
    os.environ.get("PATHSNAP_DATA", str(PROJECT_ROOT / "data"))
)

# Generated candidate tables and metadata
CACHE = Path(
    os.environ.get("PATHSNAP_CACHE", str(PROJECT_ROOT / "cache"))
)

EARTH_RADIUS = 6371000
K_NEARBY = 100
TOP_CANDIDATES = 5

FEATURES = [
    "distance_m",
    "projection_fraction",
    "segment_length_m",
    "gps_speed_mps",
    "gps_heading",
    "arc_heading",
    "heading_difference",
]

ID_COLS = [
    "trip_id",
    "observation_id",
    "arc_id",
    "from_node_id",
    "to_node_id",
    "timestamp",
]

SEED = 42
STRIDE = 4
REF_OK_M = 75.0

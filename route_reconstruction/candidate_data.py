
from pathlib import Path
import pandas as pd


def load_candidate_data():
    project_root = Path(__file__).resolve().parent.parent

    csv_path = (
        project_root
        / "data"
        / "processed"
        / "all_training_candidate_features.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = [
        "trip_id",
        "observation_id",
        "arc_id",
        "distance_m",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df.sort_values(
        ["trip_id", "observation_id", "distance_m"]
    ).reset_index(drop=True)


import math


def calculate_emissions(candidates, sigma=20.0):
    """
    Calculate the emission probability for each candidate road arc.

    candidates: list of (arc_id, distance_in_metres)
    sigma: GPS error scale in metres
    """

    if sigma <= 0:
        raise ValueError("sigma must be greater than zero")

    if not candidates:
        return {}

    log_scores = []

    for arc_id, distance in candidates:
        distance = float(distance)

        if not math.isfinite(distance) or distance < 0:
            raise ValueError("Distances must be finite and non-negative")

        log_score = -0.5 * (distance / sigma) ** 2
        log_scores.append((arc_id, log_score))

    # Normalize safely so probabilities add up to 1
    max_score = max(score for _, score in log_scores)

    weights = [
        (arc_id, math.exp(score - max_score))
        for arc_id, score in log_scores
    ]

    total = sum(weight for _, weight in weights)

    return {
        arc_id: weight / total
        for arc_id, weight in weights
    }

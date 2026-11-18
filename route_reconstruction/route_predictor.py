from route_reconstruction.emission import calculate_emissions
from route_reconstruction.transition import build_transition_probabilities
from route_reconstruction.viterbi import viterbi


def predict_route(graph, observations, sigma=20.0):
    """
    Reconstruct a route from GPS observations.

    observations format:
    [
        [(arc_id, distance_in_metres), ...],
        [(arc_id, distance_in_metres), ...],
        ...
    ]

    Returns the most likely sequence of arc IDs.
    """

    if not observations:
        raise ValueError("At least one observation is required")

    # Step 1: Calculate emission probabilities.
    emissions = {
        index: calculate_emissions(candidates, sigma)
        for index, candidates in enumerate(observations)
    }

    if any(not scores for scores in emissions.values()):
        raise ValueError("An observation has no valid candidates")

    # Step 2: Give equal starting probability to first candidates.
    first_candidates = list(emissions[0])

    initial_probability = 1.0 / len(first_candidates)

    initial_probabilities = {
        arc_id: initial_probability
        for arc_id in first_candidates
    }

    # Step 3: Calculate transitions for each consecutive pair.
    transitions_by_step = []

    for index in range(1, len(observations)):
        transitions = build_transition_probabilities(
            graph,
            list(emissions[index - 1]),
            list(emissions[index])
        )

        transitions_by_step.append(transitions)

    # Step 4: Run Viterbi using the transition probabilities.
    return viterbi(
        list(emissions.keys()),
        initial_probabilities,
        emissions,
        transitions_by_step
    )

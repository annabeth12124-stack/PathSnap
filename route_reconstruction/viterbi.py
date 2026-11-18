
import math


def viterbi(observations, initial_probabilities, emissions, transitions):
    """Find the most likely sequence of reachable road arcs."""

    if not observations:
        return []

    scores = {}
    backpointers = []

    # First observation
    first = observations[0]

    for arc_id, emission_prob in emissions[first].items():
        initial_prob = initial_probabilities.get(arc_id, 0)

        if initial_prob > 0 and emission_prob > 0:
            scores[arc_id] = (
                math.log(initial_prob) + math.log(emission_prob)
            )

    if not scores:
        raise ValueError("No valid starting candidates")

    # Process subsequent observations
    for step, observation in enumerate(observations[1:]):

        if isinstance(transitions, dict):
            step_transitions = transitions
        else:
            step_transitions = transitions[step]

        new_scores = {}
        previous_arcs = {}

        for current_arc, emission_prob in emissions[observation].items():

            if emission_prob <= 0:
                continue

            best_score = float("-inf")
            best_previous = None

            for previous_arc, previous_score in scores.items():

                transition_prob = step_transitions.get(
                    (previous_arc, current_arc), 0
                )

                if transition_prob <= 0:
                    continue

                score = (
                    previous_score
                    + math.log(transition_prob)
                    + math.log(emission_prob)
                )

                if score > best_score:
                    best_score = score
                    best_previous = previous_arc

            if best_previous is not None:
                new_scores[current_arc] = best_score
                previous_arcs[current_arc] = best_previous

        # Report a genuine gap instead of silently inventing a route.
        if not new_scores:
            raise ValueError(
                f"No valid transition at observation {observation}. "
                "The candidate sets are disconnected."
            )

        scores = new_scores
        backpointers.append(previous_arcs)

    # Select the best final reachable arc.
    final_arc = max(scores, key=scores.get)

    route = [final_arc]

    for previous_arcs in reversed(backpointers):
        final_arc = previous_arcs[final_arc]
        route.append(final_arc)

    route.reverse()

    return route

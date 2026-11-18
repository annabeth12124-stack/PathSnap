
import numpy as np


def viterbi(candidate_df, transition_model, gps_distances):
    """
    Find the most likely sequence of candidate road arcs.

    candidate_df:
        Prediction DataFrame from predict_trip()

    gps_distances:
        Dictionary mapping observation_id to distance
        travelled since the previous observation.
    """

    if candidate_df.empty:
        return []

    observations = sorted(
        candidate_df["observation_id"].unique()
    )

    states = {
        obs: group.reset_index(drop=True)
        for obs, group in candidate_df.groupby("observation_id")
    }

    # Initial probabilities
    first = states[observations[0]]

    scores = np.log(
        first["prob_obs"].to_numpy(dtype=float) + 1e-12
    )

    backpointers = []
    breaks = [False]

    # Process subsequent observations
    for obs_index in range(1, len(observations)):

        previous_obs = observations[obs_index - 1]
        current_obs = observations[obs_index]

        previous_states = states[previous_obs]
        current_states = states[current_obs]

        gps_distance = float(gps_distances.get(current_obs, 0.0))

        current_emissions = np.log(
            current_states["prob_obs"].to_numpy(dtype=float)
            + 1e-12
        )

        next_scores = np.full(len(current_states), -np.inf)
        pointers = np.zeros(len(current_states), dtype=int)

        for j, current in current_states.iterrows():

            best_score = -np.inf
            best_previous = 0

            for i, previous in previous_states.iterrows():

                probability = transition_model.probability(
                    previous,
                    current,
                    gps_distance
                )

                if probability <= 0 or not np.isfinite(probability):
                    continue

                candidate_score = scores[i] + np.log(probability)

                if candidate_score > best_score:
                    best_score = candidate_score
                    best_previous = i

            if np.isfinite(best_score):
                next_scores[j] = best_score + current_emissions[j]

            pointers[j] = best_previous

        disconnected = not np.isfinite(next_scores).any()
        if disconnected:
            # Retain this observation and explicitly start a new route segment.
            next_scores = current_emissions.copy()
            pointers[:] = 0

        scores = next_scores
        backpointers.append(pointers)
        breaks.append(disconnected)

    # Backtrack
    state_index = int(np.argmax(scores))
    selected = [state_index]

    for pointers in reversed(backpointers):
        state_index = int(pointers[state_index])
        selected.append(state_index)

    selected.reverse()

    result = []

    segment_id = 0
    for index, (obs, state_index) in enumerate(zip(observations, selected)):
        if index and breaks[index]:
            segment_id += 1
        row = states[obs].iloc[state_index]

        result.append({
            "observation_id": int(obs),
            "arc_id": int(row["arc_id"]),
            "from_node_id": int(row["from_node_id"]),
            "to_node_id": int(row["to_node_id"]),
            "segment_length_m": float(row["segment_length_m"]),
            "projection_fraction": float(row["projection_fraction"]),
            "prob_obs": float(row["prob_obs"]),
            "segment_id": segment_id,
            "transition_valid": not (index > 0 and breaks[index]),
        })

    return result

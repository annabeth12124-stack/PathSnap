
import networkx as nx


def build_transition_probabilities(
    graph,
    previous_candidates,
    current_candidates,
    max_hops=None,
):
    """
    Calculate transition probabilities between candidate road arcs.

    A transition is allowed when:
    - The previous and current candidates are the same arc, or
    - A directed path exists from the end of the previous arc
      to the start of the current arc.

    max_hops:
        None = allow any reachable directed path.
        Integer = limit the number of intermediate road-network edges.

    Shorter paths receive higher transition probabilities.
    """

    if not graph.is_directed():
        raise ValueError("The road graph must be directed")

    # Build arc ID -> (start node, end node) lookup.
    arc_endpoints = {}

    for source, target, key, data in graph.edges(
        keys=True, data=True
    ):
        arc_id = int(data.get("arc_id", key))
        arc_endpoints[arc_id] = (source, target)

    transitions = {}

    for previous_arc in previous_candidates:
        previous_arc = int(previous_arc)

        if previous_arc not in arc_endpoints:
            continue

        _, previous_end = arc_endpoints[previous_arc]

        valid_next = []

        for current_arc in current_candidates:
            current_arc = int(current_arc)

            if current_arc not in arc_endpoints:
                continue

            current_start, _ = arc_endpoints[current_arc]

            # Staying on the same road arc is allowed.
            if current_arc == previous_arc:
                valid_next.append((current_arc, 0))
                continue

            # Find the shortest directed path.
            try:
                distance = nx.shortest_path_length(
                    graph,
                    source=previous_end,
                    target=current_start,
                )

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            # Apply the optional hop limit.
            if max_hops is not None and distance > max_hops:
                continue

            valid_next.append((current_arc, distance))

        if not valid_next:
            continue

        # Shorter network paths receive higher probabilities.
        weights = [
            1.0 / (1.0 + distance)
            for _, distance in valid_next
        ]

        total = sum(weights)

        for (next_arc, _), weight in zip(valid_next, weights):
            transitions[(previous_arc, next_arc)] = (
                weight / total
            )

    return transitions

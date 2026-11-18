
import numpy as np
import networkx as nx


class TransitionModel:

    def __init__(self, nodes, arcs, beta=50.0, max_path_length=2000.0):
        self.beta = beta
        self.max_path_length = max_path_length
        self.graph = nx.MultiDiGraph()
        self._distance_cache = {}

        coords = nodes[["longitude", "latitude"]].to_numpy()

        self.arc_lengths = {}

        for arc_id, row in arcs.iterrows():
            u = int(row["from_node_id"])
            v = int(row["to_node_id"])

            lon1, lat1 = coords[u]
            lon2, lat2 = coords[v]

            lat1, lat2 = np.radians([lat1, lat2])
            dlat = lat2 - lat1
            dlon = np.radians(lon2 - lon1)

            h = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2)
                * np.sin(dlon / 2) ** 2
            )

            length = 6371000 * 2 * np.arcsin(
                np.sqrt(h)
            )

            self.arc_lengths[int(arc_id)] = float(length)

            self.graph.add_edge(
                u, v,
                key=int(arc_id),
                weight=float(length)
            )

    def network_distance(self, previous, current):

        previous_arc = int(previous["arc_id"]) if "arc_id" in previous else None
        current_arc = int(current["arc_id"]) if "arc_id" in current else None
        previous_fraction = float(previous["projection_fraction"])
        current_fraction = float(current["projection_fraction"])
        if (previous_arc is not None and previous_arc == current_arc
                and current_fraction >= previous_fraction):
            return (current_fraction - previous_fraction) * float(previous["segment_length_m"])

        previous_end = int(previous["to_node_id"])
        current_start = int(current["from_node_id"])

        if previous_end not in self._distance_cache:
            try:
                self._distance_cache[previous_end] = nx.single_source_dijkstra_path_length(
                    self.graph, previous_end,
                    cutoff=self.max_path_length,
                    weight="weight"
                )
            except nx.NodeNotFound:
                self._distance_cache[previous_end] = {}

        middle_distance = self._distance_cache[previous_end].get(current_start)
        if middle_distance is None:
            return np.inf

        previous_length = float(previous["segment_length_m"])
        current_length = float(current["segment_length_m"])

        total = (
            (1 - previous_fraction) * previous_length
            + middle_distance
            + current_fraction * current_length
        )

        if total > self.max_path_length:
            return np.inf

        return total

    def probability(self, previous, current, gps_distance):

        road_distance = self.network_distance(
            previous, current
        )

        if not np.isfinite(road_distance):
            return 0.0

        difference = abs(road_distance - gps_distance)

        return float(
            np.exp(-difference / self.beta)
        )


"""Candidate generation using a KD-tree spatial shortlist."""

import numpy as np
from scipy.spatial import cKDTree


def point_to_segment_distance(point, start, end):
    lon, lat = point
    lon1, lat1 = start
    lon2, lat2 = end

    scale_x = 111320 * np.cos(np.radians(lat))
    scale_y = 111320

    px = (lon - lon1) * scale_x
    py = (lat - lat1) * scale_y

    bx = (lon2 - lon1) * scale_x
    by = (lat2 - lat1) * scale_y

    length_squared = bx * bx + by * by

    if length_squared == 0:
        return np.hypot(px, py)

    t = (px * bx + py * by) / length_squared
    t = np.clip(t, 0, 1)

    return np.hypot(px - t * bx, py - t * by)


class CandidateGenerator:
    def __init__(self, graph):
        self.graph = graph
        self.node_ids = list(graph.nodes())

        self.coordinates = np.array([
            [
                graph.nodes[node]["longitude"],
                graph.nodes[node]["latitude"],
            ]
            for node in self.node_ids
        ])

        self.tree = cKDTree(self.coordinates)

        self.node_to_arcs = {}

        for source, target, arc_id in graph.edges(keys=True):
            arc = (int(arc_id), source, target)

            self.node_to_arcs.setdefault(source, set()).add(arc)
            self.node_to_arcs.setdefault(target, set()).add(arc)

    def generate_candidates(
        self, gps_point, radius=50, max_candidates=8
    ):
        lon, lat = gps_point

        # Use a wider shortlist radius to include long road segments.
        search_radius = 0.002

        indices = self.tree.query_ball_point(
            [lon, lat], search_radius
        )

        possible_arcs = set()

        for index in indices:
            node = self.node_ids[index]
            possible_arcs.update(
                self.node_to_arcs.get(node, set())
            )

        candidates = []

        for arc_id, source, target in possible_arcs:
            start = (
                self.graph.nodes[source]["longitude"],
                self.graph.nodes[source]["latitude"],
            )
            end = (
                self.graph.nodes[target]["longitude"],
                self.graph.nodes[target]["latitude"],
            )

            distance = point_to_segment_distance(
                gps_point, start, end
            )

            if distance <= radius:
                candidates.append((arc_id, distance))

        candidates.sort(key=lambda item: item[1])
        return candidates[:max_candidates]


def generate_candidates(
    graph, gps_point, radius=50, max_candidates=8
):
    generator = CandidateGenerator(graph)
    return generator.generate_candidates(
        gps_point, radius, max_candidates
    )

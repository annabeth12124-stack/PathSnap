
from collections import defaultdict


class RoadGraph:
    """
    Directed road graph.

    Node IDs identify road-network nodes.
    Arc IDs correspond to row indices in the original .arcs file.
    """

    def __init__(self, nodes, arcs):
        self.nodes = nodes
        self.arcs = arcs

        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)

        self.arc_lookup = {}

        for arc_id, row in arcs.iterrows():
            from_node = int(row["from_node_id"])
            to_node = int(row["to_node_id"])

            self.outgoing[from_node].append(arc_id)
            self.incoming[to_node].append(arc_id)

            self.arc_lookup[arc_id] = (
                from_node,
                to_node
            )

    def get_arc_endpoints(self, arc_id):
        return self.arc_lookup.get(int(arc_id))

    def are_connected(self, arc_a, arc_b):
        """
        Check whether arc_b can immediately follow arc_a.
        """
        endpoints_a = self.get_arc_endpoints(arc_a)
        endpoints_b = self.get_arc_endpoints(arc_b)

        if endpoints_a is None or endpoints_b is None:
            return False

        return endpoints_a[1] == endpoints_b[0]

    def get_outgoing_arcs(self, node_id):
        return self.outgoing.get(int(node_id), [])

    def get_incoming_arcs(self, node_id):
        return self.incoming.get(int(node_id), [])

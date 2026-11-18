
from pathlib import Path

from pathsnap_ml.candidates import read_trip
from route_reconstruction.road_graph import RoadGraph


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"

trip_id = "00000001"

# read_trip expects the parent directory and trip ID
track, nodes, arcs = read_trip(
    str(TRAIN_DIR / trip_id),
    trip_id
)
graph = RoadGraph(nodes, arcs)

print("GPS observations:", len(track))
print("Road nodes:", len(nodes))
print("Road arcs:", len(arcs))
print("Graph arc count:", len(graph.arc_lookup))

print("\nFirst five arcs:")
for arc_id in range(5):
    print(arc_id, graph.get_arc_endpoints(arc_id))

print("\nConnectivity test:")
print("Arc 0 -> Arc 1:", graph.are_connected(0, 1))


print("\nSearching for a valid connected arc pair...")

found = False

for arc_id, (from_node, to_node) in graph.arc_lookup.items():

    outgoing_arcs = graph.get_outgoing_arcs(to_node)

    if outgoing_arcs:
        next_arc = outgoing_arcs[0]

        print("First arc:", arc_id)
        print("First arc endpoints:", graph.get_arc_endpoints(arc_id))

        print("Next arc:", next_arc)
        print("Next arc endpoints:", graph.get_arc_endpoints(next_arc))

        print(
            "Connectivity verified:",
            graph.are_connected(arc_id, next_arc)
        )

        found = True
        break

if not found:
    print("No connected arc pair found.")

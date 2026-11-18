"""Build a directed road graph from one PathSnap trip's .nodes and .arcs files.

Dataset convention (from the teacher's Dataset Exploration document):
- .nodes: longitude latitude (no header); zero-based row number is node_id.
- .arcs: from_node_id to_node_id (no header); zero-based row number is arc_id.

A MultiDiGraph is used so that every original arc_id is preserved, even if
multiple arcs happen to connect the same pair of nodes.
"""
from pathlib import Path
import pandas as pd
import networkx as nx


def build_road_graph(nodes_file: str | Path, arcs_file: str | Path) -> nx.MultiDiGraph:
    nodes_file = Path(nodes_file)
    arcs_file = Path(arcs_file)

    if not nodes_file.is_file():
        raise FileNotFoundError(f"Nodes file not found: {nodes_file}")
    if not arcs_file.is_file():
        raise FileNotFoundError(f"Arcs file not found: {arcs_file}")

    # Both files are whitespace-separated and have no header row.
    nodes = pd.read_csv(nodes_file, sep=r"\s+", header=None, names=["longitude", "latitude"])
    arcs = pd.read_csv(arcs_file, sep=r"\s+", header=None, names=["from_node_id", "to_node_id"])

    if nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError("Expected exactly 2 columns in both .nodes and .arcs files")
    if nodes.isna().any().any() or arcs.isna().any().any():
        raise ValueError("Missing values found in .nodes or .arcs")

    # Convert IDs to integers and coordinates to numeric values.
    arcs["from_node_id"] = pd.to_numeric(arcs["from_node_id"], errors="raise").astype(int)
    arcs["to_node_id"] = pd.to_numeric(arcs["to_node_id"], errors="raise").astype(int)
    nodes["longitude"] = pd.to_numeric(nodes["longitude"], errors="raise")
    nodes["latitude"] = pd.to_numeric(nodes["latitude"], errors="raise")

    graph = nx.MultiDiGraph()

    # The row index is the node_id according to the supplied dataset guide.
    for node_id, row in nodes.iterrows():
        graph.add_node(
            int(node_id),
            longitude=float(row["longitude"]),
            latitude=float(row["latitude"]),
        )

    # The row index is the arc_id. Keep the direction as supplied in the file.
    invalid_rows = []
    for arc_id, row in arcs.iterrows():
        source = int(row["from_node_id"])
        target = int(row["to_node_id"])
        if source not in graph or target not in graph:
            invalid_rows.append((int(arc_id), source, target))
            continue
        graph.add_edge(source, target, key=int(arc_id), arc_id=int(arc_id))

    if invalid_rows:
        preview = invalid_rows[:5]
        raise ValueError(
            f"Found {len(invalid_rows)} arcs referencing node IDs that do not exist. "
            f"First examples (arc_id, from, to): {preview}"
        )

    graph.graph["nodes_file"] = str(nodes_file)
    graph.graph["arcs_file"] = str(arcs_file)
    graph.graph["node_count_from_file"] = len(nodes)
    graph.graph["arc_count_from_file"] = len(arcs)
    return graph


def print_graph_summary(graph: nx.MultiDiGraph) -> None:
    """Print basic checks; this does not claim the graph is geographically correct."""
    print(f"Nodes loaded: {graph.number_of_nodes():,}")
    print(f"Arcs loaded: {graph.number_of_edges():,}")
    print(f"Directed: {graph.is_directed()}")
    print(f"Weakly connected components: {nx.number_weakly_connected_components(graph):,}")
    print(f"Isolated nodes: {sum(1 for n in graph if graph.degree(n) == 0):,}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and inspect a PathSnap road graph")
    parser.add_argument("trip_folder", help="Folder containing TRIP_ID.nodes and TRIP_ID.arcs")
    parser.add_argument("trip_id", help="Trip ID, for example 00000001")
    args = parser.parse_args()

    folder = Path(args.trip_folder)
    graph = build_road_graph(folder / f"{args.trip_id}.nodes", folder / f"{args.trip_id}.arcs")
    print_graph_summary(graph)

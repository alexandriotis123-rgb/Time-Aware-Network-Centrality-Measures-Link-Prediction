"""
PART I - Question 3
Graph statistics functions.
"""


def compute_persistent_graph_statistics(persistent_pairs):
    """
    Compute the volumes of the persistent
    node and edge sets for every pair of
    consecutive temporal graphs.
    """

    persistent_nodes = []

    persistent_edges_1 = []

    persistent_edges_2 = []

    for graph_1, graph_2 in persistent_pairs:

        persistent_nodes.append(graph_1.number_of_nodes())

        persistent_edges_1.append(graph_1.number_of_edges())

        persistent_edges_2.append(graph_2.number_of_edges())

    return (persistent_nodes,persistent_edges_1,persistent_edges_2)
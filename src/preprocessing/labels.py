"""
Construction of labels for candidate edges.
"""


def build_labels( graph,candidate_edges):
    """
    Construct labels for candidate edges.

    """

    for node_u, node_v in candidate_edges:

        if graph.has_edge(node_u,node_v):
            yield 1
        else:
            yield 0
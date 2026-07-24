"""Construction of labels for candidate edges."""


def _canonical_edge(edge):
    """Return an undirected edge in a canonical form."""

    u, v = edge
    return (u, v) if u <= v else (v, u)


def build_labels(graph_1, graph_2, candidate_edges):
    """Label candidate edges as new links between two consecutive graphs.

    An edge receives label 1 if it appears in graph_2 but not in graph_1.
    """

    for node_u, node_v in candidate_edges:
        edge = _canonical_edge((node_u, node_v))
        has_in_graph_1 = graph_1.has_edge(*edge)
        has_in_graph_2 = graph_2.has_edge(*edge)

        if has_in_graph_2 and not has_in_graph_1:
            yield 1
        else:
            yield 0
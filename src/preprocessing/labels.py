"""Construction of labels for candidate edges."""


def _canonical_edge(edge):
    """Return an undirected edge in a canonical form."""

    u, v = edge
    return (u, v) if u <= v else (v, u)


def build_membership_labels(positive_edges, candidate_edges):
    """Label candidates by membership in an explicit positive-edge set."""

    positive_edge_set = {
        _canonical_edge(edge)
        for edge in positive_edges
    }

    for edge in candidate_edges:
        yield int(_canonical_edge(edge) in positive_edge_set)


def build_labels(graph_1, graph_2, candidate_edges):
    """Label candidate edges as new links between consecutive graphs.

    This legacy helper is retained for callers that explicitly need labels
    for edges present in ``graph_2`` but absent from ``graph_1``. The SNA experiment uses
    :func:`build_membership_labels` with separate training and test
    positive-edge sets.
    """

    for node_u, node_v in candidate_edges:
        edge = _canonical_edge((node_u, node_v))
        has_in_graph_1 = graph_1.has_edge(*edge)
        has_in_graph_2 = graph_2.has_edge(*edge)

        if has_in_graph_2 and not has_in_graph_1:
            yield 1
        else:
            yield 0

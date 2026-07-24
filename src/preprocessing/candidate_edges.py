import random
from itertools import combinations


def _canonical_edge(edge):
    """Return an undirected edge in a canonical form."""

    u, v = edge
    return (u, v) if u <= v else (v, u)


def build_candidate_edges(
    persistent_nodes,
    existing_edges=None,
    max_candidates=None,
    seed=42,
):
    """Construct candidate edges from a node set.

    If existing_edges is provided, any edge already present there is excluded
    from the candidate list. For development runs, a candidate cap can be used
    to avoid building an excessively large edge set.
    """

    node_list = sorted(persistent_nodes)
    candidate_edges = [
        _canonical_edge(edge)
        for edge in combinations(node_list, 2)
    ]

    if existing_edges is not None:
        existing_edge_set = {_canonical_edge(edge) for edge in existing_edges}
        candidate_edges = [
            edge for edge in candidate_edges if edge not in existing_edge_set
        ]

    if max_candidates is not None and len(candidate_edges) > max_candidates:
        rng = random.Random(seed)
        candidate_edges = rng.sample(candidate_edges, max_candidates)

    return candidate_edges
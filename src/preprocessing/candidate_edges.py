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
    existing_edge_set = {
        _canonical_edge(edge) for edge in existing_edges
    } if existing_edges is not None else set()

    if max_candidates is None:
        return [
            _canonical_edge(edge)
            for edge in combinations(node_list, 2)
            if _canonical_edge(edge) not in existing_edge_set
        ]

    rng = random.Random(seed)
    reservoir = []
    total = 0

    for edge in combinations(node_list, 2):
        canonical_edge = _canonical_edge(edge)
        if canonical_edge in existing_edge_set:
            continue

        total += 1
        if len(reservoir) < max_candidates:
            reservoir.append(canonical_edge)
        else:
            j = rng.randrange(total)
            if j < max_candidates:
                reservoir[j] = canonical_edge

    return reservoir
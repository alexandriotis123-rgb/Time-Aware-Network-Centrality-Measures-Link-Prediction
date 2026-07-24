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

    if max_candidates <= 0:
        return []

    rng = random.Random(seed)
    reservoir = []
    seen_edges = set()
    max_attempts = max(max_candidates * 25, 1000)
    attempts = 0

    while len(reservoir) < max_candidates and attempts < max_attempts:
        attempts += 1

        u = rng.choice(node_list)
        v = rng.choice(node_list)

        if u == v:
            continue

        canonical_edge = _canonical_edge((u, v))

        if canonical_edge in existing_edge_set or canonical_edge in seen_edges:
            continue

        reservoir.append(canonical_edge)
        seen_edges.add(canonical_edge)

    if len(reservoir) < max_candidates:
        for edge in combinations(node_list, 2):
            canonical_edge = _canonical_edge(edge)
            if canonical_edge in existing_edge_set or canonical_edge in seen_edges:
                continue

            reservoir.append(canonical_edge)
            seen_edges.add(canonical_edge)

            if len(reservoir) >= max_candidates:
                break

    return reservoir
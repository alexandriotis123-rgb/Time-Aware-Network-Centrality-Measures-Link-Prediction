from itertools import combinations

def build_candidate_edges(persistent_nodes):
    """
    Construct all candidate edges from the
    persistent node set.

    """

    return list(combinations(sorted(persistent_nodes),2))
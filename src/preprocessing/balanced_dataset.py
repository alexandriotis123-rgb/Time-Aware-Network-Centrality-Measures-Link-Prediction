"""
Balanced training dataset construction.
"""

import random


def balance_candidate_edges(
    candidate_edges,
    ground_truth_edges,
    seed=42
):
    """
    Random undersampling of the non-existing edges.

    Parameters
    ----------
    candidate_edges : iterable
        All candidate edges.

    ground_truth_edges : set
        Existing edges.

    seed : int
        Random seed.

    Returns
    -------
    list
        Balanced candidate edges.
    """

    positives = list(ground_truth_edges)

    negatives = [
        edge
        for edge in candidate_edges
        if edge not in ground_truth_edges
    ]

    rng = random.Random(seed)

    sampled_negatives = rng.sample(
        negatives,
        len(positives)
    )

    balanced_edges = positives + sampled_negatives

    rng.shuffle(balanced_edges)

    return balanced_edges
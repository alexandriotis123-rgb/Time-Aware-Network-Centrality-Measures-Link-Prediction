"""
Balanced training dataset construction.
"""

import random


def split_candidate_edges_stratified(
    candidate_edges,
    ground_truth_edges,
    val_ratio=0.3,
    seed=42,):
    """Split candidate edges into train/validation partitions
     while keeping class balance."""

    positives = [edge for edge in candidate_edges if edge in ground_truth_edges]
    negatives = [edge for edge in candidate_edges if edge not in ground_truth_edges]

    if not positives or not negatives:
        return list(candidate_edges), []

    rng = random.Random(seed)
    positives = list(positives)
    negatives = list(negatives)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    if len(positives) > 1:
        val_positive_size = int(round(len(positives) * val_ratio))
        val_positive_size = max(1, min(val_positive_size, len(positives) - 1))
    else:
        val_positive_size = 0

    if len(negatives) > 1:
        val_negative_size = int(round(len(negatives) * val_ratio))
        val_negative_size = max(1, min(val_negative_size, len(negatives) - 1))
    else:
        val_negative_size = 0

    val_candidates = positives[:val_positive_size] + negatives[:val_negative_size]
    train_candidates = positives[val_positive_size:] + negatives[val_negative_size:]

    rng.shuffle(val_candidates)
    rng.shuffle(train_candidates)

    return train_candidates, val_candidates


def balance_candidate_edges(
    candidate_edges,
    ground_truth_edges,
    seed=42):
    """
    Random undersampling of the non-existing edges.

    """

    positives = list(ground_truth_edges)

    negatives = [
        edge
        for edge in candidate_edges
        if edge not in ground_truth_edges]

    if not positives:
        return list(candidate_edges)

    rng = random.Random(seed)

    sampled_negatives = rng.sample(
        negatives,
        len(positives))

    balanced_edges = positives + sampled_negatives

    rng.shuffle(balanced_edges)

    return balanced_edges
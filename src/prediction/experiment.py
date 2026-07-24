"""
=========================================================
PART III - Questions 2 & 3
=========================================================
"""

import time
import random

from config import (
    DEBUG_MODE,
    DEBUG_MAX_CANDIDATE_EDGES,
    RANDOM_SEED,
    VALIDATION_RATIO,
)
from src.preprocessing.balanced_dataset import split_candidate_edges_stratified

from src.utils.helpers import SIMILARITY_COLUMNS, SIMILARITY_MEASURES
from src.prediction.training import (
    train_similarity_measure,
    improve_range_set,
)
from src.prediction.evaluation import (
    compute_accuracy,
    compute_balanced_accuracy,
    compute_precision,
    compute_recall,
    compute_tnr,
)
from src.preprocessing.candidate_edges import build_candidate_edges
from src.preprocessing.dataset import build_dataset


def _canonical_edge_set(edges):
    """Return canonical, non-self, undirected edge tuples."""

    return {
        (u, v) if u <= v else (v, u)
        for u, v in edges
        if u != v
    }


def _sample_edges(edges, maximum, seed):
    """Reproducibly cap an edge set while preserving all edges when possible."""

    edge_list = sorted(edges)
    if maximum is None or len(edge_list) <= maximum:
        return set(edge_list)

    rng = random.Random(seed)
    return set(rng.sample(edge_list, maximum))


def build_sna_edge_partitions(
    graph_1,
    graph_2,
    persistent_nodes,
    max_candidates=None,
    seed=RANDOM_SEED,
):
    """Construct balanced SNA training and test edge populations.

    Positive training edges are ``E_j*`` and positive test edges are
    ``E_(j+1)*``. Negative edges are sampled from pairs absent from both
    graphs, and the training/test negative samples are disjoint.
    """

    all_train_positive = _canonical_edge_set(graph_1.edges())
    all_test_positive = _canonical_edge_set(graph_2.edges())

    maximum_positives = None
    if max_candidates is not None:
        maximum_positives = max_candidates // 2

    train_positive = _sample_edges(
        all_train_positive,
        maximum_positives,
        seed,
    )
    test_positive = _sample_edges(
        all_test_positive,
        maximum_positives,
        seed + 1,
    )

    # Never label an actual edge as a negative, even when debug mode caps
    # the number of positive examples retained in either partition.
    excluded_edges = all_train_positive.union(all_test_positive)
    required_negative_count = len(train_positive) + len(test_positive)
    negative_edges = build_candidate_edges(
        persistent_nodes,
        existing_edges=excluded_edges,
        max_candidates=required_negative_count,
        seed=seed + 2,
    )

    if len(negative_edges) < required_negative_count:
        raise ValueError(
            "Not enough non-existing edges to construct balanced, "
            "disjoint training and test populations."
        )

    train_negative = negative_edges[:len(train_positive)]
    test_negative = negative_edges[
        len(train_positive):required_negative_count
    ]

    train_candidates = list(train_positive) + train_negative
    test_candidates = list(test_positive) + test_negative

    rng = random.Random(seed + 3)
    rng.shuffle(train_candidates)
    rng.shuffle(test_candidates)

    return {
        "train_positive": train_positive,
        "test_positive": test_positive,
        "train_negative": set(train_negative),
        "test_negative": set(test_negative),
        "train_candidates": train_candidates,
        "test_candidates": test_candidates,
    }


def evaluate_similarity_measures(
    train_dataset,
    val_dataset,
    train_candidates,
    val_candidates,
    ground_truth_train,
):
    """
    Train every similarity measure and store
    its optimal ranges and validation accuracy.
    """

    results = {}

    for similarity_measure in SIMILARITY_MEASURES:
        print(f"\nTraining {similarity_measure}...")
        start = time.perf_counter()

        # Train on the training partition
        ranges, train_acc, train_tpr, train_tnr, candidate_ranges = train_similarity_measure(
            train_dataset,
            train_candidates,
            ground_truth_train,
            similarity_measure,
        )

        # Compute the validation baseline for the starting range set
        similarity_index = SIMILARITY_COLUMNS[similarity_measure]
        val_acc, val_tpr, val_tnr = compute_balanced_accuracy(
            val_dataset,
            similarity_index,
            ranges,
            ground_truth_train,
            val_candidates,
        )

        # Attempt to improve the ranges using the validation partition
        improved_ranges, val_acc, val_tpr, val_tnr = improve_range_set(
            val_dataset,
            val_candidates,
            ground_truth_train,
            similarity_measure,
            ranges,
            val_acc,
            val_tpr,
            val_tnr,
            candidate_ranges,
        )

        elapsed = time.perf_counter() - start

        print(
            f"{similarity_measure:<4} | "
            f"VAL_BAL_ACC={val_acc:.4f} | "
            f"VAL_TPR={val_tpr:.4f} | "
            f"VAL_TNR={val_tnr:.4f} | "
            f"time={elapsed:.2f}s")

        results[similarity_measure] = {
            "ranges": improved_ranges,
            "train_accuracy": train_acc,
            "train_tpr": train_tpr,
            "train_tnr": train_tnr,
            "validation_accuracy": val_acc,
            "validation_tpr": val_tpr,
            "validation_tnr": val_tnr,
        }

    return results


def rank_similarity_measures(results):
    """
    Rank similarity measures according to holdout test accuracy.
    """

    ranking = sorted(
        results.items(),
        key=lambda item: item[1]["balanced_accuracy"],
        reverse=True)

    return ranking


def evaluate_trained_ranges(
    dataset,
    candidate_edges,
    ground_truth_edges,
    similarity_measure,
    similarity_ranges,
    top_k=100,
):
    """Evaluate fixed ranges on the balanced SNA test population."""

    similarity_index = SIMILARITY_COLUMNS[similarity_measure]

    predicted_edges = set()
    ranked_edges = []

    for row in dataset:
        edge = (row[0], row[1])
        score = row[similarity_index]
        ranked_edges.append((edge, score))

        if any(lower <= score <= upper for lower, upper in similarity_ranges):
            predicted_edges.add(edge)

    accuracy = compute_accuracy(predicted_edges, ground_truth_edges, candidate_edges)
    balanced_accuracy = compute_balanced_accuracy(
        predicted_edges,
        ground_truth_edges,
        candidate_edges,
    )
    precision = compute_precision(predicted_edges, ground_truth_edges)
    recall = compute_recall(predicted_edges, ground_truth_edges)
    tnr = compute_tnr(predicted_edges, ground_truth_edges, candidate_edges)

    ranked_edges = sorted(ranked_edges, key=lambda item: item[1], reverse=True)
    top_k_edges = {edge for edge, _ in ranked_edges[:top_k]}
    precision_at_k = compute_precision(top_k_edges, ground_truth_edges)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "tpr": recall,
        "tnr": tnr,
        "precision": precision,
        "recall": recall,
        "precision_at_k": precision_at_k,
    }


def run_training_experiment(persistent_pairs, persistent_node_sets):
    """
    Run the training experiment for every pair of
    successive persistent graphs.

    """

    experiment_results = []

    total_pairs = len(persistent_pairs)

    for pair_index, ((graph_1, graph_2), persistent_nodes) in enumerate(
        zip(persistent_pairs, persistent_node_sets),start=1):

        print(f"\n===== Persistent Pair {pair_index}/{total_pairs} =====")
        print(f"Persistent nodes: {len(persistent_nodes)}")

        max_candidates = DEBUG_MAX_CANDIDATE_EDGES if DEBUG_MODE else None
        edge_partitions = build_sna_edge_partitions(
            graph_1,
            graph_2,
            persistent_nodes,
            max_candidates=max_candidates,
            seed=RANDOM_SEED + 10 * pair_index,
        )

        ground_truth_train = edge_partitions["train_positive"]
        ground_truth_test = edge_partitions["test_positive"]
        training_pool = edge_partitions["train_candidates"]
        test_candidates = edge_partitions["test_candidates"]

        print(
            f"Training positives/negatives: "
            f"{len(ground_truth_train):,}/"
            f"{len(edge_partitions['train_negative']):,}"
        )
        print(
            f"Test positives/negatives: "
            f"{len(ground_truth_test):,}/"
            f"{len(edge_partitions['test_negative']):,}"
        )

        # Validation is a stratified subset of E_j* training data only.
        train_candidates, val_candidates = split_candidate_edges_stratified(
            training_pool,
            ground_truth_train,
            val_ratio=VALIDATION_RATIO,
            seed=RANDOM_SEED + pair_index,
        )

        print(f"Train candidates: {len(train_candidates):,} | Validation candidates: {len(val_candidates):,}")

        train_dataset = list(build_dataset(
            graph_1,
            graph_2,
            train_candidates,
            positive_edges=ground_truth_train,
        ))
        val_dataset = list(build_dataset(
            graph_1,
            graph_2,
            val_candidates,
            positive_edges=ground_truth_train,
        ))
        test_dataset = list(build_dataset(
            graph_1,
            graph_2,
            test_candidates,
            positive_edges=ground_truth_test,
        ))

        print(f"Train dataset size: {len(train_dataset):,} | Val size: {len(val_dataset):,} | Test size: {len(test_dataset):,}")

        # Evaluate/training per similarity measure using train + validation
        results = evaluate_similarity_measures(
            train_dataset,
            val_dataset,
            train_candidates,
            val_candidates,
            ground_truth_train,
        )

        # Final evaluation on the holdout test set using the improved ranges
        full_evaluation = {}
        for similarity_measure, info in results.items():
            full_evaluation[similarity_measure] = evaluate_trained_ranges(
                test_dataset,
                test_candidates,
                ground_truth_test,
                similarity_measure,
                info["ranges"],
            )

        print("\nValidation scores")
        for measure, info in results.items():
            print(
                f"{measure:<4} "
                f"VAL_BAL_ACC={info['validation_accuracy']:.4f} "
                f"VAL_TPR={info['validation_tpr']:.4f} "
                f"VAL_TNR={info['validation_tnr']:.4f}")

        ranking = rank_similarity_measures(full_evaluation)
        print("\nBalanced temporal test ranking")

        for position, (measure, info) in enumerate(ranking, start=1):
            evaluation_info = info
            print(
                f"{position}. "
                f"{measure:<4} "
                f"BAL_ACC={evaluation_info['balanced_accuracy']:.4f} "
                f"TPR={evaluation_info['tpr']:.4f} "
                f"TNR={evaluation_info['tnr']:.4f} "
                f"PREC={evaluation_info['precision']:.4f} "
                f"P@K={evaluation_info['precision_at_k']:.4f}")

        experiment_results.append(ranking)

    return experiment_results

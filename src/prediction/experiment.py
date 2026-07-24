"""
=========================================================
PART III - Questions 2 & 3
=========================================================
"""

import time

from config import DEBUG_MODE, DEBUG_MAX_CANDIDATE_EDGES, RANDOM_SEED, TEST_HOLDOUT_RATIO
from src.preprocessing.balanced_dataset import (balance_candidate_edges)

from src.utils.helpers import SIMILARITY_COLUMNS, SIMILARITY_MEASURES
from src.prediction.training import (
    train_similarity_measure,
    improve_range_set,
)
import random
from src.prediction.evaluation import (
    compute_accuracy,
    compute_precision,
    compute_recall,
)
from src.preprocessing.candidate_edges import build_candidate_edges
from src.preprocessing.dataset import build_dataset


def evaluate_similarity_measures(
    train_dataset,
    val_dataset,
    train_candidates,
    val_candidates,
    graph_edges,
    ground_truth_remaining,
    ground_truth_test,
):
    """
    Train every similarity measure and store
    its optimal ranges and training accuracy.
    """

    results = {}

    for similarity_measure in SIMILARITY_MEASURES:
        print(f"\nTraining {similarity_measure}...")
        start = time.perf_counter()

        # Train on the training partition
        ranges, accuracy, tpr, tnr, candidate_ranges = train_similarity_measure(
            train_dataset,
            train_candidates,
            ground_truth_remaining,
            similarity_measure,
        )

        # Attempt to improve the ranges using the validation partition
        improved_ranges, improved_acc, improved_tpr, improved_tnr = improve_range_set(
            val_dataset,
            val_candidates,
            ground_truth_remaining,
            similarity_measure,
            ranges,
            accuracy,
            tpr,
            tnr,
            candidate_ranges,
        )

        elapsed = time.perf_counter() - start

        print(
            f"{similarity_measure:<4} | "
            f"ACC={improved_acc:.4f} | "
            f"TPR={improved_tpr:.4f} | "
            f"TNR={improved_tnr:.4f} | "
            f"time={elapsed:.2f}s")

        results[similarity_measure] = {
            "ranges": improved_ranges,
            "accuracy": improved_acc,
            "tpr": improved_tpr,
            "tnr": improved_tnr,
        }

    return results


def rank_similarity_measures(results):
    """
    Rank similarity measures according to
    their prediction accuracy.
    """

    ranking = sorted(
        results.items(),
        key=lambda item: item[1]["accuracy"],
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
    """Evaluate a trained similarity range set on a full candidate population."""

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
    precision = compute_precision(predicted_edges, ground_truth_edges)
    recall = compute_recall(predicted_edges, ground_truth_edges)

    ranked_edges = sorted(ranked_edges, key=lambda item: item[1], reverse=True)
    top_k_edges = {edge for edge, _ in ranked_edges[:top_k]}
    precision_at_k = compute_precision(top_k_edges, ground_truth_edges)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "precision_at_k": precision_at_k,
    }


def run_training_experiment(persistent_pairs,persistent_node_sets):
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

        existing_edges = set(graph_1.edges())
        max_candidates = (
            DEBUG_MAX_CANDIDATE_EDGES if DEBUG_MODE else None
        )
        candidate_edges = build_candidate_edges(
            persistent_nodes,
            existing_edges=existing_edges,
            max_candidates=max_candidates,
        )

        print(f"Candidate edges: {len(candidate_edges):,}")

        # Create a reproducible holdout test split from the full candidates
        rnd = random.Random(RANDOM_SEED)
        candidate_list = list(candidate_edges)
        rnd.shuffle(candidate_list)
        test_size = int(len(candidate_list) * TEST_HOLDOUT_RATIO)
        test_candidates = candidate_list[:test_size]
        remaining_candidates = candidate_list[test_size:]

        print(f"Test candidates (holdout): {len(test_candidates):,}")

        graph_edges = {tuple(sorted(edge)) for edge in graph_2.edges()}
        candidate_edge_set_full = {tuple(sorted(edge)) for edge in candidate_list}
        ground_truth_full = graph_edges.intersection(candidate_edge_set_full)

        # ground truth for test and remaining partitions
        test_candidate_set = {tuple(sorted(edge)) for edge in test_candidates}
        remaining_candidate_set = {tuple(sorted(edge)) for edge in remaining_candidates}

        ground_truth_test = graph_edges.intersection(test_candidate_set)
        ground_truth_remaining = graph_edges.intersection(remaining_candidate_set)

        # Balance only the remaining (non-test) candidates and then split into train/val
        balanced_candidate_edges = balance_candidate_edges(
            remaining_candidates,
            ground_truth_remaining)

        print(f"Balanced candidate edges (train+val pool): {len(balanced_candidate_edges):,}")

        # Split balanced pool into train/validation (reproducible)
        rnd.shuffle(balanced_candidate_edges)
        val_ratio = 0.2
        val_size = int(len(balanced_candidate_edges) * val_ratio)
        val_candidates = balanced_candidate_edges[:val_size]
        train_candidates = balanced_candidate_edges[val_size:]

        print(f"Train candidates: {len(train_candidates):,} | Validation candidates: {len(val_candidates):,}")

        train_dataset = list(build_dataset(graph_1, graph_2, train_candidates))
        val_dataset = list(build_dataset(graph_1, graph_2, val_candidates))
        test_dataset = list(build_dataset(graph_1, graph_2, test_candidates))

        print(f"Train dataset size: {len(train_dataset):,} | Val size: {len(val_dataset):,} | Test size: {len(test_dataset):,}")

        # Evaluate/training per similarity measure using train + validation
        results = evaluate_similarity_measures(
            train_dataset,
            val_dataset,
            train_candidates,
            val_candidates,
            graph_edges,
            ground_truth_remaining,
            ground_truth_test,
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

        ranking = rank_similarity_measures(results)
        print("\nRanking")

        for position, (measure, info) in enumerate(ranking, start=1):
            evaluation_info = full_evaluation[measure]
            print(
                f"{position}. "
                f"{measure:<4} "
                f"ACC={evaluation_info['accuracy']:.4f} "
                f"PREC={evaluation_info['precision']:.4f} "
                f"REC={evaluation_info['recall']:.4f} "
                f"P@K={evaluation_info['precision_at_k']:.4f}")

        experiment_results.append(ranking)

    return experiment_results
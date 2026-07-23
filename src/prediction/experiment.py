"""
=========================================================
PART III - Questions 2 & 3
=========================================================
"""

import time

from src.preprocessing.balanced_dataset import (balance_candidate_edges)

from src.utils.helpers import SIMILARITY_MEASURES
from src.prediction.training import train_similarity_measure
from src.preprocessing.candidate_edges import build_candidate_edges
from src.preprocessing.dataset import build_dataset


def evaluate_similarity_measures(
    dataset,
    candidate_edges,
    ground_truth_edges):
    """
    Train every similarity measure and store
    its optimal ranges and training accuracy.
    """

    results = {}

    for similarity_measure in SIMILARITY_MEASURES:
        print(f"\nTraining {similarity_measure}...")
        start = time.perf_counter()

        best_ranges, accuracy,tpr, tnr = train_similarity_measure(
            dataset,
            candidate_edges,
            ground_truth_edges,
            similarity_measure)

        elapsed = time.perf_counter() - start

        print(
            f"{similarity_measure:<4} | "
            f"ACC={accuracy:.4f} | "
            f"TPR={tpr:.4f} | "
            f"TNR={tnr:.4f} | "
            f"time={elapsed:.2f}s")

        results[similarity_measure] = {
            "ranges": best_ranges,
            "accuracy": accuracy,
            "tpr": tpr,
            "tnr": tnr}

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

        candidate_edges = build_candidate_edges(persistent_nodes)

        print(f"Candidate edges: {len(candidate_edges):,}")

        graph_edges = set(graph_2.edges())

        ground_truth_edges = graph_edges.intersection(candidate_edges)

        balanced_candidate_edges = balance_candidate_edges(
            candidate_edges,
            ground_truth_edges)

        print(f"Balanced candidate edges: {len(balanced_candidate_edges):,}")

        dataset = list(
            build_dataset(
                graph_1,
                graph_2,
                balanced_candidate_edges))

        print(f"Dataset size: {len(dataset):,}")

        results = evaluate_similarity_measures(
            dataset,
            balanced_candidate_edges,
            ground_truth_edges)

        ranking = rank_similarity_measures(results)
        print("\nRanking")

        for position, (measure, info) in enumerate(ranking, start=1):
            print(
                f"{position}. "
                f"{measure:<4} "
                f"ACC={info['accuracy']:.4f} "
                f"TPR={info['tpr']:.4f} "
                f"TNR={info['tnr']:.4f}")

        experiment_results.append(ranking)

    return experiment_results
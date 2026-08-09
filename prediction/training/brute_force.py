from src.prediction.training import (
    generate_similarity_ranges as _generate_similarity_ranges,
    find_best_single_range as _find_best_single_range,
    improve_range_set as _improve_range_set,
    train_similarity_measure as _train_similarity_measure,)
from src.utils.helpers import SIMILARITY_COLUMNS


def generate_similarity_ranges(similarity_scores):
    return _generate_similarity_ranges(similarity_scores)


def find_best_single_range(dataset, candidate_edges, ground_truth_edges, similarity_measure):
    result = _find_best_single_range(
        dataset,
        candidate_edges,
        ground_truth_edges,
        similarity_measure,
)
    return result[0], result[1]


def improve_range_set(
    dataset,
    candidate_edges,
    ground_truth_edges,
    similarity_measure,
    current_ranges,
    current_accuracy,
):
    similarity_index = SIMILARITY_COLUMNS[similarity_measure]
    similarity_scores = [row[similarity_index] for row in dataset]
    candidate_ranges = _generate_similarity_ranges(similarity_scores)

    result = _improve_range_set(
        dataset,
        candidate_edges,
        ground_truth_edges,
        similarity_measure,
        current_ranges,
        current_accuracy,
        0,
        0,
        candidate_ranges,)
    return result[0], result[1]


def train_similarity_measure(
    dataset,
    candidate_edges,
    ground_truth_edges,
    similarity_measure,):
    result = _train_similarity_measure(
        dataset,
        candidate_edges,
        ground_truth_edges,
        similarity_measure,)
    return result[0], result[1]


__all__ = [
    "generate_similarity_ranges",
    "find_best_single_range",
    "improve_range_set",
    "train_similarity_measure",]

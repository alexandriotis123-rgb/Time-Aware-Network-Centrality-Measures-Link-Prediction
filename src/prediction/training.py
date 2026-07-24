"""
Brute-force training algorithm.

Description
-----------
Evaluates every possible continuous similarity interval.

This implementation is preserved for comparison with the
optimized training algorithms.
"""



from src.utils.helpers import SIMILARITY_COLUMNS
from src.prediction.evaluation import compute_accuracy, compute_balanced_accuracy, compute_lambda
from config import MAX_UNIQUE_SCORES
from config import MAX_INTERVALS
from collections import defaultdict

MIN_IMPROVEMENT = 1e-4


def generate_similarity_ranges(similarity_scores):
    """
    Generate all possible continuous similarity score ranges.

    
    """

    MAX_SCORES = MAX_UNIQUE_SCORES

    unique_scores = sorted(set(similarity_scores))

    if len(unique_scores) > MAX_SCORES:

        step = len(unique_scores) / MAX_SCORES

        unique_scores = [
            unique_scores[int(i * step)]
            for i in range(MAX_SCORES)]

    candidate_ranges = []

    number_of_scores = len(unique_scores)

    for lower_index in range(number_of_scores):
        for upper_index in range(lower_index,number_of_scores):

            lower_bound = unique_scores[lower_index]

            upper_bound = unique_scores[upper_index]

            candidate_ranges.append([(lower_bound, upper_bound)])
 
    return candidate_ranges


def find_best_single_range(
    dataset,
    candidate_edges,
    ground_truth_edges,
    similarity_measure):
    """
    Determine the optimal similarity range set that maximizes
    prediction accuracy.

    """

    similarity_index = SIMILARITY_COLUMNS[similarity_measure]

    similarity_scores = [row[similarity_index] for row in dataset]

    score_statistics = defaultdict(lambda: [0, 0])

    for row in dataset:

        score = row[similarity_index]
        edge = (row[0], row[1])

        score_statistics[score][0] += 1

        if edge in ground_truth_edges:
            score_statistics[score][1] += 1

    #print(f"\nDistribution for {similarity_measure}")

    for score in sorted(score_statistics):

        total, positives = score_statistics[score]

        #print(
            #f"Score={score:>4} | "
            #f"Total={total:>8} | "
            #f"Positives={positives:>5}")
    

    #unique_scores = len(set(similarity_scores))

    #print(f"{similarity_measure}: {unique_scores:,} unique scores")

    candidate_ranges = generate_similarity_ranges(similarity_scores)

    #print(f"{similarity_measure}: {len(candidate_ranges):,} candidate ranges")


    best_ranges = None

    best_accuracy = float("-inf")

    best_tpr = 0
    best_tnr = 0

    for similarity_ranges in candidate_ranges:

        accuracy, tpr, tnr = compute_balanced_accuracy(
            dataset,
            similarity_index,
            similarity_ranges,
            ground_truth_edges,
            candidate_edges)

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_ranges = similarity_ranges
            best_tpr = tpr
            best_tnr = tnr

            #print(
                #f"NEW BEST -> "
             #   f"ACC={accuracy:.10f} "
              #  f"TPR={tpr:.4f} "
              #  f"TNR={tnr:.4f} "
              #  f"range={similarity_ranges}")

    print(f"\n{similarity_measure}")
    print(f"Candidate edges: {len(candidate_edges)}")
    print(f"Ground truth: {len(ground_truth_edges)}")
    print(f"Best ranges: {best_ranges}")
    print(f"BAL_ACC={best_accuracy:.4f}  TPR={best_tpr:.4f}  TNR={best_tnr:.4f}")
    print(f"lambda={compute_lambda(ground_truth_edges, candidate_edges):.4f}")

    return ( best_ranges,best_accuracy,best_tpr,best_tnr,candidate_ranges)

def overlaps(interval_1,interval_2):
    """
    Check whether two intervals overlap.
    """

    lower_1, upper_1 = interval_1
    lower_2, upper_2 = interval_2

    return not (upper_1 < lower_2 or upper_2 < lower_1)


def improve_range_set(
    dataset,
    candidate_edges,
    ground_truth_edges,
    similarity_measure,
    current_ranges,
    current_accuracy,
    current_tpr,
    current_tnr,
    candidate_ranges,
    max_intervals=MAX_INTERVALS):
    """
    Improve a similarity range set by adding
    non-overlapping intervals.
    """

    similarity_index = SIMILARITY_COLUMNS[similarity_measure]

    # Evaluate the current ranges on the validation dataset first, so the
    # improvement threshold is compared consistently on validation.
    best_accuracy, best_tpr, best_tnr = compute_balanced_accuracy(
        dataset,
        similarity_index,
        current_ranges,
        ground_truth_edges,
        candidate_edges,
    )

    best_ranges = list(current_ranges)

    # Keep track of which candidates are still available
    remaining_candidates = list(candidate_ranges)

    # Stop when no candidate gives meaningful improvement or when we've
    # reached the configured maximum number of intervals.
    while True:
        found_improvement = False
        best_step = None
        best_step_accuracy = best_accuracy
        best_step_tpr = best_tpr
        best_step_tnr = best_tnr

        for candidate in remaining_candidates:
            candidate_interval = candidate[0]

            if any(overlaps(candidate_interval, interval) for interval in best_ranges):
                continue

            new_ranges = sorted(best_ranges + [candidate_interval], key=lambda interval: interval[0])

            accuracy, tpr, tnr = compute_balanced_accuracy(
                dataset,
                similarity_index,
                new_ranges,
                ground_truth_edges,
                candidate_edges,
            )

            improvement = accuracy - best_accuracy

            if improvement > MIN_IMPROVEMENT and accuracy > best_step_accuracy:
                best_step = candidate_interval
                best_step_accuracy = accuracy
                best_step_tpr = tpr
                best_step_tnr = tnr
                found_improvement = True

        if not found_improvement:
            break

        best_ranges = sorted(best_ranges + [best_step], key=lambda interval: interval[0])
        best_accuracy = best_step_accuracy
        best_tpr = best_step_tpr
        best_tnr = best_step_tnr

        print(f"Added interval {best_step} -> BAL_ACC={best_accuracy:.4f} | intervals={len(best_ranges)}")

        if len(best_ranges) >= max_intervals:
            break

        remaining_candidates = [
            c for c in remaining_candidates
            if not any(overlaps(c[0], r) for r in best_ranges)
        ]

    return (best_ranges, best_accuracy, best_tpr, best_tnr)


def train_similarity_measure(
    dataset,
    candidate_edges,
    ground_truth_edges,
    similarity_measure):
    
    ranges, accuracy, tpr, tnr, candidate_ranges = find_best_single_range(
        dataset,
        candidate_edges,
        ground_truth_edges,
        similarity_measure)
    '''
    ranges, accuracy,tpr, tnr, = improve_range_set(
        dataset,
        candidate_edges,
        ground_truth_edges,
        similarity_measure,
        ranges,
        accuracy,
        tpr,
        tnr,
        candidate_ranges)
    print(f"\n=== {similarity_measure} ===")
    print(f"Best ranges: {ranges}")
    print(f"Best accuracy: {accuracy:.10f}")
    '''
    # Return candidate_ranges as well so callers can attempt
    # to improve the range set using a validation partition.
    return (ranges, accuracy, tpr, tnr, candidate_ranges)
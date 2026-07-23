"""
Brute-force training algorithm.

Description
-----------
Evaluates every possible continuous similarity interval.

This implementation is preserved for comparison with the
optimized training algorithms.
"""



from src.utils.helpers import SIMILARITY_COLUMNS
from src.prediction.evaluation import compute_accuracy
from config import MAX_UNIQUE_SCORES
from collections import defaultdict


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

        accuracy,tpr,tnr = compute_accuracy(
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
    print(f"ACC={best_accuracy:.4f}  TPR={best_tpr:.4f}  TNR={best_tnr:.4f}")
    print(f"lambda={len(ground_truth_edges)/len(candidate_edges):.4f}")

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
    candidate_ranges):
    """
    Improve a similarity range set by adding
    non-overlapping intervals.
    """

    similarity_index = SIMILARITY_COLUMNS[similarity_measure]


    best_ranges = current_ranges
    best_accuracy = current_accuracy

    best_tpr = current_tpr
    best_tnr = current_tnr

    for candidate in candidate_ranges:

        candidate_interval = candidate[0]

        overlap = False

        for interval in current_ranges:
            if overlaps(candidate_interval,interval):
                overlap = True
                break
        if overlap:
            continue

        new_ranges = sorted(
            current_ranges + candidate,
            key=lambda interval: interval[0])

        accuracy,tpr,tnr = compute_accuracy(
            dataset,
            similarity_index,
            new_ranges,
            ground_truth_edges,
            candidate_edges)
        
        improvement = accuracy - best_accuracy

        if improvement > 0:
            print(
                f"Candidate {new_ranges} "
                f"improves ACC by {improvement:.10f}")

        if accuracy > best_accuracy:
            print("Improvement found")
            print(f"Previous ranges: {best_ranges}")
            print(f"Candidate ranges: {new_ranges}")
            print(f"ACC: {best_accuracy:.6f} -> {accuracy:.6f}")


            best_accuracy = accuracy
            best_ranges = new_ranges
            best_tpr = tpr
            best_tnr = tnr
            

    return (best_ranges,best_accuracy,best_tpr,best_tnr)


def train_similarity_measure(
    dataset,
    candidate_edges,
    ground_truth_edges,
    similarity_measure):
    
    ranges, accuracy,tpr, tnr,candidate_ranges = find_best_single_range(
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
    improved = True

    while improved:

        previous_accuracy = accuracy

        ranges, accuracy, tpr, tnr = improve_range_set(
            dataset,
            candidate_edges,
            ground_truth_edges,
            similarity_measure,
            ranges,
            accuracy,
            tpr,
            tnr,
            candidate_ranges)

        improved = accuracy > previous_accuracy
        print(f"Current ranges: {ranges}")
        print(f"ACC={accuracy:.4f}")
    return (ranges,accuracy,tpr, tnr,)
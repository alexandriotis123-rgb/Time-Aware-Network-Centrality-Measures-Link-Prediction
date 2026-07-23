
def compute_accuracy(
    dataset,
    similarity_index,
    similarity_ranges,
    ground_truth_edges,
    candidate_edges,
):
    """
    Compute prediction accuracy directly from the dataset,
    without constructing predicted edge sets.
    """

    total_positive = len(ground_truth_edges)
    total_negative = len(candidate_edges) - total_positive

    tp = 0
    tn = 0

    for row in dataset:

        edge = (row[0], row[1])
        score = row[similarity_index]

        predicted = False

        for lower, upper in similarity_ranges:
            if lower <= score <= upper:
                predicted = True
                break

        actual = edge in ground_truth_edges

        if predicted:
            if actual:
                tp += 1
        else:
            if not actual:
                tn += 1

    lambda_coefficient = total_positive / len(candidate_edges)

    tpr = tp / total_positive if total_positive else 0

    tnr = tn / total_negative if total_negative else 0


    accuracy = (
    lambda_coefficient * tpr
    + (1 - lambda_coefficient) * tnr)

    return accuracy, tpr, tnr
''' ΑΛΛΑΖΩ ΤΟ FILE ΚΑΙΝΟΥΡΓΙΑ ΛΥΣΗ Η ΠΑΝΩ.

def compute_lambda(ground_truth_edges,candidate_edges):
    """
    Compute the lambda coefficient.

    """

    return (len(ground_truth_edges)/len(candidate_edges))


def compute_tpr(predicted_edges,ground_truth_edges):
    """
    Compute the True Positive Rate (TPR).

    """

    true_positive_edges = (predicted_edges.intersection(ground_truth_edges))

    return (len(true_positive_edges)/len(ground_truth_edges))


def compute_tnr(predicted_edges,ground_truth_edges,candidate_edges):
    """
    Compute the True Negative Rate (TNR).

    """

    candidate_edges = set(candidate_edges)
    predicted_negative_edges = (candidate_edges - predicted_edges)

    ground_truth_negative_edges = (candidate_edges- ground_truth_edges)

    true_negative_edges = (
        predicted_negative_edges.intersection(
            ground_truth_negative_edges))

    return (len(true_negative_edges)/len(ground_truth_negative_edges))


def compute_accuracy(predicted_edges,ground_truth_edges,candidate_edges):
    """
    Compute the prediction accuracy.

    """
    lambda_coefficient = compute_lambda(
        ground_truth_edges,
        candidate_edges)

    true_positive_rate = compute_tpr(
        predicted_edges,
        ground_truth_edges)

    true_negative_rate = compute_tnr(
        predicted_edges,
        ground_truth_edges,
        candidate_edges)

    return (
        lambda_coefficient
        * true_positive_rate
        +
        (1 - lambda_coefficient)
        * true_negative_rate)

'''
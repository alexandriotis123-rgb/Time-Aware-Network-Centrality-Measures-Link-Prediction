def _coerce_edge_set(edge_collection):

    if edge_collection is None:
        return set()

    if isinstance(edge_collection, set):
        return edge_collection

    return set(edge_collection)


def _compute_accuracy_from_dataset(
    dataset,
    similarity_index,
    similarity_ranges,
    ground_truth_edges,
    candidate_edges,):

    ground_truth_edges = _coerce_edge_set(ground_truth_edges)
    candidate_edges = _coerce_edge_set(candidate_edges)

    ground_truth_in_candidates = ground_truth_edges.intersection(candidate_edges)

    total_positive = len(ground_truth_in_candidates)
    total_negative = len(candidate_edges) - total_positive

    tp = 0
    tn = 0

    for row in dataset:
        edge = (row[0], row[1])

        if edge not in candidate_edges:
            continue

        score = row[similarity_index]

        predicted = False
        for lower, upper in similarity_ranges:
            if lower <= score <= upper:
                predicted = True
                break

        actual = edge in ground_truth_in_candidates

        if predicted and actual:
            tp += 1
        elif (not predicted) and (not actual):
            tn += 1

    lambda_coefficient = (total_positive / len(candidate_edges) if candidate_edges else 0)

    tpr = tp / total_positive if total_positive else 0
    tnr = tn / total_negative if total_negative else 0
    accuracy = (
        lambda_coefficient * tpr
        + (1 - lambda_coefficient) * tnr)

    return accuracy, tpr, tnr


def _compute_balanced_accuracy_from_dataset(
    dataset,
    similarity_index,
    similarity_ranges,
    ground_truth_edges,
    candidate_edges,):

    ground_truth_edges = _coerce_edge_set(ground_truth_edges)
    candidate_edges = _coerce_edge_set(candidate_edges)

    ground_truth_in_candidates = ground_truth_edges.intersection(candidate_edges)
    total_positive = len(ground_truth_in_candidates)
    total_negative = len(candidate_edges) - total_positive

    tp = 0
    tn = 0

    for row in dataset:
        edge = (row[0], row[1])
        if edge not in candidate_edges:
            continue

        score = row[similarity_index]
        predicted = any(lower <= score <= upper for lower, upper in similarity_ranges)
        actual = edge in ground_truth_in_candidates

        if predicted and actual:
            tp += 1
        elif (not predicted) and (not actual):
            tn += 1

    tpr = tp / total_positive if total_positive else 0
    tnr = tn / total_negative if total_negative else 0
    balanced_accuracy = (tpr + tnr) / 2

    return balanced_accuracy, tpr, tnr


def compute_lambda(ground_truth_edges, candidate_edges):

    ground_truth_edges = _coerce_edge_set(ground_truth_edges)
    candidate_edges = _coerce_edge_set(candidate_edges)

    if not candidate_edges:
        return 0

    ground_truth_in_candidates = ground_truth_edges.intersection(candidate_edges)
    return len(ground_truth_in_candidates) / len(candidate_edges)


def compute_tpr(predicted_edges, ground_truth_edges):

    predicted_edges = _coerce_edge_set(predicted_edges)
    ground_truth_edges = _coerce_edge_set(ground_truth_edges)

    if not ground_truth_edges:
        return 0

    true_positive_edges = predicted_edges.intersection(ground_truth_edges)
    return len(true_positive_edges) / len(ground_truth_edges)


def compute_precision(predicted_edges, ground_truth_edges):

    predicted_edges = _coerce_edge_set(predicted_edges)
    ground_truth_edges = _coerce_edge_set(ground_truth_edges)

    if not predicted_edges:
        return 0

    return len(predicted_edges.intersection(ground_truth_edges)) / len(predicted_edges)


def compute_recall(predicted_edges, ground_truth_edges):

    predicted_edges = _coerce_edge_set(predicted_edges)
    ground_truth_edges = _coerce_edge_set(ground_truth_edges)

    if not ground_truth_edges:
        return 0

    return len(predicted_edges.intersection(ground_truth_edges)) / len(ground_truth_edges)


def compute_tnr(predicted_edges, ground_truth_edges, candidate_edges):

    predicted_edges = _coerce_edge_set(predicted_edges)
    ground_truth_edges = _coerce_edge_set(ground_truth_edges)
    candidate_edges = _coerce_edge_set(candidate_edges)

    ground_truth_negative_edges = candidate_edges - ground_truth_edges

    if not ground_truth_negative_edges:
        return 0

    predicted_negative_edges = candidate_edges - predicted_edges
    true_negative_edges = predicted_negative_edges.intersection(
        ground_truth_negative_edges)

    return len(true_negative_edges) / len(ground_truth_negative_edges)


def compute_accuracy(*args):

    if len(args) == 3:
        predicted_edges, ground_truth_edges, candidate_edges = args
        predicted_edges = _coerce_edge_set(predicted_edges)
        ground_truth_edges = _coerce_edge_set(ground_truth_edges)
        candidate_edges = _coerce_edge_set(candidate_edges)
        ground_truth_edges = ground_truth_edges.intersection(candidate_edges)

        lambda_coefficient = compute_lambda(ground_truth_edges, candidate_edges)
        true_positive_rate = compute_tpr(predicted_edges, ground_truth_edges)
        true_negative_rate = compute_tnr(
            predicted_edges,
            ground_truth_edges,
            candidate_edges,)

        return (
            lambda_coefficient * true_positive_rate
            + (1 - lambda_coefficient) * true_negative_rate)

    if len(args) == 5:
        return _compute_accuracy_from_dataset(*args)

    raise TypeError("compute_accuracy expects either 3 or 5 arguments")


def compute_balanced_accuracy(*args):

    if len(args) == 3:
        predicted_edges, ground_truth_edges, candidate_edges = args
        predicted_edges = _coerce_edge_set(predicted_edges)
        ground_truth_edges = _coerce_edge_set(ground_truth_edges)
        candidate_edges = _coerce_edge_set(candidate_edges)
        ground_truth_edges = ground_truth_edges.intersection(candidate_edges)

        true_positive_rate = compute_tpr(predicted_edges, ground_truth_edges)
        true_negative_rate = compute_tnr(
            predicted_edges,
            ground_truth_edges,
            candidate_edges,)

        return (true_positive_rate + true_negative_rate) / 2

    if len(args) == 5:
        return _compute_balanced_accuracy_from_dataset(*args)

    raise TypeError("compute_balanced_accuracy expects either 3 or 5 arguments")

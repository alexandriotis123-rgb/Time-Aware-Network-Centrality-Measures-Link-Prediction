"""
Link prediction based on similarity score ranges.
"""

from src.utils.helpers import SIMILARITY_COLUMNS


def predict_edges(dataset,similarity_measure,similarity_ranges):
    """
    Predict edges using one or more similarity score ranges.

    
    """

    if similarity_measure not in SIMILARITY_COLUMNS:
        raise ValueError( f"Unknown similarity measure: {similarity_measure}")

    similarity_index = SIMILARITY_COLUMNS[similarity_measure]

    predicted_edges = set()

    for row in dataset:

        node_u = row[0]
        node_v = row[1]

        similarity_score = row[similarity_index]

        for lower_bound, upper_bound in similarity_ranges:

            if (lower_bound<= similarity_score <= upper_bound):
                predicted_edges.add((node_u, node_v))
                break

    return predicted_edges
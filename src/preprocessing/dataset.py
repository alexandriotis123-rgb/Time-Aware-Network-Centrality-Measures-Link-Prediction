# TODO:
# If candidate_edges becomes an iterator in the future,
# this implementation should be updated to avoid
# traversing the candidate edges twice.



"""

Construction of the link prediction dataset.
"""

from src.preprocessing.feature_vectors import (build_feature_vectors)

from src.preprocessing.labels import(build_labels)


def build_dataset(graph_1,graph_2, candidate_edges):
    """
    Construct the final dataset for link prediction.

    Yields
    ------
    tuple
        (u, v, SP, CN, JC, AA, PA, label)
    """

    feature_vectors = build_feature_vectors(graph_1, candidate_edges)

    labels = build_labels(graph_2,candidate_edges)

    for feature_vector, label in zip(feature_vectors,labels):
        yield (*feature_vector,label)
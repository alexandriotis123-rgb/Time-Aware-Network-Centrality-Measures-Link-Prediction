"""Construction of the link prediction dataset."""

from src.preprocessing.feature_vectors import build_feature_vectors
from src.preprocessing.labels import build_labels


def build_dataset(graph_1, graph_2, candidate_edges):
    """Build a dataset for predicting new links between graph_1 and graph_2.

    The feature vectors are computed from graph_1, while labels are derived
    from whether the candidate edge exists in graph_2.
    """

    feature_vectors = build_feature_vectors(graph_1, candidate_edges)
    labels = build_labels(graph_1, graph_2, candidate_edges)

    for feature_vector, label in zip(feature_vectors, labels):
        yield (*feature_vector, label)
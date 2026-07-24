"""Construction of the link prediction dataset."""

from src.preprocessing.feature_vectors import build_feature_vectors
from src.preprocessing.labels import build_labels, build_membership_labels


def build_dataset(
    graph_1,
    graph_2,
    candidate_edges,
    positive_edges=None,
):
    """Build a scored edge-classification dataset.

    Features are always computed from ``graph_1``. When ``positive_edges`` is
    supplied, labels indicate membership in that explicit edge set. This is
    used by the SNA protocol to label ``E_j*`` during training and
    ``E_(j+1)*`` during testing.

    If ``positive_edges`` is omitted, legacy labels for edges present in
    ``graph_2`` but absent from ``graph_1`` are used.
    """

    feature_vectors = build_feature_vectors(graph_1, candidate_edges)
    if positive_edges is None:
        labels = build_labels(graph_1, graph_2, candidate_edges)
    else:
        labels = build_membership_labels(positive_edges, candidate_edges)

    for feature_vector, label in zip(feature_vectors, labels):
        yield (*feature_vector, label)

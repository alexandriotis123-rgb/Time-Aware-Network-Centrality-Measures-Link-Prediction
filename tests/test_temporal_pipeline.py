import networkx as nx

from src.analysis.similarity import shortest_path_similarity
from src.preprocessing.balanced_dataset import split_candidate_edges_stratified
from src.preprocessing.candidate_edges import build_candidate_edges
from src.preprocessing.dataset import build_dataset
from src.preprocessing.labels import build_labels
from src.prediction.evaluation import compute_balanced_accuracy, compute_lambda


def test_shortest_path_similarity_uses_negative_infinity_for_disconnected_nodes():
    graph = nx.path_graph(3)

    assert shortest_path_similarity(graph, 0, 2) == -2
    assert shortest_path_similarity(graph, 0, 10) == float("-inf")


def test_build_candidate_edges_canonicalizes_edges_and_excludes_existing_edges():
    graph = nx.Graph()
    graph.add_edge(2, 1)
    graph.add_edge(3, 4)

    candidate_edges = build_candidate_edges([1, 2, 3, 4], existing_edges=graph.edges())

    assert (1, 2) not in candidate_edges
    assert (3, 4) not in candidate_edges
    assert (1, 3) in candidate_edges
    assert all(isinstance(edge, tuple) and edge == tuple(sorted(edge)) for edge in candidate_edges)


def test_build_dataset_uses_new_links_as_positive_labels():
    graph_1 = nx.Graph()
    graph_1.add_edge(1, 2)

    graph_2 = nx.Graph()
    graph_2.add_edge(1, 2)
    graph_2.add_edge(2, 3)

    candidate_edges = [(1, 2), (2, 3)]

    dataset = list(build_dataset(graph_1, graph_2, candidate_edges))
    labels = [row[-1] for row in dataset]

    assert labels == [0, 1]


def test_build_candidate_edges_respects_max_candidates_and_excludes_existing_edges():
    graph = nx.Graph()
    graph.add_edge(1, 2)
    graph.add_edge(3, 4)

    candidate_edges = build_candidate_edges(
        [1, 2, 3, 4],
        existing_edges=graph.edges(),
        max_candidates=2,
        seed=42,
    )

    assert len(candidate_edges) == 2
    assert (1, 2) not in candidate_edges
    assert (3, 4) not in candidate_edges
    assert all(edge == tuple(sorted(edge)) for edge in candidate_edges)


def test_compute_lambda_uses_ground_truth_inside_candidate_subset():
    ground_truth_edges = {(1, 2), (2, 3)}
    candidate_edges = {(1, 2)}

    assert compute_lambda(ground_truth_edges, candidate_edges) == 1.0


def test_compute_balanced_accuracy_averages_tpr_and_tnr():
    predicted_edges = {(1, 2)}
    ground_truth_edges = {(1, 2), (2, 3)}
    candidate_edges = {(1, 2), (2, 3)}

    assert compute_balanced_accuracy(predicted_edges, ground_truth_edges, candidate_edges) == 0.25


def test_split_candidate_edges_stratified_keeps_both_classes_in_validation():
    candidate_edges = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    ground_truth_edges = {(1, 2), (1, 3)}

    train_candidates, val_candidates = split_candidate_edges_stratified(
        candidate_edges,
        ground_truth_edges,
        val_ratio=0.5,
        seed=7,
    )

    assert any(edge in ground_truth_edges for edge in val_candidates)
    assert any(edge not in ground_truth_edges for edge in val_candidates)
    assert any(edge in ground_truth_edges for edge in train_candidates)
    assert any(edge not in ground_truth_edges for edge in train_candidates)

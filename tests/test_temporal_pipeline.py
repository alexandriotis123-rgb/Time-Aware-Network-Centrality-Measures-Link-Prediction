import networkx as nx
import pandas as pd

from src.analysis.centrality import (
    build_centrality_distributions,
    compute_betweenness_centrality,
    compute_eigenvector_centrality,
    compute_katz_centrality,
    consecutive_kl_divergences,
    kl_divergence,)
from src.analysis.similarity import (
    adamic_adar_similarity,
    shortest_path_similarity,)
from src.preprocessing.balanced_dataset import split_candidate_edges_stratified
from src.preprocessing.candidate_edges import build_candidate_edges
from src.preprocessing.dataset import build_dataset
from src.preprocessing.labels import build_labels
from src.prediction.experiment import build_sna_edge_partitions
from src.prediction.evaluation import (
    compute_accuracy,
    compute_balanced_accuracy,
    compute_lambda,)
from src.prediction.training import improve_range_set
from src.preprocessing.temporal_partition import partition_temporal_network


def test_shortest_path_similarity_uses_reciprocal_distance_and_zero_when_disconnected():
    graph = nx.path_graph(3)

    assert shortest_path_similarity(graph, 0, 1) == 1.0
    assert shortest_path_similarity(graph, 0, 2) == 0.5
    assert shortest_path_similarity(graph, 0, 10) == 0.0


def test_shortest_path_similarity_ignores_existing_direct_edge_when_requested():
    graph = nx.Graph([(1, 2), (1, 3), (3, 2)])
    edges_before = set(graph.edges())

    score = shortest_path_similarity(graph, 1, 2,ignore_direct_edge=True)

    assert score == 0.5
    assert set(graph.edges()) == edges_before


def test_adamic_adar_uses_base_two_logarithm():
    graph = nx.Graph([(1, 3), (2, 3), (3, 4), (3, 5)])

    assert adamic_adar_similarity(graph, 1, 2) == 0.5


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
        seed=42)

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


def test_accuracy_ignores_ground_truth_outside_candidate_population():
    predicted_edges = {(1, 2)}
    ground_truth_edges = {(1, 2), (9, 10)}
    candidate_edges = {(1, 2), (1, 3)}

    assert compute_accuracy(
        predicted_edges,
        ground_truth_edges,
        candidate_edges,
    ) == 1.0


def test_split_candidate_edges_stratified_keeps_both_classes_in_validation():
    candidate_edges = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    ground_truth_edges = {(1, 2), (1, 3)}

    train_candidates, val_candidates = split_candidate_edges_stratified(
        candidate_edges,
        ground_truth_edges,
        val_ratio=0.5,
        seed=7)

    assert any(edge in ground_truth_edges for edge in val_candidates)
    assert any(edge not in ground_truth_edges for edge in val_candidates)
    assert any(edge in ground_truth_edges for edge in train_candidates)
    assert any(edge not in ground_truth_edges for edge in train_candidates)


def test_build_dataset_accepts_explicit_sna_positive_edges():
    feature_graph = nx.Graph()
    feature_graph.add_nodes_from([1, 2, 3])
    feature_graph.add_edge(1, 2)

    candidate_edges = [(1, 2), (1, 3)]
    dataset = list(build_dataset(
        feature_graph,
        feature_graph,
        candidate_edges,
        positive_edges={(1, 2)}))

    assert [row[-1] for row in dataset] == [1, 0]


def test_sna_edge_partitions_are_balanced_temporal_and_disjoint():
    graph_1 = nx.Graph()
    graph_2 = nx.Graph()
    graph_1.add_nodes_from(range(1, 6))
    graph_2.add_nodes_from(range(1, 6))
    graph_1.add_edges_from([(1, 2), (2, 3)])
    graph_2.add_edges_from([(2, 3), (3, 4)])

    partitions = build_sna_edge_partitions(
        graph_1,
        graph_2,
        persistent_nodes=set(range(1, 6)),
        seed=11,)

    assert partitions["train_positive"] == {(1, 2), (2, 3)}
    assert partitions["test_positive"] == {(2, 3), (3, 4)}
    assert len(partitions["train_negative"]) == 2
    assert len(partitions["test_negative"]) == 2
    assert partitions["train_negative"].isdisjoint(
        partitions["test_negative"])
    assert partitions["train_negative"].isdisjoint(
        partitions["train_positive"] | partitions["test_positive"])
    assert partitions["test_negative"].isdisjoint(
        partitions["train_positive"] | partitions["test_positive"])


def test_temporal_partition_removes_self_links():
    dataframe = pd.DataFrame(
        [
            (1, 1, 0),
            (1, 2, 0),
            (2, 3, 10),
        ],
        columns=["source", "target", "timestamp"],)

    _, _, subgraphs = partition_temporal_network(dataframe, num_periods=2)

    assert all(nx.number_of_selfloops(graph) == 0 for graph in subgraphs)
    assert subgraphs[0].has_edge(1, 2)


def test_sna_edge_partitions_exclude_self_links():
    graph_1 = nx.Graph([(1, 1), (1, 2)])
    graph_2 = nx.Graph([(2, 2), (2, 3)])

    partitions = build_sna_edge_partitions(
        graph_1,
        graph_2,
        persistent_nodes={1, 2, 3, 4},
        seed=4,)

    assert all(
        node_u != node_v
        for key in ("train_positive", "test_positive")
        for node_u, node_v in partitions[key])


def test_range_improvement_never_exceeds_interval_limit():
    dataset = [
        (1, 2, 0.1, 0, 0, 0, 0, 1),
        (1, 3, 0.2, 0, 0, 0, 0, 0),
        (1, 4, 0.3, 0, 0, 0, 0, 1),
        (1, 5, 0.4, 0, 0, 0, 0, 0)]
    candidates = [(row[0], row[1]) for row in dataset]
    positives = {(1, 2), (1, 4)}
    existing_ranges = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]
    candidate_ranges = [[(0.4, 0.4)]]

    ranges, _, _, _ = improve_range_set(
        dataset,
        candidates,
        positives,
        "GD",
        existing_ranges,
        0.0,
        0.0,
        0.0,
        candidate_ranges,
        max_intervals=3,)

    assert ranges == existing_ranges
    assert len(ranges) == 3


def test_centrality_distributions_use_shared_bins_and_smoothed_kl():
    _, distributions = build_centrality_distributions(
        [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
        bins=3,)

    assert all(abs(distribution.sum() - 1.0) < 1e-12 for distribution in distributions)
    assert kl_divergence(distributions[0], distributions[0]) == 0.0
    assert consecutive_kl_divergences(distributions) == [0.0]


def test_centrality_helpers_handle_empty_and_small_graphs():
    empty_graph = nx.Graph()
    small_graph = nx.path_graph(3)

    assert compute_betweenness_centrality(empty_graph) == []
    assert compute_eigenvector_centrality(empty_graph) == []
    assert compute_katz_centrality(empty_graph) == []
    assert len(compute_betweenness_centrality(small_graph, k=500)) == 3


def test_capped_candidate_builder_handles_too_few_nodes():
    assert build_candidate_edges([], max_candidates=5) == []
    assert build_candidate_edges([1], max_candidates=5) == []

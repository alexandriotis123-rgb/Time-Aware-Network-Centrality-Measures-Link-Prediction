import networkx as nx

from src.analysis.similarity import shortest_path_similarity
from src.preprocessing.candidate_edges import build_candidate_edges
from src.preprocessing.dataset import build_dataset
from src.preprocessing.labels import build_labels


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

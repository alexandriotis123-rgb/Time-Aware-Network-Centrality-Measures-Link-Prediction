"""
Similarity measures between pairs of nodes.
"""

import math
import networkx as nx


def shortest_path_similarity(graph, node_u, node_v, ignore_direct_edge=False):
    """Return the SNA shortest-path similarity ``1 / distance``.

    Disconnected pairs receive zero, as specified in ``SNA.pdf``.
    """

    if node_u == node_v:
        return 1.0

    if node_u not in graph or node_v not in graph:
        return 0.0

    if ignore_direct_edge and graph.has_edge(node_u, node_v):
        graph.remove_edge(node_u, node_v)
        try:
            distance = nx.shortest_path_length(graph, source=node_u, target=node_v)
            return 1.0 / distance
        except nx.NetworkXNoPath:
            return 0.0
        finally:
            graph.add_edge(node_u, node_v)

    try:
        distance = nx.shortest_path_length(graph, source=node_u, target=node_v)
        return 1.0 / distance
    except nx.NetworkXNoPath:
        return 0.0
    
def common_neighbors_similarity(graph, node_u, node_v):
    """Compute the Common Neighbors similarity between two nodes."""

    if node_u not in graph or node_v not in graph:
        return 0

    neighbors_u = set(graph.neighbors(node_u))
    neighbors_v = set(graph.neighbors(node_v))

    common_neighbors = neighbors_u.intersection(neighbors_v)

    return len(common_neighbors)


def jaccard_similarity(graph, node_u, node_v):
    """Compute the Jaccard Similarity between two nodes."""

    if node_u not in graph or node_v not in graph:
        return 0.0

    neighbors_u = set(graph.neighbors(node_u))
    neighbors_v = set(graph.neighbors(node_v))

    intersection = neighbors_u.intersection(neighbors_v)
    union = neighbors_u.union(neighbors_v)

    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)

def adamic_adar_similarity(graph, node_u, node_v):
    """Compute the Adamic-Adar Similarity between two nodes."""

    if node_u not in graph or node_v not in graph:
        return 0.0

    neighbors_u = set(graph.neighbors(node_u))
    neighbors_v = set(graph.neighbors(node_v))

    common_neighbors = neighbors_u.intersection(neighbors_v)

    similarity = 0.0

    for neighbor in common_neighbors:
        degree = graph.degree(neighbor)

        if degree > 1:
            similarity += 1 / math.log2(degree)

    return similarity


def preferential_attachment_similarity(graph, node_u, node_v):
    """Compute the Preferential Attachment similarity between two nodes."""

    if node_u not in graph or node_v not in graph:
        return 0

    degree_u = graph.degree(node_u)
    degree_v = graph.degree(node_v)

    return degree_u * degree_v
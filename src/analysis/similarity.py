"""
Similarity measures between pairs of nodes.
"""

import networkx as nx
import math

def shortest_path_similarity(graph, node_u, node_v):

    if node_u == node_v:
        return 0

    try:

        distance = nx.shortest_path_length(
            graph,
            source=node_u,
            target=node_v)

        return -distance

    except nx.NetworkXNoPath:

        return 0.0
    
def common_neighbors_similarity(
    graph,
    node_u,
    node_v):
    """
    Compute the Common Neighbors similarity between two nodes.
    """

    neighbors_u = set(graph.neighbors(node_u))

    neighbors_v = set(graph.neighbors(node_v))

    common_neighbors = neighbors_u.intersection(neighbors_v)

    return len(common_neighbors)


def jaccard_similarity( graph, node_u,node_v):
    """
    Compute the Jaccard Similarity
    between two nodes.
    """


    neighbors_u = set( graph.neighbors(node_u))

    neighbors_v = set(graph.neighbors(node_v))

    intersection = neighbors_u.intersection( neighbors_v )

    union =neighbors_u.union( neighbors_v )

    if len(union) == 0:
        return 0.0

    return (len(intersection) / len(union))

def adamic_adar_similarity( graph,node_u,node_v):
    """
    Compute the Adamic-Adar Similarity
    between two nodes.
    """

    neighbors_u = set(graph.neighbors(node_u))

    neighbors_v = set(graph.neighbors(node_v))

    common_neighbors = neighbors_u.intersection( neighbors_v)

    similarity = 0.0

    for neighbor in common_neighbors:

        degree = graph.degree(neighbor)

        # Avoid division by zero or log(1)
        if degree > 1:

            similarity += 1 / math.log(degree)

    return similarity


def preferential_attachment_similarity( graph,node_u,node_v):
    """
    Compute the Preferential Attachment similarity
    between two nodes.
    """

    degree_u = graph.degree(node_u)

    degree_v = graph.degree(node_v)

    return degree_u * degree_v
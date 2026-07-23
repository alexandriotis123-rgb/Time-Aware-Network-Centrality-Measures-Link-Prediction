"""
PART II - Question 1(a)

Functions for identifying persistent nodes
between consecutive temporal graphs.
"""

import networkx as nx


def find_persistent_nodes(
    graph_1,
    graph_2
):
    """
    Find the nodes that persist between two
    consecutive temporal graphs.

    Parameters
    ----------
    graph_1 : networkx.Graph

    graph_2 : networkx.Graph

    Returns
    -------
    set
        Nodes appearing in both graphs.
    """

    # Nodes of the first temporal graph
    nodes_1 = set(graph_1.nodes())

    # Nodes of the second temporal graph
    nodes_2 = set(graph_2.nodes())

    # Persistent nodes
    persistent_nodes = nodes_1.intersection(nodes_2)

    return persistent_nodes


def restrict_graph(
    graph,
    persistent_nodes
):
    """
    Restrict a temporal graph to the set
    of persistent nodes.

    Parameters
    ----------
    graph : networkx.Graph

    persistent_nodes : set
        Nodes that persist between two
        consecutive temporal graphs.

    Returns
    -------
    networkx.Graph
        Restricted graph containing only
        persistent nodes and the edges
        between them.
    """

    # Create the restricted graph induced
    # by the persistent nodes
    restricted_graph = graph.subgraph(
        persistent_nodes
    ).copy()

    return restricted_graph

def build_persistent_pairs(subgraphs):
    """
    Build restricted graph pairs for every
    pair of consecutive temporal graphs.

    """
    persistent_pairs = []
    persistent_node_sets = []

    for i in range(len(subgraphs) - 1):

        graph_1 = subgraphs[i]

        graph_2 = subgraphs[i + 1]

        persistent_nodes = find_persistent_nodes(
            graph_1,
            graph_2
        )

        persistent_node_sets.append(persistent_nodes)

        restricted_graph_1 = restrict_graph(
            graph_1,
            persistent_nodes
        )

        restricted_graph_2 = restrict_graph(
            graph_2,
            persistent_nodes
        )

        persistent_pairs.append(
            (
                restricted_graph_1,
                restricted_graph_2
            )
        )

    return ( persistent_pairs, persistent_node_sets)
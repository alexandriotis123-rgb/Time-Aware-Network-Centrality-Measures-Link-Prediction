"""

PART I - Question 4
Centrality analysis functions.
"""

import networkx as nx
from config import RANDOM_SEED


def compute_degree_centrality(graph):
    """
    Compute Degree Centrality for every node
    of a temporal graph.

    Parameters
    ----------
    graph : networkx.Graph

    Returns
    -------
    list
        Degree centrality values.
    """

    degree = nx.degree_centrality(graph)

    return list(degree.values())

def compute_closeness_centrality(graph):
    """
    Compute Closeness Centrality for every node
    of a temporal graph.

    Parameters
    ----------
    graph : networkx.Graph

    Returns
    -------
    list
        Closeness centrality values.
    """

    closeness = nx.closeness_centrality(graph)

    return list(closeness.values())

def compute_betweenness_centrality(
    graph,
    k=500
):
    
    # Set k=None for the exact Brandes algorithm.
    # Use k=500 (or another suitable value) for a faster approximation
    # during development or for very large temporal graphs.
    """
    Compute Betweenness Centrality for every node
    of a temporal graph.

    Parameters
    ----------
    graph : networkx.Graph

    k : int or None, optional
        Number of sampled nodes used for approximation.
        If None, the exact Brandes algorithm is executed.

    Returns
    -------
    list
        Betweenness centrality values.
    """

    # Exact computation (k=None).
    # If execution becomes prohibitively slow for large snapshots,
    # use k=500 (or another suitable value)
    # to obtain a faster approximate solution.

    betweenness = nx.betweenness_centrality(
        graph,
        k=k,
        normalized=True,
        weight=None,
        endpoints=False,
        seed=RANDOM_SEED
    )

    return list(betweenness.values())


def compute_eigenvector_centrality(graph):
    """
    Compute Eigenvector Centrality for every node
    of a temporal graph.

    Parameters
    ----------
    graph : networkx.Graph

    Returns
    -------
    list
        Eigenvector centrality values.
    """

    eigenvector = nx.eigenvector_centrality(
        graph,
        max_iter=1000,
        tol=1e-06
    )

    return list(eigenvector.values())


def compute_katz_centrality(graph,alpha=0.005):
    """
    Compute Katz Centrality for every node
    of a temporal graph.

    Parameters
    ----------
    graph : networkx.Graph

    alpha : float, optional
        Attenuation factor.
        It can be adjusted depending on the graph size.

    Returns
    -------
    list
        Katz centrality values.
    """

    # Alpha is exposed as a parameter so it can be tuned if needed.
    # Smaller values improve convergence on large temporal graphs.

    katz = nx.katz_centrality(
        graph,
        alpha=alpha,
        beta=1.0,
        max_iter=1000,
        tol=1e-06
    )

    return list(katz.values())


def run_centrality_analysis(
    subgraphs,
    centrality_function,
    centrality_name,
    plot_function,
    log_scale=False
):
    """
    Compute and plot a centrality measure
    for every temporal graph.
    """

    print(f"\nComputing {centrality_name}...")

    for i, graph in enumerate(subgraphs, start=1):

        values = centrality_function(graph)

        plot_function(
            values=values,
            title=f"{centrality_name} - T{i}",
            filename=f"{centrality_name.lower().replace(' ', '_')}_T{i}.png",
            log_scale=log_scale)

        print(f"T{i} completed.")
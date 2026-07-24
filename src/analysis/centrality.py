"""

PART I - Question 4
Centrality analysis functions.
"""

import csv
from pathlib import Path

import networkx as nx
import numpy as np

from config import RANDOM_SEED, RESULTS_FOLDER


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

    if graph.number_of_nodes() == 0:
        return []

    sample_size = (
        min(k, graph.number_of_nodes())
        if k is not None
        else None
    )

    betweenness = nx.betweenness_centrality(
        graph,
        k=sample_size,
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

    if graph.number_of_nodes() == 0:
        return []

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

    if graph.number_of_nodes() == 0:
        return []

    katz = nx.katz_centrality(
        graph,
        alpha=alpha,
        beta=1.0,
        max_iter=1000,
        tol=1e-06
    )

    return list(katz.values())


def build_centrality_distributions(values_by_period, bins=30):
    """Build normalized histograms over one shared centrality support."""

    finite_values = [
        value
        for values in values_by_period
        for value in values
        if np.isfinite(value)
    ]

    if not finite_values:
        raise ValueError("Centrality distributions require finite values.")

    bin_edges = np.histogram_bin_edges(finite_values, bins=bins)
    distributions = []

    for values in values_by_period:
        counts, _ = np.histogram(values, bins=bin_edges)
        total = counts.sum()
        if total == 0:
            distribution = np.zeros_like(counts, dtype=float)
        else:
            distribution = counts.astype(float) / total
        distributions.append(distribution)

    return bin_edges, distributions


def kl_divergence(distribution_p, distribution_q, epsilon=1e-4):
    

    p = np.asarray(distribution_p, dtype=float) + epsilon
    q = np.asarray(distribution_q, dtype=float) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def consecutive_kl_divergences(distributions, epsilon=1e-4):
    """Compute KL divergence for every consecutive temporal pair."""

    return [
        kl_divergence(
            distributions[index],
            distributions[index + 1],
            epsilon=epsilon,
        )
        for index in range(len(distributions) - 1)
    ]


def save_kl_results(centrality_name, kl_values):
    """Persist consecutive KL values as a compact CSV result."""

    results_path = Path(RESULTS_FOLDER)
    results_path.mkdir(parents=True, exist_ok=True)
    filename = (
        centrality_name.lower().replace(" ", "_")
        + "_kl_divergence.csv"
    )

    with (results_path / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["period_from", "period_to", "kl_divergence"])
        for index, value in enumerate(kl_values, start=1):
            writer.writerow([index, index + 1, value])


def run_centrality_analysis(
    subgraphs,
    centrality_function,
    centrality_name,
    plot_function,
    log_scale=False,
    kl_plot_function=None,
    bins=30,
    epsilon=1e-4,
):
    """
    Compute and plot a centrality measure
    for every temporal graph.
    """

    print(f"\nComputing {centrality_name}...")

    values_by_period = [
        centrality_function(graph)
        for graph in subgraphs
    ]
    bin_edges, distributions = build_centrality_distributions(
        values_by_period,
        bins=bins,
    )

    for i, values in enumerate(values_by_period, start=1):

        plot_function(
            values=values,
            title=f"{centrality_name} - T{i}",
            filename=f"{centrality_name.lower().replace(' ', '_')}_T{i}.png",
            log_scale=log_scale,
            bins=bin_edges,
        )

        print(f"T{i} completed.")

    kl_values = consecutive_kl_divergences(
        distributions,
        epsilon=epsilon,
    )
    save_kl_results(centrality_name, kl_values)

    if kl_plot_function is not None:
        kl_plot_function(kl_values, centrality_name)

    print(f"{centrality_name} consecutive KL divergences:")
    for index, value in enumerate(kl_values, start=1):
        print(f"T{index} -> T{index + 1}: {value:.6f}")

    return {
        "values": values_by_period,
        "bin_edges": bin_edges,
        "distributions": distributions,
        "kl_divergences": kl_values,
    }

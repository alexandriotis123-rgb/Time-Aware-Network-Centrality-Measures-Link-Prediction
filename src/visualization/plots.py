"""
PART I - Question 2
Visualization functions.
"""

import matplotlib.pyplot as plt


def plot_network_evolution(subgraphs):
    """
    Plot the evolution of the number of nodes and edges
    across all temporal network snapshots.
    """

    node_counts = []
    edge_counts = []

    for graph in subgraphs:
        node_counts.append(graph.number_of_nodes())
        edge_counts.append(graph.number_of_edges())

    periods = list(range(1, len(subgraphs) + 1))

    plt.figure(figsize=(10, 8))

    # -------------------------
    # Number of Nodes
    # -------------------------
    plt.subplot(2, 1, 1)

    plt.plot(
        periods,
        node_counts,
        marker="o",
        linewidth=2
    )

    plt.title("Temporal Evolution of |V|")

    plt.xlabel("Time Period")

    plt.ylabel("|V|")

    plt.xticks(periods)

    plt.grid(True)

    # -------------------------
    # Number of Edges
    # -------------------------
    plt.subplot(2, 1, 2)

    plt.plot(
        periods,
        edge_counts,
        marker="s",
        linewidth=2
    )

    plt.title("Temporal Evolution of |E|")

    plt.xlabel("Time Period")

    plt.ylabel("|E|")

    plt.xticks(periods)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("outputs/figures/network_evolution.png")

    plt.show()


def plot_centrality_histogram(
    values,
    title,
    filename,
    log_scale=False):
    """
    Plot the probability density histogram
    of a centrality measure.
    """

    plt.figure(figsize=(8, 5))

    plt.hist(
        values,
        bins=30,
        density=True,
        edgecolor="black"
    )
    if log_scale:
        plt.yscale("log")

    plt.title(title)

    plt.xlabel("Centrality Value")

    if log_scale:
        plt.ylabel("Probability Density (log scale)")
    else:
        plt.ylabel("Probability Density")

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(f"outputs/figures/{filename}")

    plt.show()


def plot_persistent_graph_statistics(
    persistent_nodes,
    persistent_edges_1,
    persistent_edges_2
):
    """
Plot the evolution of the persistent
node and edge volumes between
consecutive temporal graphs.
"""

    plt.figure(figsize=(9,5))

    pairs = range(1, len(persistent_nodes)+1)

    plt.plot(
        pairs,
        persistent_nodes,
        marker="o",
        label="|V*|"
    )

    plt.plot(
        pairs,
        persistent_edges_1,
        marker="s",
        label="|E1*|"
    )

    plt.plot(
        pairs,
        persistent_edges_2,
        marker="^",
        label="|E2*|"
    )

    plt.xlabel("Temporal Pair")

    plt.ylabel("Number of Nodes / Edges")

    plt.title("Persistent Node and Edge Volumes")

    plt.xticks(pairs)

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/persistent_graph_statistics.png"
    )

    plt.show()
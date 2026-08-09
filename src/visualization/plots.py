"""
PART I - Question 2
Visualization functions.
"""

from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

from config import FIGURE_DPI, FIGURES_FOLDER, SAVE_FIGURES


def _finish_figure(filename):

    if SAVE_FIGURES:
        output_path = Path(FIGURES_FOLDER) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=FIGURE_DPI)

    plt.close()


def plot_network_evolution(subgraphs):

    node_counts = []
    edge_counts = []

    for graph in subgraphs:
        node_counts.append(graph.number_of_nodes())
        edge_counts.append(graph.number_of_edges())

    periods = list(range(1, len(subgraphs) + 1))

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)

    plt.plot(periods,node_counts,marker="o", linewidth=2)

    plt.title("Temporal Evolution of |V|")

    plt.xlabel("Time Period")

    plt.ylabel("|V|")

    plt.xticks(periods)

    plt.grid(True)

    plt.subplot(2, 1, 2)

    plt.plot(periods,edge_counts,marker="s", linewidth=2)

    plt.title("Temporal Evolution of |E|")

    plt.xlabel("Time Period")

    plt.ylabel("|E|")

    plt.xticks(periods)

    plt.grid(True)

    plt.tight_layout()

    _finish_figure("network_evolution.png")


def plot_centrality_histogram(
    values,
    title,
    filename,
    log_scale=False,
    bins=30,):

    plt.figure(figsize=(8, 5))

    plt.hist(
        values,
        bins=bins,
        density=True,
        edgecolor="black")
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

    _finish_figure(filename)


def plot_kl_divergence(kl_values, centrality_name):

    pairs = list(range(1, len(kl_values) + 1))

    plt.figure(figsize=(9, 5))
    plt.plot(pairs, kl_values, marker="o", linewidth=2)
    plt.xlabel("Temporal Pair (Tj, Tj+1)")
    plt.ylabel("KL Divergence")
    plt.title(f"{centrality_name}: Consecutive Distribution KL Divergence")
    plt.xticks(pairs)
    plt.grid(True)
    plt.tight_layout()

    filename = (
        centrality_name.lower().replace(" ", "_")
        + "_kl_divergence.png")
    _finish_figure(filename)


def plot_persistent_graph_statistics(
    persistent_nodes,
    persistent_edges_1,
    persistent_edges_2):

    plt.figure(figsize=(9,5))

    pairs = range(1, len(persistent_nodes)+1)

    plt.plot(
        pairs,
        persistent_nodes,
        marker="o",
        label="|V*|")

    plt.plot(pairs, persistent_edges_1, marker="s", label="|E1*|")

    plt.plot( pairs, persistent_edges_2, marker="^", label="|E2*|")

    plt.xlabel("Temporal Pair")

    plt.ylabel("Number of Nodes / Edges")

    plt.title("Persistent Node and Edge Volumes")

    plt.xticks(pairs)

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    _finish_figure("persistent_graph_statistics.png")


def plot_accuracy_comparison(train_accs=[71.88, 66.78, 94.75, 94.77, 94.45], test_accs=[67.32, 65.21, 58.84, 58.81, 58.80], measures=None):
    """
    Plot the comparison of Train and Test Balanced Accuracy
    across all similarity measures.
    """

    if measures is None:
        measures = ['PA', 'GD', 'JC', 'AA', 'CN']

    x = np.arange(len(measures))
    width = 0.35

    plt.figure(figsize=(9, 5.5))

    rects1 = plt.bar(
        x - width / 2,
        train_accs,
        width,
        label='Train Balanced Accuracy (%)',
        color='#4C72B0',
        edgecolor='black',
        alpha=0.85,)
    rects2 = plt.bar(
        x + width / 2,
        test_accs,
        width,
        label='Test Balanced Accuracy (%)',
        color='#55A868',
        edgecolor='black',
        alpha=0.85,)

    plt.ylabel('Balanced Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title(
        'Link Prediction Performance Comparison Across Similarity Measures',
        fontsize=14,
        fontweight='bold',
        pad=15,)
    plt.xticks(
        x,
        ['PA', 'GD', 'JC', 'AA', 'CN'],
        fontsize=11,
        fontweight='bold',)
    plt.legend(
        frameon=True,
        facecolor='white',
        edgecolor='gray',
        fontsize=11,
        loc='upper right',)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        plt.annotate(
            f'{height:.2f}%',
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=9.5,
            fontweight='bold',)

    plt.tight_layout()
    if SAVE_FIGURES:
        output_path1 = Path(FIGURES_FOLDER) / "accuracy_comparison.png"
        output_path2 = Path(FIGURES_FOLDER) / "link_prediction_comparison.png"
        output_path1.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path1, dpi=FIGURE_DPI)
        plt.savefig(output_path2, dpi=FIGURE_DPI)
    plt.close()



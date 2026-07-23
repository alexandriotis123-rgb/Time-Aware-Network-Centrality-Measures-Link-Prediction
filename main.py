from config import (DATASET_FILE,
                    NUMBER_OF_PERIODS,
                    RUN,
                    RUN_CANDIDATE_EDGES,
                    RUN_FEATURE_VECTORS,
                    RUN_DATASET,
                    RUN_TRAINING_EXPERIMENT)
from src.io.data_loader import load_dataset
from src.preprocessing.temporal_partition import partition_temporal_network
from src.visualization.plots import plot_network_evolution
from src.analysis.centrality import (
    compute_degree_centrality,
    compute_closeness_centrality,
    compute_betweenness_centrality,
    compute_eigenvector_centrality,
    compute_katz_centrality,
    run_centrality_analysis)
from src.visualization.plots import plot_centrality_histogram
from src.preprocessing.persistent_nodes import build_persistent_pairs
from src.analysis.graph_statistics import (compute_persistent_graph_statistics)
from src.visualization.plots import plot_persistent_graph_statistics
from src.preprocessing.candidate_edges import build_candidate_edges
from src.preprocessing.feature_vectors import build_feature_vectors
from src.preprocessing.dataset import build_dataset
from src.prediction.experiment import run_training_experiment



def main():

    df = load_dataset(DATASET_FILE)

    time_points, time_periods, subgraphs = partition_temporal_network(
        df=df,
        num_periods=NUMBER_OF_PERIODS)
    
    persistent_pairs, persistent_node_sets = build_persistent_pairs(subgraphs)
    
    persistent_nodes, persistent_edges_1, persistent_edges_2 = (
    compute_persistent_graph_statistics(persistent_pairs))
    print("\nPersistent graph statistics:\n")


    plot_persistent_graph_statistics(
        persistent_nodes,
        persistent_edges_1,
        persistent_edges_2
    )

    for i in range(len(persistent_nodes)):

        print(
            f"Pair {i+1}: "
            f"|V*| = {persistent_nodes[i]}, "
            f"|E1*| = {persistent_edges_1[i]}, "
            f"|E2*| = {persistent_edges_2[i]}"
        )
    
    print(f"\nPersistent graph pairs: {len(persistent_pairs)}")

    for i, (graph_1, graph_2) in enumerate(persistent_pairs, start=1):

        print(
            f"Pair {i}: "
            f"G{i}* -> {graph_1.number_of_nodes()} nodes, "
            f"{graph_1.number_of_edges()} edges | "
            f"G{i+1}* -> {graph_2.number_of_nodes()} nodes, "
            f"{graph_2.number_of_edges()} edges"
        )

    print("\nSubgraphs created:", len(subgraphs))

    for i, graph in enumerate(subgraphs, start=1):
        print(f"T{i}: "f"{graph.number_of_nodes()} nodes, "f"{graph.number_of_edges()} edges")
    if RUN["network_evolution"]:
        plot_network_evolution(subgraphs)

    if RUN["degree"]:

        run_centrality_analysis(
            subgraphs=subgraphs,
            centrality_function=compute_degree_centrality,
            centrality_name="Degree Centrality",
            plot_function=plot_centrality_histogram,
            log_scale=True
        )

    if RUN["closeness"]:

        run_centrality_analysis(
            subgraphs=subgraphs,
            centrality_function=compute_closeness_centrality,
            centrality_name="Closeness Centrality",
            plot_function=plot_centrality_histogram,
            log_scale=False
        )


    if RUN["betweenness"]:

        run_centrality_analysis(
            subgraphs=subgraphs,
            centrality_function=compute_betweenness_centrality,
            centrality_name="Betweenness Centrality",
            plot_function=plot_centrality_histogram,
            log_scale=True)


    if RUN["eigenvector"]:

        run_centrality_analysis(
            subgraphs=subgraphs,
            centrality_function=compute_eigenvector_centrality,
            centrality_name="Eigenvector Centrality",
            plot_function=plot_centrality_histogram,
            log_scale=True)


    if RUN["katz"]:

        run_centrality_analysis(
            subgraphs=subgraphs,
            centrality_function=compute_katz_centrality,
            centrality_name="Katz Centrality",
            plot_function=plot_centrality_histogram,
            log_scale=True)
        

    if RUN_CANDIDATE_EDGES:

        for i, node_set in enumerate(persistent_node_sets, start=1):

            candidate_edges = build_candidate_edges(node_set)
            expected_edges = len(node_set) * (len(node_set) - 1) // 2

            print(f"\nPair {i}")
            print("-" * 40)
            print(f"Persistent Nodes : {len(node_set)}")
            print(f"Candidate Edges  : {len(candidate_edges)}")
            print(f"Expected Edges  : {expected_edges}")
            print(f"Candidate Edges : {len(candidate_edges)}")

            print("\nFirst 10 candidate edges:")

            for edge in candidate_edges[:10]:
                print(edge)


    if RUN_FEATURE_VECTORS:

        graph_1, graph_2 = persistent_pairs[0]

        candidate_edges = build_candidate_edges(persistent_node_sets[0])

        feature_vectors = build_feature_vectors(graph_1,candidate_edges)

        print("\nFirst 5 Feature Vectors")
        print("-" * 40)

        for i, feature_vector in enumerate(feature_vectors):

            print(feature_vector)

            if i == 4:
                break

    if RUN_DATASET:

        graph_1, graph_2 = persistent_pairs[0]

        candidate_edges = build_candidate_edges(persistent_node_sets[0])

        dataset = build_dataset(graph_1,graph_2,candidate_edges)

        print("\nFirst 5 dataset rows:\n")

        for i, row in enumerate(dataset):
            print(row)

            if i == 4:
                break

    if RUN_TRAINING_EXPERIMENT:

        training_results = run_training_experiment(
            persistent_pairs,
            persistent_node_sets)

        for pair_index, ranking in enumerate(training_results, start=1):

            print(f"\nPersistent Pair {pair_index}")

            for position, (measure, info) in enumerate(ranking, start=1):

                print(
                    f"{position}. {measure} "
                    f"(accuracy = {info['accuracy']:.4f})")
        
if __name__ == "__main__":
    main()
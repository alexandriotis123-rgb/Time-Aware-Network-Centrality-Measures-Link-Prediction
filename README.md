# Temporal Network Analysis 2026

## Project Description

This project implements the assignment of the course Social Network Analysis.

The objective is to analyze the StackOverflow Temporal Network, study its temporal evolution, compute graph centrality measures, perform link prediction using similarity measures and evaluate the prediction performance.

Implementation language:
- Python 3.x

Main Libraries:
- pandas
- numpy
- networkx
- matplotlib
- scipy

---

# Project Structure

```
TEMPORAL_NETWORK_ANALYSIS_2026
│
├── data
├── notebooks
├── outputs
│   ├── figures
│   ├── models
│   └── results
├── report
├── src
├── tests
│
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

---

# Development Pipeline

Dataset

↓

Load Dataset

↓

Temporal Partition

↓

Temporal Graph Construction

↓

Graph Statistics

↓

Centrality Measures

↓

Similarity Measures

↓

Link Prediction

↓

Training

↓

Evaluation

↓

Visualization

---

# Modules Description

## io/data_loader.py

Purpose

Load the StackOverflow temporal dataset from disk.

Main Functions

- **load_dataset(file_path)** – loads the StackOverflow temporal dataset into a pandas DataFrame.

Input

- Dataset path

Output

- pandas DataFrame

---

## preprocessing/temporal_partition.py

Purpose

Partition the complete temporal interval into N equal time periods and construct temporal subgraphs.

Main Functions

- **compute_time_periods()** – calculates temporal partition boundaries.
- **partition_edges()** – splits edges into temporal slices.
- **create_temporal_graphs()** – builds graph objects for each time period.

Input

- DataFrame
- Number of periods N

Output

- List of temporal graphs

---

## analysis/graph_statistics.py

Purpose

Compute basic graph statistics for every temporal graph.

Main Functions

- **number_of_nodes()** – returns the node count for a graph.
- **number_of_edges()** – returns the edge count for a graph.
- **graph_evolution()** – tracks graph changes over time.

Output

- Statistics
- Evolution plots

---

## analysis/centrality.py

Purpose

Compute node centrality measures.

Main Functions

- **compute_degree()** – calculates degree centrality.
- **compute_closeness()** – calculates closeness centrality.
- **compute_betweenness()** – calculates betweenness centrality.
- **compute_eigenvector()** – calculates eigenvector centrality.
- **compute_katz()** – calculates Katz centrality.

Output

- Centrality values
- Histograms

---

## analysis/similarity.py

Purpose

Compute similarity matrices for node pairs.

Main Functions

- **graph_distance_similarity()** – computes distance-based similarity.
- **common_neighbors()** – counts shared neighbors between nodes.
- **jaccard_similarity()** – Jaccard similarity of neighbor sets.
- **adamic_adar()** – Adamic/Adar similarity measure.
- **preferential_attachment()** – preferential attachment similarity.

Output

- Similarity matrices

---

## prediction/link_prediction.py

Purpose

Predict future links using similarity measures.

Main Functions

- generate_candidate_edges()
- predict_links()

Output

- Predicted edge set

---

## prediction/training.py

Purpose

Determine the optimal similarity threshold (RX).

Main Functions

- train_threshold()

Output

- Optimal threshold
- Training accuracy

---

## prediction/evaluation.py

Purpose

Evaluate prediction performance.

Main Functions

- **compute_tpr()** – computes true positive rate.
- **compute_tnr()** – computes true negative rate.
- **compute_accuracy()** – computes overall accuracy.
- **rank_similarity_measures()** – ranks similarity metrics.

Output

- Accuracy
- Ranking

---

## visualization/plots.py

Purpose

Generate every figure required by the assignment.

Main Functions

- **plot_graph_evolution()** – visualizes graph evolution over time.
- **plot_histograms()** – creates histograms of measures.
- **plot_accuracy()** – plots accuracy across thresholds.

Output

- Figures saved in outputs/figures

---

## utils/helpers.py

Purpose

General helper functions shared across the project.

---

# Expected Workflow

1. Load dataset
2. Partition temporal interval
3. Create temporal graphs
4. Compute graph statistics
5. Compute centrality measures
6. Compute similarity measures
7. Perform link prediction
8. Train thresholds
9. Evaluate performance
10. Generate figures

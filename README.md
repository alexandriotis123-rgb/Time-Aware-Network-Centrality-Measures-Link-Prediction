# Temporal Network Analysis 2026

## Project Description

This project implements the assignment of the course Social Network Analysis.

The objective is to analyze the StackOverflow Temporal Network, study its temporal evolution, compute graph centrality measures, perform link prediction using similarity measures and evaluate the prediction performance across consecutive temporal graph snapshots.

Implementation language:
- Python 3.x

Main Libraries:
- pandas
- numpy
- networkx
- matplotlib
- scipy

---


# Dataset

The project uses the Stack Overflow Temporal Network dataset provided by the
Stanford Network Analysis Project (SNAP).

The dataset is not included in the project submission because of its large size.

After downloading the dataset, place the file:

    sx-stackoverflow.txt

inside the following directory:

    data/sx-stackoverflow.txt

The expected dataset format is:

    source target timestamp

where each row represents a temporal interaction between two users.

For the experimental evaluation reported in this project, the implementation
uses the first 500,000 temporal interactions. This is controlled through the
following configuration:

    DEBUG_MODE = True
    DEBUG_MAX_ROWS = 500000

The complete experimental methodology remains unchanged; the reduced dataset
is used to make the experimental pipeline computationally manageable.

---

# Installation

Install the required Python dependencies using:

```bash
pip install -r requirements.txt

# Project Structure

```
TEMPORAL_NETWORK_ANALYSIS_2026
│
├── outputs
│   ├── figures
│   ├── models
│   |── results
|   └──main_output.log
├── src
|   ├──analysis
|   |   ├──centrality
|   |   ├──similarity
|   |   ├──graph_statistics  
|   ├──io
|   |   ├──data_loader.py
|   ├──prediction
|   |   ├──brute_force
|   |   ├──evaluation
|   |   ├──experiment
|   |   ├──link_prediction
|   |   ├──training
|   ├──preprocessing
|   |   ├──balanced_dataset
|   |   ├──candidate_edges
|   |   ├──dataset
|   |   ├──feature_vectors
|   |   ├──labels
|   |   ├──persistent_nodes
|   |   ├──temporal_partition
|   ├──utils
|   |   ├──helpers
|   ├──visualization
|   |   ├──plots
├── config.py
├──conftest.py
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

- **compute_degree_centrality()** – calculates degree centrality.
- **compute_closeness_centrality()** – calculates closeness centrality.
- **compute_betweenness_centrality()** – calculates betweenness centrality.
- **compute_eigenvector_centrality()** – calculates eigenvector centrality.
- **compute_katz_centrality()** – calculates Katz centrality.

Output

- Centrality values
- Histograms

---

## analysis/similarity.py

Purpose

Compute similarity matrices for node pairs.

Main Functions

- **graph_distance_similarity()** – computes distance-based similarity.
- **common_neighbors_similarity()** – counts shared neighbors between nodes.
- **jaccard_similarity()** – Jaccard similarity of neighbor sets.
- **adamic_adar_similarity()** – Adamic/Adar similarity measure.
- **preferential_attachment_similarity()** – preferential attachment similarity.

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

## src/prediction/training.py

Purpose

Train similarity-based link prediction rules by identifying similarity score
ranges that maximize Balanced Accuracy.

Main Functions

- `generate_similarity_ranges()` – generates candidate similarity intervals.
- `find_best_single_range()` – selects the best initial interval.
- `improve_range_set()` – iteratively refines the selected range set.
- `train_similarity_measure()` – executes the complete training procedure.

The training procedure can use multiple non-overlapping similarity intervals,
with the maximum number controlled by `MAX_INTERVALS`.

---

## src/prediction/evaluation.py

Purpose

Compute the performance metrics used to evaluate the link prediction
framework.

Main Functions

- `compute_lambda()` – computes the positive-edge proportion in the candidate
  population.
- `compute_tpr()` – computes the True Positive Rate.
- `compute_tnr()` – computes the True Negative Rate.
- `compute_precision()` – computes Precision.
- `compute_recall()` – computes Recall.
- `compute_accuracy()` – computes Accuracy.
- `compute_balanced_accuracy()` – computes Balanced Accuracy.

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

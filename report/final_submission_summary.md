# Final implementation summary

The implementation now follows the protocol in `SNA.pdf`.

## Temporal centrality analysis

- The dataset is divided into ten non-overlapping temporal snapshots.
- Degree, closeness, approximate betweenness, eigenvector, and Katz
  centralities are computed for every snapshot.
- Every centrality uses one shared histogram support across all periods.
- Consecutive distributions are compared with smoothed
  \(D_{KL}(P_j \parallel P_{j+1})\), using \(\epsilon=10^{-4}\).
- Histograms, KL plots, and KL CSV files are written under `outputs/`.

## Link-prediction protocol

For each persistent pair \(G_j^*, G_{j+1}^*\):

1. Positive training examples are all edges in \(E_j^*\).
2. Positive test examples are all edges in \(E_{j+1}^*\).
3. Negative examples are node pairs absent from
   \(E_j^* \cup E_{j+1}^*\).
4. Training and test negatives are balanced against their respective
   positives and are mutually disjoint.
5. Train/validation splitting is stratified and uses only the training
   population.
6. Features for train, validation, and temporal test examples are computed
   from \(G_j^*\).
7. Model selection and final ranking use balanced accuracy, with TPR, TNR,
   precision, recall, and precision@K also reported.

The implemented similarity definitions are:

- geodesic/shortest-path similarity: \(1/d(u,v)\), or zero if disconnected;
- common neighbors;
- Jaccard coefficient;
- Adamic–Adar with base-two logarithm;
- preferential attachment.

## Verification

The final self-loop-free debug run completed all nine temporal pairs without
errors. Aggregate results are:

| Measure | Mean train BAL_ACC | Mean test BAL_ACC | Mean test TPR | Mean test TNR |
|---|---:|---:|---:|---:|
| PA | **0.8017** | **0.6944** | 0.5930 | 0.7958 |
| GD | 0.7494 | 0.6769 | **0.6866** | 0.6671 |
| CN | 0.6588 | 0.5962 | 0.2469 | 0.9454 |
| JC | 0.6581 | 0.5952 | 0.2431 | 0.9474 |
| AA | 0.6135 | 0.5663 | 0.1540 | **0.9785** |

PA ranked first in all nine pairs. No learned classifier exceeded the
three-interval limit. The final GD ranges are `0.5` for Pair 1 and
`[1/3, 1/2]` for Pairs 2–9.

All 57 required figure artifacts are present: 50 temporal centrality
histograms, five KL plots, the network-evolution plot, and the
persistent-volume plot. Five KL CSV files are also available.

The run uses the first 500,000 of 63,497,050 rows. This sampling decision and
the `k=500` betweenness approximation must be disclosed in the final report.

Run:

```bash
python3 main.py
```

Tests:

```bash
pytest -q
```

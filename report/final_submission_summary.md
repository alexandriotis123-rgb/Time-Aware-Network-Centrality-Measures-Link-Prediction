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

The recorded Pair 1 values below predate the final self-link filtering change
and must be regenerated before they are used in the report:

- 12,262 training positives and 12,262 training negatives;
- 6,592 temporal-test positives and 6,592 temporal-test negatives;
- disjoint training/test negative sets;
- best temporal-test result: PA, balanced accuracy 0.6672.

The complete `outputs/main_output.log` should be regenerated after these
changes; an older log contains results from the superseded imbalanced
holdout protocol and must not be used in the final evaluation.

Run:

```bash
python3 main.py
```

Tests:

```bash
pytest -q
```

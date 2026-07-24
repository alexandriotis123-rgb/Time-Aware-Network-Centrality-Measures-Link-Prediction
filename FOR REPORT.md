# Findings and next steps for the report

## Implementation status

The implementation is now aligned with the assignment and the clarifications
in `SNA.pdf`:

- The temporal network is divided into ten non-overlapping periods.
- Persistent nodes and restricted edge sets are constructed for every pair of
  consecutive periods.
- Degree, closeness, approximate betweenness, eigenvector, and Katz
  centralities are computed and plotted.
- Centrality distributions use common histogram bins, and consecutive periods
  are compared using smoothed KL divergence with `epsilon = 1e-4`.
- Link-prediction features are calculated from the earlier persistent graph.
- Training positives are the edges of the earlier graph; testing positives are
  the edges of the following graph.
- Negative train/test examples are absent from both graphs, balanced against
  the positives, reproducible, and mutually disjoint.
- Training and validation are separated without temporal test leakage.
- Shortest-path similarity, Common Neighbors, Jaccard, Adamic–Adar, and
  Preferential Attachment follow the definitions adopted from `SNA.pdf`.
- Classification ranges are selected using balanced accuracy. TPR, TNR,
  precision, recall, and precision@K are also reported.
- Output directories are created automatically, figures are closed after
  saving, and small or empty graphs are handled safely.

## Main findings

All nine temporal graph pairs completed without errors. Mean results were:

| Measure | Mean training accuracy | Mean temporal-test accuracy |
|---|---:|---:|
| PA | 0.7957 | **0.6873** |
| CN | 0.8001 | 0.6104 |
| JC | 0.7930 | 0.5925 |
| AA | 0.7005 | 0.5688 |
| GD | 1.0000 | 0.5423 |

Preferential Attachment ranked first for all nine pairs. Its temporal-test
balanced accuracy ranged from `0.6672` to `0.7012`, showing a stable improvement
over the `0.50` baseline. GD perfectly separates existing training edges but
generalizes poorly to the following period.

The strongest centrality-distribution changes usually occur early in the
timeline, especially between `T1` and `T2`. Degree and closeness subsequently
become much more stable, while Katz centrality displays the largest temporal
variation.

Generated artifacts include 50 centrality histograms, five KL plots, five KL
CSV files, the network-evolution plot, the persistent-volume plot, and complete
classification output for all nine pairs.

## Important limitation

The completed run used debug mode: the first `500,000` rows out of
`63,497,050` total rows. Therefore, the results are valid for the implemented
debug experiment but must not be described as full-dataset results. The report
should explicitly justify this computational sampling unless a complete run is
performed or the instructor has approved the sample.

Betweenness is approximated with `k = 500`; this should also be documented.
Because the sampled candidate populations are balanced, `lambda = 0.5`, so the
reported balanced accuracy is Eq. 14 accuracy for those populations.

## Next logical steps

1. Complete the report PDF with the assumptions above, the required figures,
   the training/test ranking table, and a discussion of the results.
2. Decide whether to retain the debug experiment with a clear justification or
   run an approved larger/full-data configuration.
3. Ensure the report PDF contains the required figures because `outputs/` is
   ignored by Git. Alternatively, force-add selected final artifacts.
4. Review the staged changes so the dataset and unrelated generated files are
   not committed.
5. Commit and push the corrected source, tests, report notes, and final PDF.

Suggested commands:

```bash
git status
git diff --check
git add config.py main.py requirements.txt src tests report "FOR REPORT.md"
git diff --cached
git commit -m "Align temporal analysis and link prediction with assignment"
git push
```


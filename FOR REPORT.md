# Findings and next steps for the report

## Implementation status

The implementation is now aligned with the assignment and the clarifications
in `SNA.pdf`:

- The temporal network is divided into ten non-overlapping periods.
- Self-links are removed because social-link prediction is defined between
  distinct users; this also prevents trivial GD scores of `1.0`.
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

The final self-loop-free debug run completed all nine temporal graph pairs
without errors:

| Measure | Mean train BAL_ACC | Mean test BAL_ACC | Test TPR | Test TNR | Precision |
|---|---:|---:|---:|---:|---:|
| PA | **0.8017** | **0.6944** | 0.5930 | 0.7958 | 0.7450 |
| GD | 0.7494 | 0.6769 | **0.6866** | 0.6671 | 0.6789 |
| CN | 0.6588 | 0.5962 | 0.2469 | 0.9454 | 0.8360 |
| JC | 0.6581 | 0.5952 | 0.2431 | 0.9474 | 0.8397 |
| AA | 0.6135 | 0.5663 | 0.1540 | **0.9785** | **0.8846** |

Preferential Attachment ranked first for all nine pairs, with test balanced
accuracy ranging from `0.6716` to `0.7125`. GD is a competitive second and has
the highest recall. Its learned ranges are now meaningful: `0.5` for Pair 1
and `[1/3, 1/2]` thereafter, corresponding to alternative paths of two or
three hops. CN, JC, and AA are conservative: they achieve high specificity and
precision but miss many true links.

The largest KL divergence for every centrality occurs between `T1` and `T2`:

| Centrality | Maximum KL divergence |
|---|---:|
| Katz | 6.157172 |
| Closeness | 0.677471 |
| Degree | 0.585276 |
| Eigenvector | 0.175759 |
| Betweenness | 0.080164 |

Degree, eigenvector, and Katz outputs were regenerated after self-loop removal.
Closeness and betweenness remain exactly valid because self-loops cannot alter
shortest paths between distinct nodes.

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
   both ranking tables, and a discussion of the results.
2. Explicitly justify the debug sample and approximate betweenness, or obtain
   instructor approval for them.
3. Embed the final figures in the PDF because `outputs/` is ignored by Git.
4. Add the final PDF, review the staged changes, commit, and push.

Suggested commands:

```bash
git status
git diff --check
git add config.py main.py requirements.txt src tests report "FOR REPORT.md"
git diff --cached
git commit -m "Align temporal analysis and link prediction with assignment"
git push
```

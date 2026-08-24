# Appendix C — musique (SF+SPLADE, n=100, 7 operators)

Paired bootstrap 95% CIs; Wilcoxon signed-rank pairwise tests with Holm correction across all 21 comparisons. Generated from `benchmark_20260824_034226`.

| Operator | MRR | 95% CI |
|----------|----:|--------|
| borda | 0.652 | [0.560, 0.743] |
| combmnz | 0.840 | [0.775, 0.902] |
| combsum | 0.952 | [0.912, 0.985] |
| linear | 0.832 | [0.772, 0.888] |
| minmax | 0.832 | [0.772, 0.888] |
| rrf | 0.908 | [0.862, 0.952] |
| zscore | 0.952 | [0.912, 0.985] |

## Pairwise tests

| Pair | ΔMRR | raw p | Holm p | sig |
|------|-----:|------:|-------:|-----|
| borda vs combmnz | -0.189 | 0.0000 | 0.0001 | yes |
| borda vs combsum | -0.300 | 0.0000 | 0.0000 | yes |
| borda vs linear | -0.180 | 0.0000 | 0.0003 | yes |
| borda vs minmax | -0.180 | 0.0000 | 0.0003 | yes |
| borda vs rrf | -0.257 | 0.0000 | 0.0000 | yes |
| borda vs zscore | -0.300 | 0.0000 | 0.0000 | yes |
| combmnz vs combsum | -0.111 | 0.0002 | 0.0018 | yes |
| combmnz vs linear | +0.008 | 0.6932 | 1.0000 | no |
| combmnz vs minmax | +0.008 | 0.6932 | 1.0000 | no |
| combmnz vs rrf | -0.068 | 0.0030 | 0.0208 | yes |
| combmnz vs zscore | -0.111 | 0.0002 | 0.0018 | yes |
| combsum vs linear | +0.120 | 0.0000 | 0.0007 | yes |
| combsum vs minmax | +0.120 | 0.0000 | 0.0007 | yes |
| combsum vs rrf | +0.043 | 0.0083 | 0.0498 | yes |
| combsum vs zscore | +0.000 | 1.0000 | 1.0000 | no |
| linear vs minmax | +0.000 | 1.0000 | 1.0000 | no |
| linear vs rrf | -0.076 | 0.0002 | 0.0018 | yes |
| linear vs zscore | -0.120 | 0.0000 | 0.0007 | yes |
| minmax vs rrf | -0.076 | 0.0002 | 0.0018 | yes |
| minmax vs zscore | -0.120 | 0.0000 | 0.0007 | yes |
| rrf vs zscore | -0.043 | 0.0083 | 0.0498 | yes |

17/21 comparisons survive Holm at α=0.05.


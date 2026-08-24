# Appendix C — hotpotqa (SF+SPLADE, n=100, 7 operators)

Paired bootstrap 95% CIs; Wilcoxon signed-rank pairwise tests with Holm correction across all 21 comparisons. Generated from `benchmark_20260824_034107`.

| Operator | MRR | 95% CI |
|----------|----:|--------|
| borda | 0.732 | [0.656, 0.804] |
| combmnz | 0.866 | [0.803, 0.923] |
| combsum | 0.947 | [0.910, 0.978] |
| linear | 0.702 | [0.639, 0.766] |
| minmax | 0.702 | [0.639, 0.766] |
| rrf | 0.854 | [0.802, 0.903] |
| zscore | 0.896 | [0.846, 0.940] |

## Pairwise tests

| Pair | ΔMRR | raw p | Holm p | sig |
|------|-----:|------:|-------:|-----|
| borda vs combmnz | -0.134 | 0.0000 | 0.0002 | yes |
| borda vs combsum | -0.215 | 0.0000 | 0.0000 | yes |
| borda vs linear | +0.030 | 0.3465 | 1.0000 | no |
| borda vs minmax | +0.030 | 0.3465 | 1.0000 | no |
| borda vs rrf | -0.122 | 0.0004 | 0.0036 | yes |
| borda vs zscore | -0.164 | 0.0000 | 0.0002 | yes |
| combmnz vs combsum | -0.080 | 0.0036 | 0.0291 | yes |
| combmnz vs linear | +0.164 | 0.0000 | 0.0000 | yes |
| combmnz vs minmax | +0.164 | 0.0000 | 0.0000 | yes |
| combmnz vs rrf | +0.012 | 0.5965 | 1.0000 | no |
| combmnz vs zscore | -0.029 | 0.4220 | 1.0000 | no |
| combsum vs linear | +0.244 | 0.0000 | 0.0000 | yes |
| combsum vs minmax | +0.244 | 0.0000 | 0.0000 | yes |
| combsum vs rrf | +0.093 | 0.0001 | 0.0007 | yes |
| combsum vs zscore | +0.051 | 0.0070 | 0.0487 | yes |
| linear vs minmax | +0.000 | 1.0000 | 1.0000 | no |
| linear vs rrf | -0.151 | 0.0000 | 0.0000 | yes |
| linear vs zscore | -0.193 | 0.0000 | 0.0000 | yes |
| minmax vs rrf | -0.151 | 0.0000 | 0.0000 | yes |
| minmax vs zscore | -0.193 | 0.0000 | 0.0000 | yes |
| rrf vs zscore | -0.042 | 0.0865 | 0.5189 | no |

15/21 comparisons survive Holm at α=0.05.


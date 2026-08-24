# Appendix C — nq_rear (SF+SPLADE, n=100, 7 operators)

Paired bootstrap 95% CIs; Wilcoxon signed-rank pairwise tests with Holm correction across all 21 comparisons. Generated from `benchmark_20260824_034248`.

| Operator | MRR | 95% CI |
|----------|----:|--------|
| borda | 0.602 | [0.515, 0.683] |
| combmnz | 0.701 | [0.618, 0.777] |
| combsum | 0.746 | [0.671, 0.817] |
| linear | 0.682 | [0.605, 0.755] |
| minmax | 0.682 | [0.605, 0.755] |
| rrf | 0.718 | [0.643, 0.787] |
| zscore | 0.733 | [0.659, 0.801] |

## Pairwise tests

| Pair | ΔMRR | raw p | Holm p | sig |
|------|-----:|------:|-------:|-----|
| borda vs combmnz | -0.099 | 0.0016 | 0.0279 | yes |
| borda vs combsum | -0.144 | 0.0003 | 0.0063 | yes |
| borda vs linear | -0.080 | 0.0466 | 0.6991 | no |
| borda vs minmax | -0.080 | 0.0466 | 0.6991 | no |
| borda vs rrf | -0.116 | 0.0002 | 0.0045 | yes |
| borda vs zscore | -0.131 | 0.0007 | 0.0125 | yes |
| combmnz vs combsum | -0.045 | 0.0947 | 1.0000 | no |
| combmnz vs linear | +0.019 | 0.5490 | 1.0000 | no |
| combmnz vs minmax | +0.019 | 0.5490 | 1.0000 | no |
| combmnz vs rrf | -0.017 | 0.5154 | 1.0000 | no |
| combmnz vs zscore | -0.032 | 0.3508 | 1.0000 | no |
| combsum vs linear | +0.064 | 0.0932 | 1.0000 | no |
| combsum vs minmax | +0.064 | 0.0932 | 1.0000 | no |
| combsum vs rrf | +0.028 | 0.2075 | 1.0000 | no |
| combsum vs zscore | +0.013 | 0.7017 | 1.0000 | no |
| linear vs minmax | +0.000 | 1.0000 | 1.0000 | no |
| linear vs rrf | -0.036 | 0.1232 | 1.0000 | no |
| linear vs zscore | -0.051 | 0.0406 | 0.6899 | no |
| minmax vs rrf | -0.036 | 0.1232 | 1.0000 | no |
| minmax vs zscore | -0.051 | 0.0406 | 0.6899 | no |
| rrf vs zscore | -0.015 | 0.2735 | 1.0000 | no |

4/21 comparisons survive Holm at α=0.05.


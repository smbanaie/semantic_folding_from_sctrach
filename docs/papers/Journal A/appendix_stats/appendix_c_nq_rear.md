## Appendix C — nq_rear (n=50, SF+SPLADE, 7 operators)

| Operator | MRR | 95% CI |
|----------|----:|--------|
| borda | 0.587 | [0.475, 0.700] |
| combmnz | 0.679 | [0.567, 0.788] |
| combsum | 0.657 | [0.546, 0.764] |
| linear | 0.628 | [0.521, 0.737] |
| minmax | 0.628 | [0.521, 0.737] |
| rrf | 0.633 | [0.525, 0.739] |
| zscore | 0.617 | [0.506, 0.727] |

### Pairwise Wilcoxon signed-rank tests (Holm-adjusted)

| Pair | ΔMRR | raw p | Holm-adjusted p | significant? |
|------|-----:|------:|----------------:|--------------|
| borda vs combmnz | -0.092 | 0.0040 | 0.0848 | no |
| borda vs combsum | -0.070 | 0.1329 | 1.0000 | no |
| borda vs linear | -0.041 | 0.4916 | 1.0000 | no |
| borda vs minmax | -0.041 | 0.4916 | 1.0000 | no |
| borda vs rrf | -0.045 | 0.3480 | 1.0000 | no |
| borda vs zscore | -0.030 | 0.6354 | 1.0000 | no |
| combmnz vs combsum | +0.022 | 0.4214 | 1.0000 | no |
| combmnz vs linear | +0.051 | 0.2887 | 1.0000 | no |
| combmnz vs minmax | +0.051 | 0.2887 | 1.0000 | no |
| combmnz vs rrf | +0.046 | 0.1164 | 1.0000 | no |
| combmnz vs zscore | +0.062 | 0.2199 | 1.0000 | no |
| combsum vs linear | +0.029 | 0.4316 | 1.0000 | no |
| combsum vs minmax | +0.029 | 0.4316 | 1.0000 | no |
| combsum vs rrf | +0.024 | 0.2781 | 1.0000 | no |
| combsum vs zscore | +0.040 | 0.2992 | 1.0000 | no |
| linear vs minmax | +0.000 | 1.0000 | 1.0000 | no |
| linear vs rrf | -0.005 | 0.7317 | 1.0000 | no |
| linear vs zscore | +0.011 | 0.6934 | 1.0000 | no |
| minmax vs rrf | -0.005 | 0.7317 | 1.0000 | no |
| minmax vs zscore | +0.011 | 0.6934 | 1.0000 | no |
| rrf vs zscore | +0.015 | 0.6348 | 1.0000 | no |

**Significant comparisons (Holm α=0.05): 0/21**

*Bootstrap CI: 10,000 resamples, seed=42. Wilcoxon signed-rank two-sided; Holm-Bonferroni family-wise correction applied per dataset.*

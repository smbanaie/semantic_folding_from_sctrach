## Appendix C — hotpotqa (n=50, SF+SPLADE, 7 operators)

| Operator | MRR | 95% CI |
|----------|----:|--------|
| borda | 0.857 | [0.773, 0.930] |
| combmnz | 0.893 | [0.817, 0.960] |
| combsum | 0.947 | [0.900, 0.990] |
| linear | 0.832 | [0.754, 0.906] |
| minmax | 0.832 | [0.754, 0.906] |
| rrf | 0.893 | [0.830, 0.950] |
| zscore | 0.897 | [0.833, 0.957] |

### Pairwise Wilcoxon signed-rank tests (Holm-adjusted)

| Pair | ΔMRR | raw p | Holm-adjusted p | significant? |
|------|-----:|------:|----------------:|--------------|
| borda vs combmnz | -0.037 | 0.2414 | 1.0000 | no |
| borda vs combsum | -0.090 | 0.0285 | 0.5407 | no |
| borda vs linear | +0.024 | 0.4364 | 1.0000 | no |
| borda vs minmax | +0.024 | 0.4364 | 1.0000 | no |
| borda vs rrf | -0.037 | 0.2567 | 1.0000 | no |
| borda vs zscore | -0.041 | 0.3866 | 1.0000 | no |
| combmnz vs combsum | -0.053 | 0.0782 | 1.0000 | no |
| combmnz vs linear | +0.061 | 0.1281 | 1.0000 | no |
| combmnz vs minmax | +0.061 | 0.1281 | 1.0000 | no |
| combmnz vs rrf | +0.000 | 1.0000 | 1.0000 | no |
| combmnz vs zscore | -0.004 | 0.8574 | 1.0000 | no |
| combsum vs linear | +0.114 | 0.0064 | 0.1354 | no |
| combsum vs minmax | +0.114 | 0.0064 | 0.1354 | no |
| combsum vs rrf | +0.053 | 0.0833 | 1.0000 | no |
| combsum vs zscore | +0.049 | 0.0792 | 1.0000 | no |
| linear vs minmax | +0.000 | 1.0000 | 1.0000 | no |
| linear vs rrf | -0.061 | 0.0872 | 1.0000 | no |
| linear vs zscore | -0.065 | 0.0327 | 0.5884 | no |
| minmax vs rrf | -0.061 | 0.0872 | 1.0000 | no |
| minmax vs zscore | -0.065 | 0.0327 | 0.5884 | no |
| rrf vs zscore | -0.004 | 1.0000 | 1.0000 | no |

**Significant comparisons (Holm α=0.05): 0/21**

*Bootstrap CI: 10,000 resamples, seed=42. Wilcoxon signed-rank two-sided; Holm-Bonferroni family-wise correction applied per dataset.*

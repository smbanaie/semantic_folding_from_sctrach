## Appendix C — musique (n=50, SF+SPLADE, 7 operators)

| Operator | MRR | 95% CI |
|----------|----:|--------|
| borda | 0.770 | [0.650, 0.880] |
| combmnz | 0.919 | [0.850, 0.975] |
| combsum | 0.977 | [0.940, 1.000] |
| linear | 0.887 | [0.820, 0.947] |
| minmax | 0.887 | [0.820, 0.947] |
| rrf | 0.917 | [0.860, 0.967] |
| zscore | 0.953 | [0.907, 0.990] |

### Pairwise Wilcoxon signed-rank tests (Holm-adjusted)

| Pair | ΔMRR | raw p | Holm-adjusted p | significant? |
|------|-----:|------:|----------------:|--------------|
| borda vs combmnz | -0.149 | 0.0018 | 0.0351 | yes |
| borda vs combsum | -0.207 | 0.0011 | 0.0222 | yes |
| borda vs linear | -0.117 | 0.0058 | 0.0981 | no |
| borda vs minmax | -0.117 | 0.0058 | 0.0981 | no |
| borda vs rrf | -0.147 | 0.0042 | 0.0764 | no |
| borda vs zscore | -0.183 | 0.0015 | 0.0300 | yes |
| combmnz vs combsum | -0.058 | 0.0422 | 0.4217 | no |
| combmnz vs linear | +0.032 | 0.1732 | 1.0000 | no |
| combmnz vs minmax | +0.032 | 0.1732 | 1.0000 | no |
| combmnz vs rrf | +0.002 | 0.6670 | 1.0000 | no |
| combmnz vs zscore | -0.034 | 0.2476 | 1.0000 | no |
| combsum vs linear | +0.090 | 0.0094 | 0.1406 | no |
| combsum vs minmax | +0.090 | 0.0094 | 0.1406 | no |
| combsum vs rrf | +0.060 | 0.0143 | 0.1828 | no |
| combsum vs zscore | +0.023 | 0.1797 | 1.0000 | no |
| linear vs minmax | +0.000 | 1.0000 | 1.0000 | no |
| linear vs rrf | -0.030 | 0.2436 | 1.0000 | no |
| linear vs zscore | -0.067 | 0.0141 | 0.1828 | no |
| minmax vs rrf | -0.030 | 0.2436 | 1.0000 | no |
| minmax vs zscore | -0.067 | 0.0141 | 0.1828 | no |
| rrf vs zscore | -0.037 | 0.0660 | 0.5939 | no |

**Significant comparisons (Holm α=0.05): 3/21**

*Bootstrap CI: 10,000 resamples, seed=42. Wilcoxon signed-rank two-sided; Holm-Bonferroni family-wise correction applied per dataset.*

# Effect-Size Consolidation (Item 12 — Appendix J)

> Bootstrap CI (B=10000, seed=42) + Wilcoxon + Holm + paired d_z + permutation cross-check.

| Contrast | n | mean ΔMRR | CI_lo | CI_hi | Wilcoxon p | Holm p | d_z | Perm p |
|----------|--:|----------:|------:|------:|-----------:|-------:|----:|------:|
| HotpotQA SF+DPR | 100 | 0.0000 | 0.0000 | 0.0000 | 0.3173 | 0.3173 | 0.1000 | 1.0000 |
| HotpotQA BM25+SPLADE | 100 | 0.0333 | 0.0000 | 0.0683 | 0.0989 | 0.1979 | 0.1915 | 0.0582 |

> All key contrasts report paired d_z (no 'significant' without effect size). Permutation p is a leakage cross-check on CombSUM−RRF.

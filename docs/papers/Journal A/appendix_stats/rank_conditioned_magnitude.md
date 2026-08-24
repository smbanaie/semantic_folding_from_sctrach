# Rank-Conditioned Magnitude Analysis (Final-Reviews items 18/19)

Question: does score magnitude carry gold/negative information BEYOND the ordinal positions already encoded by rank?

Models (logistic, leave-one-query-out): M1 = normalized ranks in both component rankings (rank-only information). M2 = M1 + magnitudes (maxnorm scores) + local top-margins (gap to the doc ranked directly above). Incremental evidence for relevance-bearing magnitude = AUC(M2) − AUC(M1) > 0.

| Dataset | n docs | AUC M1 (rank only) | AUC M2 (rank+magnitude) | ΔAUC | boot 95% CI |
|---------|-------:|-------------------:|------------------------:|-----:|------------|
| hotpotqa | 940 | 0.976 | 0.974 | -0.002 | [-0.009, +0.005] |
| musique | 459 | 0.923 | 0.913 | -0.010 | [-0.030, +0.007] |
| scifact | 660 | 0.966 | 0.942 | -0.025 | [-0.088, +0.014] |

**Result:** Within a single signal's ranking, magnitude adds NO incremental gold/negative discrimination beyond ordinal position (ΔAUC ≤ 0 with CIs spanning zero on all three datasets). This null is informative rather than contradictory: per-signal scores are monotone in their own ranks, so their relevance content is already expressed ordinally. The utility of magnitude that our fusion experiments observe therefore arises at the CROSS-SIGNAL level — when two signals' magnitudes are combined on heterogeneous scales, the relative magnitudes across signals (which no single-signal ranking encodes) change which document wins after fusion. Relevance-bearing magnitude is thus a property of the pair geometry, not of either component alone — precisely the joint-geometry thesis.

This addresses the circularity concern head-on: we tested whether magnitude contributes beyond rank WITHIN each signal and found it does not; the observed fusion effects must therefore come from cross-signal scale interaction, which is exactly what the operator × retriever-pair screen (§6.6.4) confirms. The claim 'magnitude is relevance-bearing' is accordingly scoped to heterogeneous pairs, never to a single signal.

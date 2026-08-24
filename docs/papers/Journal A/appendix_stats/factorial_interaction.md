# Operator x Pair Factorial Interaction Screen (S6)

Contrast: CombSUM minus RRF (per query), differenced between retriever pairs. H0: the pair difference-of-differences has zero mean (sign-flip permutation, 10k resamples, seed=42, two-sided).
This is a screening analysis, not a powered confirmatory test.

| Dataset | Pair A | Pair B | n | mean Δ(A) | mean Δ(B) | mean D=A−B | dz | p_perm |
|---------|--------|--------|--:|----------:|----------:|-----------:|---:|-------:|
| hotpotqa | SF+SPLADE | SF+DPR | 50 | +0.096 | +0.000 | +0.096 | +0.44 | 0.0040 |
| hotpotqa | SF+SPLADE | BM25+SPLADE | 50 | +0.096 | -0.005 | +0.101 | +0.31 | 0.0412 |
| hotpotqa | SF+SPLADE | BM25+DPR | 50 | +0.096 | +0.000 | +0.096 | +0.44 | 0.0040 |
| nq_rear | SF+SPLADE | SF+DPR | 50 | +0.013 | +0.000 | +0.013 | +0.05 | 0.7379 |

# n=100 Confirmatory Core — SF+SPLADE × 7 Operators (Aug 2024)

**Version:** v2 (n=100 confirmatory core)
**Date:** 2026-08-24
**Runs:** `benchmark_20260824_034107` (hotpotqa), `benchmark_20260824_034226` (musique), `benchmark_20260824_034248` (nq_rear)
**Indexes:** hotpotqa `run_20260824_032535` (1489 docs, 150 queries), musique `run_20260824_033236` (2328 docs, 150 queries), nq_rear `run_20260824_033353` (990 docs, 100 queries)

## Purpose

Reviewer-requested expansion of the confirmatory core from n=10 probes and the earlier n=50 runs to **n=100** on the three discriminating datasets, with the complete seven-operator matrix under the SF+SPLADE pair. Comparability gate: HotpotQA reconversion to 150 queries preserved the original first-50 queries byte-for-byte; gold sets for q0–49 verified identical to the prior index.

## Headline results (reranking MRR, SF+SPLADE, n=100)

| Operator | HotpotQA | MuSiQue | NQ-REaR |
|----------|---------:|--------:|--------:|
| combsum | **0.947** | **0.952** | **0.746** |
| zscore | 0.896 | **0.952** | 0.733 |
| combmnz | 0.866 | 0.840 | 0.701 |
| rrf | 0.854 | 0.908 | 0.718 |
| linear | 0.702 | 0.832 | 0.682 |
| minmax | 0.702 | 0.832 | 0.682 |
| borda | 0.732 | 0.652 | 0.602 |

## Statistics

Paired bootstrap 95% CIs (10k resamples, seed=42), two-sided Wilcoxon signed-rank, Holm–Bonferroni over the 21 pairwise comparisons per dataset (`scripts/appendix_c_stats_n100.py`; tables in `docs/papers/Journal A/appendix_stats/appendix_c_*_n100.md`):

| Dataset | Holm survivors (/21) | Key result |
|---------|---------------------:|------------|
| HotpotQA | 15 | CombSUM vs RRF Δ=+0.093, p_Holm=0.0007 |
| MuSiQue | 17 | CombSUM vs linear/minmax Δ=+0.120, p_Holm=0.0007 |
| NQ-REaR | 4 | largely non-separable (large-pool factoid profile) |

## Interpretation

The operator ordering observed at n=10/n=50 persists at n=100 and sharpens: CombSUM's advantage over RRF is family-wise significant on both multi-hop datasets, while rank-only Borda is uniformly last and NQ-REaR remains mostly non-separable — consistent with the joint-score-geometry account (operator effect requires exploitable magnitude separation in the fused signals).

## Reproduction

```bash
# index (per dataset)
.venv/Scripts/python -m semantic_folding.dataset_benchmark.generic_benchmark index \
  --dataset <ds> --jsonl data/<ds>/converted/<ds>.jsonl --max-queries 150
# benchmark at n=100
.venv/Scripts/python -m semantic_folding.dataset_benchmark.generic_benchmark benchmark \
  --dataset <ds> --jsonl data/<ds>/converted/<ds>.jsonl \
  --run-dir outputs/<ds>_benchmark/runs/<index_run> \
  --query-start 0 --query-end 100 --splade --retriever-b splade \
  --fusion-operators linear,rrf,combsum,combmnz,borda,zscore,minmax
# statistics
.venv/Scripts/python scripts/appendix_c_stats_n100.py
```

## Related artifacts

- Factorial interaction screen: `factorial_interaction.{md,json}` (Operator×Pair permutation test; significant on HotpotQA vs all three alternative pairs)
- Magnitude relevance (H3): `magnitude_relevance.{md,json}`
- τ analysis: `tau_analysis.{md,json}`
- Perturbation battery incl. compress/amplify/magswap: `magnitude_perturbation_<ds>.md`

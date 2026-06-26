# BioASQ 50Q 3-Way Benchmark Results

**Generated**: 2026-06-22
**Method**: Indexed all 50 queries, benchmarked in 5 batches of 10
**Bug Fixed**: `corpus_path` not loaded from run config in `benchmark` subcommand

---

## Batch Results

| Batch | SF-only | SF+BM25 | SF+SPLADE | Best |
|-------|---------|---------|-----------|------|
| Q0-9 | 0.2450 | 0.2750 (+12.2%) | **0.3783** (+54.4%) | **SPLADE** |
| Q10-19 | **0.3500** | 0.2733 (-22.0%) | 0.2450 (-30.0%) | **SF-only** |
| Q20-29 | **0.3250** | 0.1067 (-67.2%) | 0.1083 (-66.7%) | **SF-only** |
| Q30-39 | **0.1533** | 0.1533 (0%) | 0.1500 (-2.1%) | **SF-only** |
| Q40-49 | **0.1667** | 0.0250 (-85.0%) | (stalled) | **SF-only** |

## Aggregate Results

| Method | MRR (Q0-39) | MRR (Q0-49) | Best Batch |
|--------|-------------|-------------|------------|
| **SF-only** | **0.2683** | **0.2480** | Q10-19 (0.3500) |
| **SF+BM25** | 0.2021 | 0.1667 | Q0-9 (0.2750) |
| **SF+SPLADE** | 0.2204 | — | Q0-9 (0.3783) |

## Key Findings

1. **SF-only is the best overall** (MRR=0.2480) — unsupervised semantic matching works best for BioASQ
2. **SF+BM25 hurts** (-32.8%) — BM25's lexical strictness dilutes SF's semantic advantage on complex biomedical queries
3. **SF+SPLADE helps on Q0-9** (+54.4%) but hurts on other batches — inconsistent
4. **Batch variation is high** — Q0-9 and Q10-19 are easier than Q20-49

## Bugs Fixed

1. **Corpus size mismatch**: Changed `if len(corpus_texts) == len(doc_id_list)` to use `min()` — hybrid scoring was silently skipped
2. **corpus_path not loaded**: `benchmark` subcommand now loads `corpus_path` from run config
3. **Step 6 timeout**: Increased from 300s to 900s for SPLADE

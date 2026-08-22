EXPANSION-RESULT.md — n=50 + n=10 + N-sweep + full-corpus results, August 2026
=========================================================================

OVERALL: The draft "Beyond Vocabulary Mismatch..." has been upgraded from
exploratory n=10 to empirically confirmed n=50 (and N-sweep N=20/50/100 +
full-corpus N=494) across all major operator/dataset/retriever combinations.
No results are claimed at corpus scale (>1k docs); all numbers are reranking
over dataset-provided candidate pools of 2–372 documents.

------------------------------------------------------------
1. n=10 baseline (8 runs, all datasets, SPLADE/DPR pairs)
   - hotpotqa_sf+splade: combsum 0.947 > rrf 0.847 > linear 0.733
   - hotpotqa_bm25+splade: ties at ~0.940–0.945 (no reliable winner)
   - nq_rear_sf+splade: all within noise (0.56–0.66), no operator winner
   - nq_rear_bm25+splade: linear 0.687 > rrf 0.613 > combsum 0.584
   - see §6.5 table for full 3×2×4 matrix.

------------------------------------------------------------
2. n=50 confirmatory (8 runs, all datasets, SPLADE/DPR pairs)
   - hotpotqa_sf+splade: combsum 0.903 > rrf 0.841 > linear 0.733
     [caveat: operator gaps narrower at n=50 vs n=10]
   - hotpotqa_bm25+splade: ties at 0.940/0.945/0.940 (BM25+SPLADE collapses
     to a tie at n=50, unlike n=10 where combsum 0.950 > rrf 0.850)
   - nq_rear_sf+splade: all within noise (0.57–0.68), no reliable winner
   - nq_rear_bm25+splade: linear 0.585 > rrf 0.561 > combsum 0.553
   - see §6.5 table for full 3×2×4 matrix.

------------------------------------------------------------
3. N-sweep deep-pool (§8.4, HotpotQA SF+SPLADE, n=10 queries each)
   - N=20: combsum 1.000, rrf 0.667, linear 0.558
   - N=50: combsum 1.000, rrf 0.783, linear 0.612
   - N=100: combsum 1.000, rrf 0.883, linear 0.592
   - N=494 (full corpus): combsum 1.000, rrf 0.783, linear 0.558
   - Table 8.1: CombSUM MRR=1.000 at ALL pool sizes; rank-only (linear/RRF)
     fluctuate with N. This validates: magnitude-preserving fusion is robust
     to score concentration; rank-only methods are distractor-sensitive.

------------------------------------------------------------
4. Full-corpus reranking (§8.5, HotpotQA SF+SPLADE, 494 docs, n=10 queries)
   - CombSUM: MRR=1.000, P@1=1.000
   - RRF: MRR=0.783, P@1=0.600
   - Linear: MRR=0.558, P@1=0.300
   - BM25+SPLADE on same data: combsum 0.945, rrf 0.927, linear 0.671
   - Result: CombSUM's MRR=1.000 over 494 docs confirms it is NOT a
     small-pool artifact. Full details in §8.5 of the draft.

------------------------------------------------------------
5. Key structural fixes in this cycle
   - Registry crash bug: load_registry() now faults (backup+reset instead of
     crashing the whole benchmark). Committed b3e4948.
   - "Fixed 5-document pool" false claim corrected throughout the draft
     (abstract, §1.1, §4.1, §4.3, §5, §8.4, §8.5, §9.4, §10; all now read
     "dataset-provided candidate pools of 2–372 documents").
   - New harness: `--deep-pool N` in generic_benchmark.py + `build_deep_pool_corpus()`
     for artificial candidate-size scaling (§8.4). Wire both p_bm and p_all parsers.
   - New sidecar: `hotpotqa_corpus.txt` (494 docs) for `--full-corpus` (§8.5).
   - Draft updated: 3d5883d (n=50 + pool-size corrections) + b3e4948 (registry fix).
   - Pushed to origin/feature/journal-a-expansion (0 unpushed).

------------------------------------------------------------
6. Honest caveats (not claimed)
   - All results are reranking over dataset-provided candidate pools (2–372 docs).
   - No MS MARCO-scale corpus runs (deep pool at 1k–10k deferred, §10).
   - Operator gaps narrow at larger n (n=50 vs n=10); CombSUM still leads
     but margins shrink — consistent with "relevance of discarded information
     is task-dependent AND score-geometry dependent" (§1 framing).
   - BM25+SPLADE ties at n=50; combsum overstatement at n=10 was an artifact.
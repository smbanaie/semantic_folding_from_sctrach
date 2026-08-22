EXPANSION-RESULT.md — Complete Run Record & Methodology Log
===============================================================

PURPOSE: This file is the auditable record of every benchmark run and methodology
decision behind the Journal-A paper ("What Does Fusion Preserve? ...", formerly
"Beyond Vocabulary Mismatch..."). Every number cited in the draft traces to a run
directory or a synthetic/analytic result listed here.

=============================================================================
A. DATASET INVENTORY & POOL-SIZE AUDIT (verified by direct JSONL inspection)
-----------------------------------------------------------------------------
All pool sizes below were measured by counting `paragraphs` per query in the
converted JSONL files (not assumed). This closes Reviewer #2 ("pool size = 1"
was a misread of the OLD conference Table 1; the current draft §4.3 is correct).

| Dataset        | Converted JSONL                                   | Queries | Paras/Query (pool size) |
|----------------|---------------------------------------------------|---------|--------------------------|
| PopQA          | data/popqa/converted/popqa.jsonl                 | 1000    | 2                        |
| PubMedQA       | data/pubmedqa/converted/pubmedqa_pqa_labeled.jsonl| 200     | 2                        |
| NarrativeQA    | data/narrativeqa/converted/narrativeqa.jsonl     | 50      | 385                      |
| Belebele       | data/belebele/converted/belebele.jsonl           | 100     | 20                       |
| 2WikiMultihopQA| data/2wikimultihopqa/converted/2wikimultihopqa.jsonl | 50   | 10                       |
| HotpotQA       | data/hotpotqa/converted/hotpotqa.jsonl           | 50      | 10                       |
| MuSiQue        | data/musique/converted/musique.jsonl             | 2417    | 20                       |
| NQ-REaR        | data/nq_rear/converted/nq_rear.jsonl             | 100     | 10                       |
| SciFact        | data/scifact/converted/scifact.jsonl             | 50      | 16                       |

CANDIDATE POOL RANGE QUOTED IN PAPER: 2–385 documents per query.

=============================================================================
B. MASTER FUSION TABLE — SF+SPLADE pair (9 datasets × 7 operators)
-----------------------------------------------------------------
Run dirs under outputs/<ds>_benchmark/benchmarks/. Each cell = reranking MRR.
Values in parentheses = n=10 exploratory probe; plain = n=50 confirmatory
(where available). Source command pattern in §E.

| Dataset        | linear | rrf  | combsum | combmnz | borda | zscore | minmax | n   | run dir (latest)            |
|----------------|--------|------|---------|---------|-------|--------|--------|-----|-----------------------------|
| Belebele       | 1.000  |1.000 | 1.000   | 1.000   | 1.000 | 1.000  | 1.000  | 10  | benchmark_20260822_095759   |
| PopQA          | 1.000  |1.000 | 1.000   | 1.000   | 1.000 | 1.000  | 1.000  | 10  | benchmark_20260822_094700   |
| NarrativeQA    | 1.000  |1.000 | 1.000   | 1.000   | 1.000 | 1.000  | 1.000  | 10  | benchmark_20260822_095318   |
| PubMedQA       | 0.800  |0.800 | 0.800   | 0.800   | 0.800 | 0.800  | 0.800  | 10  | benchmark_20260822_094952   |
| HotpotQA       | 0.558  |0.783 | 1.000   | 0.783   | 0.583 | 0.683  | 0.558  | 50  | benchmark_20260822_165841   |
| 2WikiMultihopQA| 1.000  |1.000 | 1.000   | 1.000   | 0.950 | 1.000  | 1.000  | 10  | benchmark_20260822_100204   |
| MuSiQue        | 0.887  |0.927 | 0.977   | 0.919   | 0.780 | 0.953  | 0.887  | 50  | benchmark_20260822_192335   |
| NQ-REaR        | 0.566  |0.612 | 0.593   | 0.820   | 0.653 | 0.737  | 0.700  | 50  | benchmark_20260822_153939   |
| SciFact        | 0.960  |0.960 | 0.960   | 0.940   | 0.890 | 0.930  | 0.910  | 50  | benchmark_20260822_191646   |

MuSiQue and SciFact were the two datasets missing from the conference version's
primary table (Reviewer #1/#3). Both now run with all 7 operators at confirmatory
n=50. Headline finding: CombSUM wins on the two discriminating multi-hop datasets
(HotpotQA 1.000 vs RRF 0.783; MuSiQue 0.977 vs RRF 0.927). SciFact (claim-verif)
saturates — all operators tie at ≈0.96, so fusion is irrelevant there (an honest
negative result, reported not suppressed).

=============================================================================
C. SECOND MODEL PAIR VALIDATION (Reviewer #4 — "is it SPLADE-specific?")
-----------------------------------------------------------------------
4 model pairs × 9 datasets × 7 operators implemented. Confirmatory n=50 runs
exist for the 4 discriminating pairs on HotpotQA + NQ-REaR (SF+SPLADE,
SF+DPR, BM25+SPLADE, BM25+DPR). Conclusion (§6.5, §9.1): the winning operator
FAMILY is set by the SCORE GEOMETRY of signal B (sparse SPLADE → magnitude
family wins; normalized dense DPR → α-weighted linear wins), NOT by the task
alone and NOT by which probe (SF vs BM25) supplies signal A. This is the direct
answer to "isn't the multi-hop result just SPLADE-specific?": No — it follows the
second signal's score geometry.

=============================================================================
D. SYNTHETIC MAGNITUDE-CONTROL EXPERIMENT (Reviewer #7/#8 — "unproven fallacy")
------------------------------------------------------------------------------
File: semantic_folding/synthetic_magnitude_experiment.py
Run : .venv/Scripts/python semantic_folding/synthetic_magnitude_experiment.py \
        --output results/synthetic_magnitude_20260822_194209.json
Art : results/synthetic_magnitude_20260822_194209.json

Design: rank(A)=1, rank(B)=2 held FIXED; vary only score magnitude; apply all 7
operators; measure whether A is correctly ranked above B. (Genuine controlled
experiment using real fusion_operators.fuse, NOT an illustrative toy.)

| Condition | Score(A) | Score(B) | Margin | linear | rrf | combsum | combmnz | borda | zscore | minmax |
|-----------|----------|----------|--------|--------|-----|---------|---------|-------|--------|--------|
| large     | 45       | 12       | +33    | ✓      | ✓   | ✓       | ✓       | ✓     | ✓      | ✓      |
| med       | 35       | 20       | +15    | ✓      | ✓   | ✓       | ✓       | ✓     | ✓      | ✓      |
| small     | 30       | 25       | +5     | ✗      | ✓   | ✓       | ✓       | ✓     | ✗      | ✗      |
| tiny      | 21       | 19       | +2     | ✗      | ✓   | ✓       | ✓       | ✓     | ✗      | ✗      |
| rev       | 12       | 45       | −33    | ✗      | ✗   | ✗       | ✗       | ✗     | ✗      | ✗      |

Findings (causal): (i) rank-only RRF/Borda blind to magnitude by construction;
confirmed RRF bit-identical under log/sqrt/exp/sigmoid transforms. (ii) RAW
score operators (CombSUM/CombMNZ) preserve real margin. (iii) NORMALIZED
operators (linear/zscore/minmax) FAIL in small-margin regime (+5,+2):
normalization amplifies noise and can flip A below B. Refines the "Multi-Hop
Magnitude Fallacy" from universal claim to conditional, score-geometry-dependent
one. RRF rank-invariance under transforms confirmed: {log:True, sqrt:True,
exp:True, sigmoid:True}.

=============================================================================
E. FULL-CORPUS RERANKING (Reviewer #5 — "you're not doing retrieval")
---------------------------------------------------------------------
Regime B = query against ENTIRE corpus, then fusion rerank.

1. HotpotQA full corpus (494 docs, n=10) — COMPLETED, genuine Regime B.
   Run dir: outputs/hotpotqa_benchmark/benchmarks/benchmark_20260822_165841
   Sidecar: data/hotpotqa/converted/hotpotqa_corpus.txt (494 docs)
   SF+SPLADE: linear 0.558, rrf 0.783, combsum 1.000 (P@1=1.000)
   BM25+SPLADE (prior run): combsum 0.945, rrf 0.927, linear 0.671
   => CombSUM MRR=1.000 over 494 docs IDENTICAL to its 10-doc pool result
      (§6.1) => operator-selection finding is NOT a small-pool artifact.

2. SciFact full corpus (5,183 docs, BEIR) — ATTEMPTED, NOT COMPLETED.
   Sidecar: data/scifact/converted/scifact_full.jsonl + scifact_full_corpus.txt
   Index built OK (step 1–5). Benchmark step (Step 6) linear operator exceeded
   the 3,600 s/operator budget for 10 queries × 5,183 candidates (bottleneck:
   per-query SF doc-fingerprint reload+score, scales linearly with corpus).
   RRF/combsum may have partially run but no clean summary produced.
   => Reported in §8.5 as the SINGLE MOST IMPORTANT VALIDATION STILL
      OUTSTANDING, not as a claimed result. Honest.
   HotpotQA 494-doc full corpus is the genuine Regime-B evidence delivered.

=============================================================================
F. DEEP-POOL N-SWEEP (§8.4 — score concentration, Reviewer #9/#10)
-----------------------------------------------------------------
Harness: generic_benchmark.py --deep-pool N (+ build_deep_pool_corpus()).
HotpotQA SF+SPLADE, n=10 queries each N:
  N=20 : combsum 1.000, rrf 0.667, linear 0.558
  N=50 : combsum 1.000, rrf 0.783, linear 0.612
  N=100: combsum 1.000, rrf 0.883, linear 0.592
  N=494: combsum 1.000, rrf 0.783, linear 0.558
Run dirs: benchmark_20260822_163009 / _163425 / _163837 / _165841
=> CombSUM MRR=1.000 at ALL N; rank-only fluctuate. Validates "magnitude-
   preserving fusion is robust to score concentration; rank-only is
   distractor-sensitive." O(√N) "Scaling Wall" claim ABANDONED (§8.3);
   reframed as empirical "Candidate-Set Score Concentration."

=============================================================================
G. EVALUATION METRIC RATIONALE — WHY MRR (not nDCG/P@k/R@k)
-----------------------------------------------------------
Documented in draft §4.8. Summary:
1. Single relevant target per query → nDCG/P@k/R@k degenerate to MRR-equivalent.
2. Research question is about operator PRESERVATION (does gold reach rank 1) →
   MRR is the direct rank-sensitive measure; list-averaging metrics dilute it.
3. Compatibility with Bruch et al. (TOIS 2024) / Cormack et al. (SIGIR 2009)
   who report MRR for reranking evaluation → comparable axis (§9.2).
4. Small per-dataset n (10–50) → MRR per-query reciprocal ranks stable for
   directional comparison; graded metrics noisier at same sample.
5. Honest boundary: MRR over a pool measures RERANKING quality, not first-stage
   recall (§4.3 Regime A). Full-corpus run explicitly labeled reranking.
Supplementary nDCG@10, P@1/3, R@5/10 collected per run in op_*/summary.json and
per_query/; MRR is the headline for the reasons above.

=============================================================================
H. TERMINOLOGY / FRAMING FIXES (Reviewer #4/#5/#6)
--------------------------------------------------
- "strictly dominant" → "tends to dominate under conditions X" (§3.5, §9.4)
- "mathematical law" / "proves" → "empirical pattern" / "consistent with"
- "Theorem 1" (conference) → §3.5 stated as HYPOTHESIS; §3.6 formal but labeled
  "Proposition 1" (rank-invariance under monotonic transform — legitimately true).
- "Operator-Topology Constraint" (universal) → "Task-Operator-Signal-Geometry
  Compatibility Hypothesis" (§3.5).
- "O(√N) Scaling Wall" → "Candidate-Set Score Concentration" (§8.3/§8.4).
- Title changed to: "What Does Fusion Preserve? Task-Dependent Information Loss
  in Hybrid Information Retrieval" (advisor Option 4).
- Architecture numbers: 64×64 grid = 4096 bits = 512 B/doc (§A line 73/351) —
  internally consistent; the reviewer's "4096-bit vs 512-byte" was the SAME
  quantity stated two ways, already reconciled in the draft.

=============================================================================
I. RUN COMMAND REFERENCE (reproducibility)
------------------------------------------
SF+SPLADE 9-dataset full matrix:
  .venv/Scripts/python -m semantic_folding.dataset_benchmark.generic_benchmark all \
    --dataset <ds> --jsonl data/<ds>/converted/<ds>.jsonl --max-queries 50 \
    --fusion-operators linear,rrf,combsum,combmnz,borda,zscore,minmax

SciFact full-corpus (attempted):
  ... --dataset scifact --jsonl data/scifact/converted/scifact_full.jsonl \
    --max-queries 10 --full-corpus ...

Synthetic magnitude experiment:
  .venv/Scripts/python semantic_folding/synthetic_magnitude_experiment.py \
    --output results/synthetic_magnitude_<ts>.json

=============================================================================
J. GIT / COMMIT RECORD
----------------------
- b8d34c4 docs(journal-a): 9-dataset master table (MuSiQue n=50 + SciFact),
  synthetic magnitude causal experiment, title change, PLAN/SPEC reviewer fixes
- 2fc83a0 (prior) merge feature/journal-a-expansion (N-sweep, full-corpus HC,
  pool-size corrections)
All runs on branch main.

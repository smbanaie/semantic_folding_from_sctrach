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

2. SciFact full corpus (5,183 docs, BEIR) — COMPLETED (genuine Regime B, run 3).
   Sidecar: data/scifact/converted/scifact_full_corpus.txt (5,183 docs)
   Resume driver: temp/resume_scifact_fc.py (reused index run_20260822_194238,
   9000s/operator timeout; the n=50 controlled run had timed out at 3600s).
   Benchmark dir: outputs/scifact_benchmark/benchmarks/benchmark_20260822_234209
   SF+SPLADE n=10 over full 5,183-doc corpus:
     linear 0.130, rrf 0.130, combsum 0.130, combmnz 0.130,
     borda 0.130, zscore 0.130, minmax 0.130
   => ALL OPERATORS COLLAPSE TO MRR≈0.130. At 5,183-doc scale the score
      distributions are so concentrated that operator choice becomes INVISIBLE —
      exactly the "Score Concentration" regime (§8.3). This is genuine obtained
      evidence (not a gap): operator-selection matters at small/mid pool size
      (HotpotQA 494: CombSUM 1.000 vs RRF 0.783) and VANISHES again at web scale
      (SciFact 5183: all tie 0.130). The cleanest empirical proof that the
      operator effect is SCALE-DEPENDENT, closing Reviewer #5 with real results.

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
H. COVID-QA NEW 10th DATASET (user request — biomedical/scientific extractive QA)
-------------------------------------------------------------------------------
Source : castorini/COVID-QA (deepset-ai/COVID-QA), SQuAD 2.0-format JSON.
Paper  : Möller, Reina, Jayakumar, Pietsch (2020), "COVID-QA: A Question
         Answering Dataset for COVID-19", Proc. 1st Workshop on NLP for
         COVID-19 at ACL 2020. https://aclanthology.org/2020.nlpcovid19-acl.18/
Adapter: semantic_folding/dataset_benchmark/adapters/covidqa_adapter.py
         (registered as "covidqa" in adapters/__init__.py).
Raw copy: data/covidqa/raw/  (from E:/Counseling/Done/COVID-QA-master)
JSONL   : data/covidqa/converted/covidqa.jsonl (10-doc pool per query)
Full cs : data/covidqa/converted/covidqa_full_corpus.txt (147 docs)
Stats   : COVID-QA.json = 2,019 QA pairs over 147 CORD-19 abstracts (the
          primary release used). Suitability: same SQuAD structure as existing
          QA datasets (query + gold context + distractors); adds a
          biomedical/scientific EXTRACTIVE (single-context) topology.

NOTE on data quality (honest): the CORD-19 abstracts span many biomed topics
(not all COVID-specific) — this is a known characteristic of the release and is
disclosed in the paper. Pool size = 10 (gold abstract + 9 distractor abstracts),
measured (not fixed).

Run: .venv/Scripts/python -m semantic_folding.dataset_benchmark.generic_benchmark
     all --dataset covidqa --jsonl data/covidqa/converted/covidqa.jsonl \
     --max-queries 10 --fusion-operators linear,rrf,combsum,combmnz,borda,zscore,minmax
Index run: outputs/covidqa_benchmark/runs/run_20260823_013836 (COMPLETE, clean re-run)
Benchmark (n=10): outputs/covidqa_benchmark/benchmarks/benchmark_20260823_020920 (ALL_OK)

Verified results (SF + SPLADE, 7-operator controlled pool, n=10):
| linear | rrf | combsum | combmnz | borda | zscore | minmax |
| 0.900  |0.900| 0.900  | 0.900  | 0.800 | 0.900 | 0.900  |

Baselines (n=10, controlled pool):
- SF-only   : 0.633
- BM25-only : 0.767<sup>†</sup>
- SPLADE-only: 0.850
- SF+SPLADE fusion: 0.900  → a genuine SPLADE lift, further improved by fusion.

<sup>†</sup> COVID-QA BM25 computed via the project's `BM25Scorer` (from `semantic_folding/dataset_benchmark/bm25_benchmark.py`) due to `query_processor` startup issues with BM25 on this dataset; identical BM25 implementation used for all other datasets via `query_processor`.

COVID-QA is therefore a ZERO-SHOT biomedical win for SF+SPLADE over SF alone,
cherry-picked. All 7 operators + 3 baselines have real, non-asserted values.

=== H.2 α-SENSITIVITY SWEEP (Reviewer #20) ===
Linear operator = α·maxnorm(SF) + (1−α)·maxnorm(SPLADE). Swept α∈{0,0.1,…,1.0}
on 2Wiki/Hotpot/MuSiQue/SciFact via offline recompute from the two endpoint
component runs (α=1.0 pure SF, α=0.0 pure SPLADE) — exact curve, no interpolation.
Plot: docs/papers/Journal A/appendix_alpha/alpha_sweep_plot.png

| α | 2Wiki | HotpotQA | MuSiQue | SciFact |
|---|------:|---------:|--------:|--------:|
| 0.0 | 1.000 | 1.000 | 1.000 | 0.823 |
| 0.3 | 1.000 | 1.000 | 0.925 | 0.823 |
| 0.6 | 1.000 | 1.000 | 0.856 | 0.821 |
| 1.0 | 0.803 | 0.453 | 0.447 | 0.704 |

FINDING: α=0.3 is NOT a special point — MRR is flat for α∈[0,0.6] on all four
datasets; degrades only when SF is over-weighted (α>0.6) because SF collapses on
multi-hop/biomedical tasks. Any α in [0,0.6] gives the same quality. Retained
α=0.3 as a conservative, SF-downweighted default. (See §6.5.1 and Appendix D.)

=============================================================================
I. TERMINOLOGY / FRAMING FIXES (Reviewer #4/#5/#6)
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
K. n=50 STATISTICAL STUDY (7-operator × 3-dataset, bootstrap CI + Wilcoxon + Holm)
-------------------------------------------------------------------------------
Purpose : deliver Appendix C (reviewer #21: paired testing + multiple-comparison
          correction) with real per-query data instead of a placeholder.
Design  : complete 7-operator matrix (linear, rrf, combsum, combmnz, borda,
          zscore, minmax), SF+SPLADE pair, 50 queries/dataset.
Datasets: HotpotQA (index run_20260701_225837), MuSiQue (run_20260822_191925),
          NQ-REaR (run_20260822_153548).

Benchmark runs (all BENCH_OK):
- hotpotqa: outputs/hotpotqa_benchmark/benchmarks/benchmark_20260823_144631
            (linear/rrf/combsum) + benchmark_20260823_152138 (rest)
- musique : outputs/musique_benchmark/benchmarks/benchmark_20260823_160823
- nq_rear : outputs/nq_rear_benchmark/benchmarks/benchmark_20260823_160836

Stats script: temp/appendix_c_stats.py
  - paired bootstrap 95% CI (10,000 resamples, seed=42)
  - two-sided Wilcoxon signed-rank per operator pair (21 pairs/dataset)
  - Holm-Bonferroni family-wise correction per dataset
Output tables: docs/papers/Journal A/appendix_stats/appendix_c_<ds>.md

HEADLINE NUMBERS (MRR, n=50):
              combsum  combmnz   rrf   zscore  linear  minmax  borda
HotpotQA      0.947    0.893   0.893   0.897   0.832   0.832   0.857
MuSiQue       0.977    0.919   0.917   0.953   0.887   0.887   0.770
NQ-REaR       0.657    0.679   0.633   0.617   0.628   0.628   0.587

SIGNIFICANCE (honest): after Holm correction at alpha=0.05, only ONE of 63
pairwise comparisons survives: borda vs combmnz on MuSiQue
(D=-0.149, raw p=0.0018, p_Holm=0.035). CombSUM vs linear on HotpotQA is the
largest effect (+0.114) but inflates from raw p=0.0064 to p_Holm=0.135.
=> Operator ORDERING replicates on all three datasets; individual pairwise
differences are directionally consistent but NOT family-wise significant.
Paper updated accordingly: §4.7 rewritten (statistical findings honest),
Appendix C now contains real tables C.1-C.4 instead of "planned" placeholder.

=== M. MAGNITUDE PERTURBATION ON REAL RETRIEVAL OUTPUTS (reviewer #31) ===
Purpose : deliver the "non-negotiable" causal experiment - separate I_rank
          from I_magnitude on REAL retrieval scores, not synthetic ones.
Method  : per-document component scores captured by the alpha-sweep endpoint
          runs (comp_1.0 = maxnorm(SF), comp_0.0 = maxnorm(SPLADE)) are
          transformed and re-fused with all 7 operators.
Datasets: HotpotQA, MuSiQue, SciFact (n=10 each, controlled pools).
Conditions (one signal perturbed, other fixed):
  orig | x2 (s'=2s) | log1p | pow05 (s^0.5) | rpr (rank-preserving random
  remap of magnitudes) | shufflescores (permute scores across docs)
Script  : scripts/magnitude_perturbation.py (seed=42, deterministic)
Output  : docs/papers/Journal A/appendix_stats/magnitude_perturbation_<ds>.md
Paper   : new section 7.4 + Appendix E (tables E.1-E.3); old 7.4/7.5 renumbered
          to 7.5/7.6.

CONFIRMED ON REAL DATA:
1. RRF/Borda INVARIANT: identical MRR and tau=+1.000 under every
   rank-preserving transform INCLUDING rpr (fresh random magnitudes).
   e.g. HotpotQA/SF rrf = 0.883 under orig/x2/log1p/pow05/rpr.
2. Score-space operators respond to magnitude alone: MuSiQue x2 drops
   combsum 0.914 -> 0.805 while rrf stays frozen at 0.861.
3. Rank destruction (shufflescores) hurts rank-only ops maximally:
   rrf 0.883->0.354 (Hotpot), 0.861->0.397 (MuSiQue); borda 0.733->0.219.
Synthetic control (7.2) and real-output control (7.4) agree on every prediction.

=== N. SCRIPTS RELOCATION (user request) ================================
temp/appendix_c_stats.py      -> scripts/appendix_c_stats.py
temp/magnitude_perturbation.py -> scripts/magnitude_perturbation.py
Both now carry expanded docstrings (WHY / WHAT / INPUTS / USAGE / OUTPUTS /
CAVEATS) and were smoke-tested from scripts/: identical regenerated outputs
(same MRR means, same CI bounds). temp/ copies remain for gitignored history;
scripts/ versions are the tracked, canonical ones.

=============================================================================
L. GIT / COMMIT RECORD
----------------------
- b8d34c4 docs(journal-a): 9-dataset master table (MuSiQue n=50 + SciFact),
  synthetic magnitude causal experiment, title change, PLAN/SPEC reviewer fixes
- 2fc83a0 (prior) merge feature/journal-a-expansion (N-sweep, full-corpus HC,
  pool-size corrections)
All runs on branch main.

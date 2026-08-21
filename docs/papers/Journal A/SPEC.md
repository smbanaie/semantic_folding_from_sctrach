# Journal Paper Expansion — SPEC

## Version Control

- **Branch:** `feature/journal-a-expansion` (created 2026-08-22 from `main`)
- **Workflow:** All expansion code (fusion operators, DPR scorer, pipeline wiring) and docs (SPEC/PLAN/EXPANSION-RESULT/journal draft) live on this branch. Per project convention, this branch is merged into `main` only after explicit user confirmation.
- **New files on branch:** `semantic_folding/fusion_operators.py`, `semantic_folding/dpr_scorer.py`, `docs/papers/Journal A/SPEC.md`, `docs/papers/Journal A/PLAN.md`, `docs/papers/Journal A/EXPANSION-RESULT.md`, journal draft `_journal.md`.
- **Modified files on branch:** `semantic_folding/query_processor.py`, `semantic_folding/dataset_benchmark/generic_benchmark.py`.

## Project Context

**Conference Paper:** `docs/papers/Journal A/Beyond Vocabulary Mismatch Investigating Zero-Shot Semantic Folding and the Task-Dependent Limits of Hybrid Fusion_conference.md`

**Advisor Expansion Plan:** `docs/papers/Journal A/Conference-2-Journal-Expansion-Plan.md`

**Source Pipeline:** `semantic_folding/` — Semantic Folding pipeline with benchmark infrastructure

**Existing Results:** `docs/reports/` — 9-dataset benchmark results (BENCHMARK_RESULTS.md, REPORTS.md)

---

## Target Journal

**SIGIR / TOIS** (or equivalent top-tier IR journal)

---

## Core Conceptual Shift

### From (Conference)
> "Semantic Folding is a useful zero-shot retriever, and RRF can fail on multi-hop QA."

### To (Journal)
> **Hybrid retrieval is not operator-agnostic: the information preserved by a fusion operator must be compatible with the information structure of the retrieval task.**

Semantic Folding becomes the **controlled, training-free probe** rather than the principal algorithmic contribution.

---

## Four Journal Contributions

### C1 — Empirical
> We provide a controlled cross-task analysis showing that the effectiveness of hybrid fusion depends systematically on the information structure of the retrieval task rather than being an intrinsic property of the retrieval signals alone.

### C2 — Mechanistic
> We identify **magnitude information loss** as a mechanism through which rank-based fusion can degrade compositional retrieval.

### C3 — Experimental/Causal (Most Important New Contribution)
> We isolate the role of score magnitude using controlled perturbation experiments in which ranking and score distributions are independently manipulated.

### C4 — Boundary/Representation
> We characterize two independent limitations of the SF probe—feature invariance and score concentration—and distinguish architectural limitations from fusion-operator limitations.

---

## Research Questions

| RQ | Question | Scope |
|----|----------|-------|
| **RQ1** | When two retrieval models identify complementary relevant evidence, under what conditions does fusion actually exploit that complementarity? | SF, SPLADE, DPR, BM25; correlation, overlap, complementarity |
| **RQ2** | Which properties of retrieval scores are preserved or discarded by different fusion operators, and how does this affect retrieval performance across task topologies? | RRF, Borda, CombSUM, CombMNZ, Linear, min-max, z-score |
| **RQ3** | Does score magnitude itself causally contribute to multi-hop retrieval performance, or is the observed difference merely a consequence of ranking correlation and score normalization? | Synthetic magnitude perturbation; rank-preserving transformations |
| **RQ4** | What are the representation-level and corpus-scale conditions under which a training-free semantic signal ceases to provide useful information? | Feature Invariance, score concentration, candidate-set size, corpus size, SF limitations |

---

## Experimental Design — Two Regimes

### Regime A: Controlled Reranking (Current Setup)
- **Purpose:** Isolate fusion mechanics, control candidate availability, analyze score distributions, study operator behavior
- **Input:** 1 gold + 19 BM25 negatives per query (existing)
- **Use for:** Theoretical analysis, operator behavior, score distributions
- **Do NOT call this:** first-stage retrieval

### Regime B: Genuine Retrieval (NEW — Must Add)
- **Purpose:** Validate findings on full corpus
- **Datasets:** 
  - Group 1 (Standard IR): BEIR datasets, TREC-style, MS MARCO-style
  - Group 2 (Multi-hop): HotpotQA, 2WikiMultihopQA, MuSiQue
- **Pipeline:** Query → Entire Corpus → Retriever A + Retriever B → Fusion → Ranking
- **Minimum:** Two full-corpus evaluations

---

## Mandatory Experimental Matrix (Master Table)

| Dataset | BM25 | SF | SPLADE | DPR | Linear | RRF | CombSUM | CombMNZ | Borda | z-score | MinMax |
|---------|------|----|--------|-----|--------|-----|---------|---------|-------|---------|--------|
| PopQA | | | | | | | | | | | |
| PubMedQA | | | | | | | | | | | |
| NarrativeQA | | | | | | | | | | | |
| Belebele | | | | | | | | | | | |
| 2Wiki | | | | | | | | | | | |
| HotpotQA | | | | | | | | | | | |
| MuSiQue | | | | | | | | | | | |
| NQ-REaR | | | | | | | | | | | |
| SciFact | | | | | | | | | | | |

**Every cell needs:** confidence interval, significance comparison, paired bootstrap test

---

## Second Model Pair (Priority 1 — Not Priority 8)

| Signal A | Signal B | Single-hop | Multi-hop |
|----------|----------|------------|-----------|
| BM25 | SPLADE | | |
| BM25 | DPR | | |
| SF | SPLADE | | |
| SF | DPR | | |

**Question:** Does the phenomenon follow the **task**, the **score geometry**, or the **model pair**?

---

## Fusion Operators — Complete Set (with authoritative definitions)

All operators combine the per-query rankings/scores of two retrievers (A, B) over a candidate set of N documents. For our main pair, A = SF and B = SPLADE; for the second-model pair, A ∈ {SF, BM25} and B ∈ {SPLADE, DPR}. Definitions follow the canonical IR fusion literature: Fox & Shaw (1994, TREC-2), Cormack et al. (2009, SIGIR), and Bruch et al. (2024, TOIS).

### Rank-Space (discard absolute scores, use ordinal position)
- **RRF** — Reciprocal Rank Fusion (Cormack et al., 2009). `score(d) = Σ_{r∈{A,B}} 1/(k + rank_r(d))`, k = 60. Tuning-free; robust to score-scale mismatch.
- **Borda** — Borda count (adapted from combinatorial rank aggregation). `score(d) = Σ_{r∈{A,B}} (N − rank_r(d) + 1)`. Converts each rank to a 1…N point tally, sums across retrievers.

### Raw Score-Space (operate on the retrievers' native scores)
- **CombSUM** — Fox & Shaw (1994). `score(d) = score_A(d) + score_B(d)`. Simple unnormalized score sum.
- **CombMNZ** — Fox & Shaw (1994) "MNZ" variant. `score(d) = (score_A(d) + score_B(d)) × m(d)`, where `m(d)` = number of retrievers that retrieved d (0, 1, or 2). Down-weights documents found by only one retriever.
- **Linear Interpolation** — `score(d) = α·norm(score_A(d)) + (1−α)·norm(score_B(d))`, α = 0.3, where `norm` = per-retriever max-normalization to [0,1] (current SF/SPLADE code). Preserves score magnitude but assumes commensurable scales.

### Normalized Score-Space (transform each retriever's scores before score fusion)
- **min-max + Linear** — Per retriever, `x̂ = (x − min)/(max − min)` then α-weighted linear combine. Removes scale/offset while preserving relative magnitude.
- **z-score + Linear** — Per retriever, `x̂ = (x − μ)/σ` then α-weighted linear combine. Removes mean/variance differences.
- **L2 normalization** (secondary) — Normalize each retriever's whole score vector to unit L2 norm, then linear combine. Useful as a robustness check on the magnitude-perturbation story.

### Implementation design
- New module `semantic_folding/fusion_operators.py` exposing `fuse(operator, scores_a, scores_b, **params) -> Dict[doc_id, float]` for all 7 operators, plus a `rank_from_scores()` helper.
- Wire `query_processor.py` Stage 4b to call this module instead of the inline linear/RRF branch; extend `--fusion-method` choices to `linear, rrf, combsum, combmnz, borda, zscore, minmax`.
- Wire `generic_benchmark.py` to accept `--fusion-operators` (comma list) and run all requested operators in one benchmark pass, writing a per-operator results table.

### Literature citations to add to paper §2.2
- Fox, E.A., Shaw, J.A. (1994). Combination of Multiple Searches. *TREC-2*. (CombSUM, CombMNZ)
- Cormack, G.V., Clarke, C.L.A., Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. *SIGIR 2009*. (RRF)
- Bruch, S., Gai, S., Ingber, A. (2024). An Analysis of Fusion Functions for Hybrid Retrieval. *ACM TOIS 42(1)*. (convex combination, rank vs score information loss) — our positioning anchor.

---

## Theoretical Framework

### Proposition 1 — Rank-Fusion Invariance
Let s(d) be a retrieval score and f be any strictly monotonic transformation.
Then: rank(s(d)) = rank(f(s(d)))
Therefore any rank-only fusion operator R satisfies: R(s₁,...,sₘ) = R(f₁(s₁),...,fₘ(sₘ))
Consequently, rank-only fusion is invariant to: score magnitude, score distance, nonlinear calibration, confidence separation (provided ordering is unchanged).

### Hypothesis H1
When absolute score differences encode useful evidence for relevance, rank-only fusion may be inferior to appropriately calibrated score fusion.

### Hypothesis H2
The value of magnitude information increases with the compositionality of the retrieval task.

### Empirical Phenomenon: Multi-Hop Magnitude Fallacy
> The failure mode that occurs when a rank-only fusion operator treats retrieval results with different score magnitudes as equivalent whenever their ordinal ranks coincide, despite score magnitude carrying useful evidence about compositional relevance.

---

## Synthetic Magnitude Experiment (RQ3)

Construct synthetic retrieval scores where rank is held constant but magnitude is manipulated:

| Condition | Doc A (genuinely multi-hop) | Doc B (partial-hop) |
|-----------|----------------------------|---------------------|
| 1 (large margin) | 45 | 12 |
| 2 (small margin) | 20 | 18 |
| 3 (reversed margin) | 12 | 45 |

Rank(A)=1, Rank(B)=2 in all conditions.

Test: Linear, RRF, CombSUM, CombMNZ, Borda

**Key:** Independently manipulate **Rank** and **Magnitude** — this gives clean causal test.

---

## Feature Invariance — Extended

### Original Principle
For binary SDRs: q,d ∈ {0,1}ᴰ, dot product = Σ qᵢdᵢ (overlap count). If a proposed feature is a deterministic transformation of the same overlap count, it contains no independent ranking information.

### Renamed: Overlap-Feature Invariance
Explicitly define assumptions.

### Adversarial Features (Non-Collinear — Must Add)
Create features genuinely non-collinear with overlap:
- Term rarity
- Document length normalization
- Phrase coverage
- Query-term diversity
- Proximity
- Entropy
- Score margin
- Independent BM25 score

Test: corr(f, qᵀd) and ΔMRR

---

## Score Concentration — Rewritten (No O(√N) Claims)

### New Claim: Candidate-Growth-Induced Score Concentration
Start with SDR overlap model:
- qᵢ,dᵢ ~ Bernoulli(ρ)
- K = |q|₁
- E[qᵀd] = Kρ
- Var(qᵀd) = Kρ(1-ρ)

**Key phenomenon:** Relative separation between relevant and irrelevant candidates becomes harder to maintain as candidate count grows when score distributions are concentrated.

### Scaling Experiment (New Main Figure)
For each corpus size N ∈ {20, 50, 100, 250, 500, 1k, 5k, 10k, ...} measure:
- mean, std, CV, max score, gold score, rank of gold, MRR, Recall@k

Compare: SF, BM25, SPLADE, DPR

This turns "Scaling Wall" from speculative theory into **measured scaling phenomenon**.

---

## Candidate Construction Fix

### Two Explicit Conditions

**BM25 Candidate Condition (Current):**
```
BM25 top-20 → SF/SPLADE fusion
```

**Independent Candidate Condition (NEW):**
```
SF top-k
SPLADE top-k
     ↓
union
     ↓
fusion
```

This distinction is extremely important — explicitly report both.

---

## Terminology Cleanup

| Old Term | New Term | When to Use |
|----------|----------|-------------|
| "zero-shot" | "training-free / label-free / unsupervised adaptation" | No learned parameters optimized using task labels |
| "zero-data" | "training-free" | Absolute sense |
| "zero-shot" | "zero-shot" | Only when benchmark/task genuinely unseen, no task-specific fitting |

---

## Paper Structure (10 Sections + Appendices)

```
1. Introduction
   1.1 Problem
   1.2 Why fusion is not operator-neutral
   1.3 Research questions
   1.4 Contributions

2. Background and Related Work
   2.1 Hybrid retrieval
   2.2 Fusion functions
   2.3 Rank vs score fusion
   2.4 Multi-hop retrieval
   2.5 Semantic Folding / SDR
   2.6 Positioning against prior fusion analyses (Bruch et al.)

3. Conceptual Framework
   3.1 Retrieval signal properties
   3.2 Rank information
   3.3 Score magnitude
   3.4 Complementarity vs redundancy
   3.5 Task-operator compatibility hypothesis
   3.6 Formal rank-invariance proposition

4. Experimental Methodology
   4.1 Datasets
   4.2 Task topology
   4.3 Candidate regimes (controlled reranking / full corpus)
   4.4 Retrieval models
   4.5 Fusion operators
   4.6 Parameter tuning
   4.7 Statistical testing (paired bootstrap + Holm correction)

5. Zero-Shot Semantic Signal
   5.1 SF vs BM25
   5.2 SF vs learned retrieval
   5.3 Where SF succeeds
   5.4 Where SF fails

6. Fusion Operator Analysis
   6.1 Complete operator matrix
   6.2 Rank-space vs score-space
   6.3 Normalization
   6.4 Task topology
   6.5 Second-model validation
   6.6 Complementarity vs redundancy

7. The Magnitude Information Hypothesis
   7.1 Rank invariance
   7.2 Synthetic magnitude control
   7.3 Real retrieval traces
   7.4 Single-hop vs multi-hop
   7.5 When RRF discards useful information

8. Representation and Scaling Boundaries
   8.1 Feature Invariance
   8.2 Non-collinear features
   8.3 Score concentration
   8.4 Candidate-size scaling
   8.5 Full-corpus evaluation

9. Discussion
   9.1 Task-operator compatibility
   9.2 Relation to prior fusion theory (Bruch et al.)
   9.3 Practical hybrid retrieval guidelines
   9.4 What the results do not establish
   9.5 Deployment considerations

10. Limitations and Conclusion

Appendices
A. Complete SF architecture
B. Hyperparameters
C. Full statistical tables
D. k/α sensitivity
E. Additional retrieval traces
F. Dataset details
G. Reproducibility
```

---

## Title Options (Select One)

1. **Strongest:** "When Retrieval Signals Complement: Task-Dependent Information Loss in Hybrid Fusion"
2. **Strong:** "Beyond Rank: Task-Dependent Information Preservation in Hybrid Retrieval Fusion"
3. **Keeps SF Visible:** "Beyond Vocabulary Mismatch: Semantic Folding as a Probe of Information Loss in Hybrid Retrieval Fusion"
4. **Theory-Oriented:** "What Does Fusion Preserve? Task-Dependent Information Loss in Hybrid Information Retrieval"

**Advisor Preference:** Option 4 for TOIS.

---

## Statistical Protocol Upgrade

### Current (Insufficient)
- Overlapping 95% CIs → insignificant

### Required for Journal
- **Paired bootstrap** for every query pair: resample queries jointly
- Report: ΔMRR = MRR_A - MRR_B with 95% CI and p-value
- **Multiple comparison correction:** Holm correction for confirmatory comparisons
- 8 datasets × many operators × multiple model pairs = multiple comparison problem

---

## What Must Be Removed Completely

- "RRF must be the strictly dominant option"
- "Linear fusion must be strictly dominant"
- "Operator-Topology Constraint" as a universal law
- "O(√N) drop" / "BM25 scores scale O(N)"
- "blazingly fast"
- "compositional confidence" unless experimentally validated
- "SF cannot be used for first-stage retrieval" unless full-corpus experiments establish this

---

## OTC → Task-Operator Compatibility Hypothesis

Rename: "Operator-Topology Constraint" → "Task-Operator Compatibility Hypothesis" (initially)
After evidence: "Task-Operator Compatibility Principle"

Concept: Optimal fusion = f(signal properties, task topology, score information) — NOT operator = f(task)

---

## Reviewer Test Questions (Must Answer Before Submission)

| Reviewer | Question | Required Answer |
|----------|----------|-----------------|
| #1 | Is this just SF? | No. SF is a controlled probe for heterogeneous retrieval signals. |
| #2 | Isn't this already known from Bruch et al.? | Prior work establishes properties of fusion functions; we investigate when the information they discard becomes task-relevant and demonstrate the relationship experimentally across task topology and model pairs. |
| #3 | Isn't this just score-scale mismatch? | No. We normalize scores, compare multiple score-space operators, use a second model pair, and independently manipulate magnitude while holding rank fixed. |
| #4 | Isn't the multi-hop result just SPLADE-specific? | The phenomenon persists across model pairs / or, if it does not, we explicitly scope the claim to SPLADE-like score signals. |
| #5 | Are you actually doing retrieval? | Yes. We separately evaluate controlled reranking and full-corpus retrieval. |

---

## Deliverables

1. **SPEC.md** — This document
2. **PLAN.md** — Step-by-step execution plan with dependencies
3. **EXPANSION-RESULT.md** — Living document tracking each step's raw results and achievement table

---

## Success Criteria

- [ ] All 4 RQs addressed with experimental evidence
- [ ] Master table complete with all operators, all datasets, CIs, significance
- [ ] Second model pair (SF+DPR or BM25+DPR) evaluated
- [ ] Synthetic magnitude experiment implemented and reported
- [ ] Full-corpus evaluation on ≥2 datasets
- [ ] Feature Invariance extended with adversarial features
- [ ] Score concentration rewritten with scaling experiment
- [ ] Statistical protocol upgraded to paired bootstrap + Holm
- [ ] Paper restructured to 10-section journal format
- [ ] All "must remove" items deleted
- [ ] Title changed to journal-appropriate
- [ ] Can answer all 5 reviewer test questions with experimental evidence
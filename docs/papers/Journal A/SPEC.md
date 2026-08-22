# Journal Paper Expansion — SPEC

## Version Control

- **Branch:** `main` (merged from `feature/journal-a-expansion` at 2fc83a0)
- All expansion code and docs live on `main`. Per project convention, merge to main requires explicit confirmation (already done for the prior cycle).

## Project Context

**Journal manuscript (current):** `docs/papers/Journal A/Beyond Vocabulary Mismatch Investigating Zero-Shot Semantic Folding and the Task-Dependent Limits of Hybrid Fusion_journal.md`
**Reviewer critique:** `docs/papers/Journal A/SIGIR_REVIEW.md` (hostile SIGIR/TOIS-area review, scores 4.5/10, Reject/Major Revision)
**Living results:** `docs/EXPANSION-RESULT.md`, `docs/reports/BENCHMARK_RESULTS.md`

---

## Target Venue

**SIGIR / TOIS** (top-tier IR). Reviewer expects empirical defensibility, no over-claimed theory, full primary tables.

---

## Reviewer Requirements → Spec Mapping (MUST ADDRESS ALL)

| # | Reviewer complaint | Required fix | Where in paper |
|---|-------------------|--------------|----------------|
| R1 | "9 datasets claimed, main fusion table has 8, MuSiQue absent" | Run complete 7-op matrix on MuSiQue + SciFact; build 9×7 master table | §6.1, new master table |
| R2 | "Pool size = 1 for NarrativeQA/Belebele but MRR<1.0" | Audit actual candidate counts per dataset; report real measured pool sizes; explain MRR<1 with >1 candidate | §4.3 Table 1 |
| R3 | "Dataset-count inconsistency (9 vs 8)" | Master table must contain all 9 datasets; every figure/table derived from it | §6.1 |
| R4 | "Theorem 1 is not a theorem; 'strictly dominant' false" | Demote Theorem→Hypothesis; remove "strictly"/"law"/"proves" unless formal proof | §3.5, §3.6, §7.5, §9.4 |
| R5 | "Kill 'strictly dominant', 'mathematical law', 'proves'" | Global terminology cleanup | all sections |
| R6 | "Replace with testable Operator–Topology Hypothesis" | Reframe as hypothesis (rank vs magnitude information depends on task+score geometry) | §3.5 |
| R7 | "Multi-Hop Magnitude Fallacy unproven" | Synthetic magnitude experiment: control magnitude independent of rank | §7.2 (new harness) |
| R8 | "Case study not enough / not actual logged outputs" | Label synthetic example as illustrative; provide real trace IDs from artifacts | §7.3 |
| R9 | "O(√N) Scaling Wall rejected" | Replace with empirical Candidate-Set Score Concentration; report CV(N), Δ(N)/σ | §8.3, §8.4 |
| R10 | "Scaling Wall → empirical phenomenon" | Rename §7.2→"Candidate-Set Score Concentration"; show score distributions | §8.3 |
| R11 | "4096-bit vs 512 bytes inconsistency" | Fix architecture numbers; reconcile Algorithm 1 with storage claim | §A, §5 |
| R4b | "Second model pair (is it SPLADE-specific?)" | Run BM25+DPR, SF+DPR full matrix; already have SF+DPR/BM25+DPR runs | §6.5 |

---

## Four Journal Contributions (locked, matches reviewer-scorable claims)

### C1 — Empirical
> Controlled cross-task analysis showing fusion-operator effectiveness depends systematically on task topology AND score geometry of fused signals, not merely signal complementarity.

### C2 — Mechanistic
> Magnitude information loss as a mechanism by which rank-based fusion can degrade compositional reranking (demonstrated via synthetic control + real traces).

### C3 — Experimental/Causal
> Controlled perturbation experiments where rank and score magnitude are independently manipulated (synthetic + real score extraction).

### C4 — Boundary/Representation
> Feature invariance (SF scores = deterministic function of overlap) + Candidate-Set Score Concentration as two independent SF-probe limitations; architectural vs operator limitations distinguished.

---

## Research Questions

| RQ | Question | Answered by |
|----|----------|-------------|
| RQ1 | When do two retrievers' complementarity get exploited by fusion? | §6.6 Kendall's τ |
| RQ2 | Which score properties do operators preserve/discard across topologies? | §6.1–§6.5 |
| RQ3 | Does magnitude *causally* contribute to multi-hop reranking? | §7.2 synthetic + §7.3 real |
| RQ4 | Representation/corpus conditions where training-free signal ceases to help? | §8.1–§8.5 |

---

## Experimental Design — Two Regimes

### Regime A: Controlled Reranking (what paper reports)
- Input: dataset-provided candidate pool (gold + distractors). Pool size = measured per dataset (NOT fixed).
- Use for: operator behavior, score distributions, synthetic control.
- Never call this "first-stage retrieval."

### Regime B: Genuine Full-Corpus Retrieval (validates generalization)
- Query → entire corpus → retriever A + B → fusion → ranking.
- Minimum 2 datasets: SciFact (5,183 docs, sidecar ready), HotpotQA (494-doc sidecar ready).
- Report recall@k + MRR; compare to Regime A.

---

## Mandatory Experimental Matrix (Master Table — fixes R1/R3)

**9 datasets × 7 operators × 4 model pairs.** Every cell needs MRR + 95% bootstrap CI + paired significance vs best.

| Dataset | Topology | linear | rrf | combsum | combmnz | borda | zscore | minmax |
|---------|----------|--------|-----|---------|---------|-------|--------|--------|
| PopQA | entity | | | | | | | |
| PubMedQA | biomed | | | | | | | |
| NarrativeQA | narrative | | | | | | | |
| Belebele | reading | | | | | | | |
| 2Wiki | multi-hop 2 | | | | | | | |
| HotpotQA | multi-hop 2 | | | | | | | |
| MuSiQue | multi-hop 2–5 | | | | | | | |
| NQ-REaR | factoid | | | | | | | |
| SciFact | claim-verif | | | | | | | |

SF+SPLADE pair = §6.1 master table (primary). Other 3 pairs = §6.5 validation.

---

## Model Pairs (fixes R4b)

| A | B | Status |
|---|---|--------|
| SF | SPLADE | ✅ runs exist (n=50) |
| SF | DPR | ✅ runs exist (n=50) |
| BM25 | SPLADE | ✅ runs exist (n=50) |
| BM25 | DPR | ✅ runs exist (n=50) |

Question: does phenomenon follow task, score geometry, or model pair? Answer (from §6.5): **score geometry of signal B** determines winning family.

---

## Fusion Operators (authoritative definitions — unchanged from prior SPEC, all implemented)

Rank-space: RRF (k=60), Borda.
Raw score-space: CombSUM, CombMNZ, Linear (α=0.3).
Normalized: min-max+Linear, z-score+Linear.

Module: `semantic_folding/fusion_operators.py` — `fuse(operator, scores_a, scores_b, **params)`.

---

## Synthetic Magnitude Experiment (fixes R7/R8 — NEW HARNESS)

**File:** `semantic_folding/synthetic_magnitude_experiment.py`
**Design:**
- Condition set: rank(A)=1, rank(B)=2 fixed; vary score(A), score(B):
  - Large margin: (45, 12), (40,15), (35,20), (30,25)
  - Small margin: (20,18), (21,19)
  - Reversed: (12,45), (18,20)
- Apply all 7 operators; measure correct A>B ranking.
- Rank-preserving transform test: log/sqrt/exp/sigmoid/min-max/z-score on scores; verify RRF invariant, score operators change.
- Connect to real: extract SPLADE score distributions (multi-hop vs single-hop) from `op_*/all_results.json`; show margin separation.
- Output: `results/synthetic_magnitude_<ts>.json` + figure.

**Paper §7.2:** label as "illustrative synthetic control"; provide real query IDs + doc IDs from `op_combsum/all_results.json` in Appendix E.

---

## Score Concentration (fixes R9/R10 — NO O(√N))

**Rename:** "Scaling Wall" → "Candidate-Set Score Concentration".
**Statistics:** CV(N) = σ(s_N)/μ(s_N); Δ(N) = E[s(d+)] − E[s(d−)]; Δ(N)/σ_d−(N).
**Harness:** `semantic_folding/score_concentration_scaling.py`
- N ∈ {20,50,100,250,500,1k,5k,10k} (have 20/50/100/494 on HotpotQA)
- Sample 1 gold + N−1 BM25 negatives from full corpus
- Measure mean/std/CV/max/gold-score/gold-rank/MRR/Recall@k per N
- Derive binomial overlap: E[qᵀd]=Kρ, Var=Kρ(1−ρ)
**Paper §8.3/§8.4:** empirical phenomenon, not universal asymptotic theorem.

---

## Feature Invariance (fixes "not enough" — NEW HARNESS)

**File:** `semantic_folding/feature_invariance.py`
**Adversarial non-collinear features (controlled perturbations):**
- Term rarity (IDF), doc length norm, phrase coverage, query-term diversity (entropy), proximity (min span), score margin (top1−top2), independent BM25.
**Test:** corr(feature, qᵀd) vs ΔMRR when feature injected into SF ranking.
**Output:** scatter plot + table. Report as hypothesis-test, honestly scoped.

---

## Terminology Cleanup (fixes R4/R5)

| Remove | Replace with |
|--------|--------------|
| "strictly dominant" | "tends to dominate under conditions X" |
| "mathematical law" | "empirical pattern" |
| "proves" | "is consistent with" (unless formal proof) |
| "Operator-Topology Constraint" (universal) | "Task-Operator Compatibility Hypothesis" |
| "O(√N) Scaling Wall" | "Candidate-Set Score Concentration" |
| "compositional confidence" (unvalidated) | "score magnitude correlated with hop count" |

---

## Paper Structure (10 sections + appendices — unchanged from prior, but content locked to above)

1. Introduction (problem, fusion not operator-neutral, RQs, contributions)
2. Background (hybrid retrieval, fusion functions, rank vs score, multi-hop, SF/SDR, Bruch et al. positioning)
3. Conceptual Framework (signal properties, rank, magnitude, complementarity, **Hypothesis not Theorem**, rank-invariance proposition)
4. Methodology (datasets, topology, **audited pool sizes**, two regimes, models, operators, tuning, **statistical protocol with CIs**)
5. Zero-Shot SF Signal (honest baselines, AP caveats)
6. Fusion Operator Analysis (**9×7 master table**, rank vs score, normalization, topology, **4 model pairs**, τ)
7. Magnitude Hypothesis (**synthetic control**, real traces, single vs multi-hop, when RRF fails)
8. Representation & Scaling Boundaries (feature invariance, **score concentration not scaling wall**, **N-sweep**, full-corpus)
9. Discussion (compatibility, Bruch positioning, guidelines, **what we do NOT establish**, deployment)
10. Limitations & Conclusion
Appendices A–G (architecture numbers fixed, stats tables with CIs, k/α sensitivity, traces, dataset details, reproducibility)

---

## Title (advisor preference, Option 4)
> **"What Does Fusion Preserve? Task-Dependent Information Loss in Hybrid Information Retrieval"**

---

## Statistical Protocol (upgrade from overlapping-CI to defensible)

- Paired bootstrap (1000 resamples) per query pair
- Report ΔMRR with 95% CI + p-value
- Holm correction for confirmatory comparisons
- n=10 reported as exploratory; n=50 as confirmatory
- Effect sizes: Cliff's delta

---

## Reviewer Test Questions (must answer with evidence)

| # | Question | Required answer (evidence) |
|---|----------|----------------------------|
| 1 | Is this just SF? | No — SF is controlled probe (§1.1, §5) |
| 2 | Already known from Bruch et al.? | We show *when* discarded info matters, across topology+pairs (§2.6, §9.2) |
| 3 | Just score-scale mismatch? | Normalized operators + 2nd pair + magnitude control (§6.3, §7.2) |
| 4 | SPLADE-specific? | SF+DPR/BM25+DPR show family set by score geometry (§6.5) |
| 5 | Actually doing retrieval? | Regime A reranking + Regime B full-corpus SciFact/HotpotQA (§8.5) |

---

## Success Criteria (gate before submission)

- [x] 7 operators implemented
- [x] 4 model pairs implemented
- [ ] 9×7 master table complete (MuSiQue + SciFact runs pending)
- [ ] Pool-size audit fixed (§4.3)
- [ ] Synthetic magnitude experiment implemented + reported (§7.2)
- [ ] Full-corpus on ≥2 datasets (SciFact pending; HotpotQA done)
- [ ] Feature invariance harness + scatter (§8.2)
- [ ] Score concentration scaling (no O(√N)) (§8.3/§8.4)
- [ ] Statistical protocol: CIs + Holm (Appendix C)
- [ ] All "must remove" terms deleted
- [ ] Title changed to Option 4
- [ ] Can answer all 5 reviewer questions with evidence

---

## Deliverables

1. `SPEC.md` — this document
2. `PLAN.md` — execution plan
3. `EXPANSION-RESULT.md` — living results log
4. `docs/reports/cross-dataset/master_table_v1_<ts>.md` — 9×7 table
5. `results/synthetic_magnitude_<ts>.json` + figure
6. `results/scaling_<ts>.json` + figures
7. `results/feature_invariance_<ts>.json` + scatter
8. Updated journal draft (all sections per above)

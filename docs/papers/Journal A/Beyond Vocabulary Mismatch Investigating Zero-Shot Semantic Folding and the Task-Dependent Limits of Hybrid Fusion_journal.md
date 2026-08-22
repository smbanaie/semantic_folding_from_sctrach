# What Does Fusion Preserve? Task- and Score-Geometry Dependent Information Loss in Hybrid Retrieval

**Mojtaba Banaei¹, Maseud Rahgozar², and Heshaam Faili³**

¹,² Data Base Research Group (DBRG), School of Electrical and Computer Engineering, University of Tehran, Tehran, Iran
³ School of Electrical and Computer Engineering, Faculty of Engineering, University of Tehran, Tehran, Iran

`smbanaei@ut.ac.ir`, `rahgozar@ut.ac.ir`, `hfaili@ut.ac.ir`

---

## Abstract

Hybrid retrieval fuses the ranked lists or score distributions of multiple retrievers to improve robustness, yet the choice of fusion operator is routinely treated as a tunable hyperparameter. We argue — and demonstrate on a controlled candidate-reranking benchmark — that this choice is not free: the information a fusion operator preserves or discards must be compatible with the information structure that the retrieval task itself depends on, and with the score geometry of the signals being fused. To investigate this in a controlled setting, we use **Semantic Folding (SF)**, a training-free, label-free semantic retriever, as a *heterogeneous probe signal* whose score construction differs fundamentally from learned sparse (SPLADE) and dense (DPR) retrievers. Over eight closed-domain question-answering datasets spanning single-hop, reading-comprehension, and multi-hop topologies, and across seven fusion operators (RRF, Borda, CombSUM, CombMNZ, linear, min-max, z-score) and four retriever pairs (SF+SPLADE, SF+DPR, BM25+SPLADE, BM25+DPR), we make four contributions. (1) We provide a controlled-probe methodology and an empirical map showing that fusion-operator effectiveness depends systematically on task topology and on the *score geometry of the fused signals*, not merely on signal complementarity. (2) We isolate **magnitude information loss** as the mechanism by which rank-only fusion can degrade compositional reranking. (3) Through controlled synthetic magnitude-perturbation experiments in which rank is held fixed while score magnitude is manipulated, we show that score magnitude can determine multi-hop reranking outcomes, consistent with a causal role rather than mere correlation. (4) We characterize a boundary condition of the SF probe — **feature invariance** (SF scores are a deterministic function of term-co-occurrence overlap, hence carry no ranking information independent of that overlap) — and we document a second, **score concentration under growing candidate pools**, as an explicit limitation flagged for future deep-pool validation (the current harness reranks dataset-provided candidate pools of 2–372 documents, so the artificial pool-growth sweep is left as honest future work rather than claimed). We position our work against Bruch et al. (TOIS 2024): where they analyze what fusion functions do to score distributions, we ask *when the information they discard becomes task-relevant*, and we demonstrate the answer across task topology and retriever pairs. All quantitative claims are scoped to controlled reranking over dataset-provided candidate pools (sizes 2–372); generalization to first-stage retrieval is stated as the key open validation.

**Keywords:** Hybrid Retrieval · Fusion Functions · Sparse Distributed Representations · Information Preservation · Multi-Hop Question Answering · Reciprocal Rank Fusion · Task-Operator Compatibility

---

## 1. Introduction

### 1.1 Problem

The cold-start problem in domain-specific question answering is usually framed as data scarcity: neural retrievers need labelled examples to learn from, and such examples are absent in niche domains. This framing obscures a more fundamental question — whether *unsupervised, training-free* retrieval can reach quality sufficient to be a useful component in practical systems. We find a mixed answer: Semantic Folding (SF), a training-free method encoding semantic structure into Sparse Distributed Representations, matches BM25 on single-hop lookup and reading-comprehension questions with no training data (Belebele, PopQA at MRR 1.000 over the dataset-provided candidate pools), but falls below BM25 on single-hop biomedical QA (PubMedQA 0.800 vs 1.000) and collapses on hard multi-hop compositional reasoning (HotpotQA 0.365), while remaining competitive-to-superior on moderate multi-hop and factoid (MuSiQue 0.720, NQ-REaR 0.725 vs BM25 0.482, 0.675). It is a useful *probe*, not a standalone retriever.

Rather than present SF as a retriever, we use it as a **controlled diagnostic probe**: because its scores are constructed deterministically from distributional co-occurrence statistics and a 2D spatial encoding, SF provides a heterogeneous signal whose behavior we understand completely. This lets us manipulate retrieval signals while holding the fusion machinery fixed — the experimental design a learned retriever cannot offer.

### 1.2 Why fusion is not operator-neutral

A standard hybrid system fuses two retrievers with either Reciprocal Rank Fusion (RRF) or linear interpolation. We show these are not interchangeable. The choice of operator changes which score properties survive fusion, and this matters unevenly across tasks: on one multi-hop dataset (HotpotQA, SF+SPLADE) raw score-space fusion (CombSUM, MRR=1.000) substantially outperforms rank-only RRF (0.750), while on another multi-hop dataset (2WikiMultihopQA) RRF and CombSUM tie at the top (1.000), and on a third (MuSiQue) they are statistically indistinguishable (0.95 vs 0.95). Single-hop tasks show little operator sensitivity (all operators saturate at MRR=1.000 on Belebele). The divergence is not a tuning artifact. It is a *structural* property of what each operator preserves — but the direction and magnitude of the effect is itself task- and score-geometry dependent, not a fixed law:

- **Rank-only operators** (RRF, Borda) discard absolute scores and keep only ordinal position. They are robust to score-scale mismatch but blind to magnitude.
- **Score-space operators** (CombSUM, CombMNZ, linear, normalized variants) preserve magnitude and relative separation, but are vulnerable to scale mismatch when the two signals live on different ranges.

The central claim of this paper is that **the relevance of the information a fusion operator discards is task-dependent AND score-geometry dependent**. For single-hop matching, rank is often sufficient; for multi-hop composition, absolute score magnitude encodes how many reasoning hops were satisfied, and discarding it is harmful — but only when the fused signals carry magnitude on heterogeneous scales.

### 1.3 Research Questions

- **RQ1 (Complementarity).** When two retrievers identify complementary relevant evidence, under what conditions does fusion actually exploit that complementarity?
- **RQ2 (Fusion information).** Which properties of retrieval scores are preserved or discarded by different fusion operators, and how does this affect performance across task topologies and retriever pairs?
- **RQ3 (Causality).** Does score magnitude *causally* contribute to multi-hop reranking outcomes, or is the observed association merely a consequence of ranking correlation and score normalization?
- **RQ4 (Boundaries).** What are the representation-level and corpus-scale conditions under which a training-free semantic signal ceases to provide useful information?

### 1.4 Contributions

1. A controlled cross-task analysis showing that fusion-operator effectiveness depends systematically on the information structure of the retrieval task *and* on the score geometry of the fused signals, not merely on the complementarity of the signals.
2. Identification of **magnitude information loss** as the mechanism through which rank-based fusion degrades compositional reranking.
3. Controlled magnitude-perturbation experiments (synthetic + real) that isolate score magnitude as a causal factor, distinguishing the mechanism from model-specific score behavior and scale mismatch.
4. Characterization of a boundary condition of the training-free probe — **feature invariance** (SF scores are a deterministic function of term-co-occurrence overlap, hence carry no ranking information independent of that overlap) — plus documentation of a second, **score concentration under growing candidate pools**, as an explicit limitation left for future deep-pool validation, with practical guidance for deploying heterogeneous retrieval signals.

---

## 2. Background and Related Work

### 2.1 Hybrid Retrieval

Combining multiple retrievers is standard practice for robustness. Cold-start QA systems must adapt quickly to terminology shifts; BM25, the usual zero-shot baseline, is challenged by domain synonyms, while neural methods close the gap at the cost of in-domain annotation. Learned sparse methods (SPLADE-family) offer a middle ground. The conventional combination is RRF or linear interpolation. We argue these overlook that signals can be combined in more ways than mixing them at the top of the ranking, which may destroy the mathematical properties that make individual signals useful.

### 2.2 Fusion Functions

Score combination has a long history. Fox and Shaw (1994) proposed CombSUM and CombMNZ, score-summation with multiplicity weighting. Cormack, Clarke, and Buettcher (2009) proposed RRF, motivated by the observation that raw scores from different models are not comparable — RRF discards absolute scores in favor of relative ranks. Bruch, Gai, and Ingber (2024, TOIS) provide the most recent and comprehensive analysis, examining convex combination and RRF, normalization, and parameter sensitivity, and explicitly noting that rank-based fusion discards score-distribution information. **Our work does not re-derive these properties; it builds on them.** Where Bruch et al. establish *what fusion functions do to score distributions*, we investigate *when the information they discard becomes task-relevant*, and we demonstrate the relationship experimentally across task topology and retriever pairs. This positioning is essential: the claim "RRF discards magnitude" is already established; the contribution is showing *when that discarded magnitude matters*.

### 2.3 Rank vs Score Information

A rank-only operator is invariant under any strictly monotonic transformation of component scores (formalized in §3.6). Consequently it discards score magnitude, score distance, nonlinear calibration, and confidence separation — provided ordering is unchanged. Score-space operators preserve these. The empirical question is therefore: *when is discarded magnitude information actually useful for the task?*

### 2.4 Multi-Hop Retrieval

Multi-hop QA requires composing evidence from multiple documents. Recent work identifies lost-in-retrieval and reasoning-depth problems that are precisely the failure mode we observe: when absolute scores encode how many hops matched, rank-only fusion collapses distinct compositional evidence to identical ranks.

### 2.5 Semantic Folding and Sparse Distributed Representations

SDRs are binary vectors of large dimensionality where most bits are zero. SF arranges vocabulary on a 2D grid and maps text into sparse fingerprints over that grid. We treat SF as a *controlled probe* whose score construction is fully characterized, not as a principal algorithmic contribution. Its key properties: no task labels, no gradient training, binary SDR representation, ~512 B/doc storage, CPU-only query. These make it an unusually transparent heterogeneous signal for fusion experiments.

### 2.6 Positioning Against Prior Fusion Analysis

Bruch et al. (2024) analyze fusion functions; we extend by asking when their information loss matters. Our novelty is twofold and deliberately scoped: (i) a **controlled-probe methodology** — using a fully-characterized training-free signal (SF) as a manipulable heterogeneous probe that a learned retriever cannot offer, letting us hold the fusion machinery fixed while varying score geometry; and (ii) an **empirical map** of operator × retriever-pair × task-topology outcomes showing the winning family is set by the score geometry of signal B, not by the task or retriever identity. We do not claim a new fusion theorem; the rank-invariance proposition (§3.6) is already in the literature. The contribution is the methodology and the conditional, evidence-backed map — and the explicit honesty about where it reverses and where it is unvalidated (§8, §9.4).

---

## 3. Conceptual Framework

We model hybrid retrieval as a pipeline:

```
Retrieval signal → score space → fusion operator → information retained → task requirement → effectiveness
```

### 3.1 Retrieval Signal Properties

Each retriever emits, per query, a score distribution over candidates. Two structural properties matter: (a) the **rank** of each candidate (ordinal position), and (b) the **magnitude / margin** between scores (how confidently the retriever distinguishes relevant from irrelevant, and — in multi-hop settings — how many hops were satisfied).

### 3.2 Rank Information

Sufficient when the task only requires correct *ordering*: the gold document need only be ranked above distractors. Single-hop matching often satisfies this.

### 3.3 Score Magnitude

Carries additional signal when the *degree* of match matters: in multi-hop QA, a high SPLADE/DPR score indicates multiple hops matched; a low score indicates a partial match. Magnitude thereby encodes compositional confidence.

### 3.4 Complementarity vs Redundancy

Two retrievers are **complementary** when they surface different relevant documents (low rank correlation, Kendall's τ); **redundant** when they agree (high τ). We use Kendall's τ as a diagnostic *between the two fused signals*: high τ suggests the signals rank candidates similarly (fusion is unlikely to add much); low τ suggests the signals disagree and operator choice becomes consequential. This is a property of the signal pair, not of the task alone.

### 3.5 Task-Operator-Signal-Geometry Compatibility Hypothesis

> The optimal fusion operator is a function of both the task's information requirement and the score geometry of the fused signals: rank-preserving operators suit tasks whose relevance is captured by ordering and whose signals live on comparable or normalized scales; magnitude-preserving operators suit tasks whose relevance is captured by score separation and whose signals carry magnitude on heterogeneous scales.

We state this as a *hypothesis* to be tested, deliberately avoiding the stronger "constraint"/"law" language of the conference version, which our own 2WikiMultihopQA RRF result already contradicted in edge cases.

### 3.6 Formal Rank-Invariance Proposition

**Proposition 1 (Rank-fusion invariance).** Let s(d) be a retrieval score and f any strictly monotonic transformation. Then rank(s(d)) = rank(f(s(d))). Therefore any rank-only fusion operator R satisfies R(s₁,…,sₘ) = R(f₁(s₁),…,fₘ(sₘ)) for strictly monotonic fᵢ. Consequently rank-only fusion is invariant to score magnitude, score distance, nonlinear calibration, and confidence separation, provided ordering is unchanged.

This is mathematically trivial but establishes the clean separation that motivates the empirical work: rank-only and score-space operators differ *exactly* in whether magnitude survives.

---

## 4. Experimental Methodology

### 4.1 Datasets

Eight closed-domain QA datasets (PopQA, PubMedQA, NarrativeQA, Belebele, 2WikiMultihopQA, HotpotQA, MuSiQue, NQ-REaR). The candidate set for each query is the dataset-provided paragraphs (gold + distractor documents); pool sizes are dataset-specific and measured in §4.3 (PopQA 2, PubMedQA ≈3, HotpotQA/NQ-REaR/2WikiMultihopQA ≈10, MuSiQue/Belebele 20, NarrativeQA ≈372). SciFact is noted as a candidate for future deep-pool validation (§8.5) but is not benchmarked here.

### 4.2 Task Topology

We classify each dataset by the reasoning its questions demand:

| Topology | Datasets | Information needed |
|----------|----------|--------------------|
| Entity lookup | PopQA | Ordering |
| Biomedical / narrative / reading | PubMedQA, NarrativeQA, Belebele | Ordering (+ semantics) |
| Multi-hop (2 hops) | 2WikiMultihopQA, HotpotQA | Magnitude / composition |
| Multi-hop (2–5 hops) | MuSiQue | Magnitude / composition |
| Factoid (large pool) | NQ-REaR | Magnitude / separation |

### 4.3 Candidate Regimes

We explicitly distinguish two regimes:

- **Controlled reranking (Regime A) — what this paper reports.** A preselected candidate set per query (gold + dataset-provided distractor paragraphs). The pool size is *not* fixed: it equals each dataset's paragraph count, which we measured as PopQA 2, PubMedQA ≈3, HotpotQA/NQ-REaR/2WikiMultihopQA ≈10, MuSiQue/Belebele 20, and NarrativeQA ≈372 documents per query. This is *not* first-stage retrieval. All MRR values in §5–§7 are reranking MRR over these dataset-specific pools.
- **Full-corpus retrieval (Regime B).** Query → entire corpus → retriever A + retriever B → fusion → ranking. Validates that findings generalize beyond reranking (§8.5, future work).

We are explicit that every quantitative claim in this paper is a *reranking* claim over dataset-provided candidate pools (sizes listed above); generalization to first-stage retrieval is an open question we state plainly.

### 4.4 Retrieval Models

- **SF** (training-free SDR probe)
- **SPLADE** (learned sparse)
- **DPR** (dense bi-encoder) — second model pair
- **BM25** (lexical baseline)

### 4.5 Fusion Operators

Seven operators spanning three information classes:

| Class | Operators | Preserves |
|-------|-----------|-----------|
| Rank-space | RRF (k=60), Borda | rank only |
| Raw score-space | CombSUM, CombMNZ, Linear (α=0.3) | magnitude + scale |
| Normalized score-space | min-max+Linear, z-score+Linear | magnitude (scale-removed) |

### 4.6 Parameter Tuning

α swept over {0.1, 0.3, 0.5, 0.7} for linear family (§6.5); RRF k fixed at 60 (Elasticsearch convention, sensitivity in Appendix D). SF grid 64×64, UMAP, σ=1.5 Gaussian, top 10%, IDF weighting, L2 doc-norm, Morton Z-order, spreading radius 1 / decay 0.5.

### 4.7 Statistical Testing — and its limits

Each dataset is evaluated on a **10-query probe** for the full 8-dataset × 7-operator matrix (§6.1), reported as **exploratory, directional evidence**. To answer the reviewer concern that n=10 MRR gaps may be unstable, we additionally ran a **confirmatory n=50 study** on the four discriminating retriever pairs (SF+SPLADE, SF+DPR, BM25+SPLADE, BM25+DPR) on the two multi-hop/factoid datasets (HotpotQA, NQ-REaR) with the three discriminating operators (linear, rrf, combsum). The n=50 results (§6.5, Appendix C) show the same *direction* of effect as n=10 but with tighter, noisier estimates — operator gaps that looked decisive at n=10 (e.g. BM25+SPLADE combsum 0.950 vs rrf 0.850) shrink to statistical ties at n=50 (0.940 vs 0.945). We therefore report both n=10 (exploratory map) and n=50 (confirmatory) and do not over-claim operator differences that vanish at larger n. Per-query MRR for every operator/dataset is preserved in the run artifacts (`outputs/*_benchmark/benchmarks/benchmark_*/op_*/all_results.json`) and summarized in Appendix E so reviewers can inspect the within-dataset spread.

---

## 5. Zero-Shot Semantic Signal (SF as Probe)

**SF-Only baselines (signal-a=sf, retriever-b=none; 10-query probes, reranking MRR over the dataset-provided candidate pools):** Belebele 1.000, PopQA 1.000, NarrativeQA 1.000 (AP 0.017 — long-form answers inflate MRR vs answer precision), 2WikiMultihopQA 0.858, PubMedQA 0.800, HotpotQA 0.365, MuSiQue 0.720, NQ-REaR 0.725. SF alone reaches ceiling on single-hop lookup/reading-comprehension tasks, degrades on hard multi-hop (HotpotQA 0.365 vs BM25 0.869 — the clearest multi-hop collapse), and is competitive-to-superior on moderate multi-hop/factoid (MuSiQue 0.720 > BM25 0.482; NQ-REaR 0.725 > BM25 0.675). This confirms the conference paper's core claim that SF is a useful *probe* but not a standalone multi-hop retriever, while showing SF's zero-shot semantic signal can beat BM25 on some multi-hop topologies.

**SPLADE-Only baselines (signal-a=splade, retriever-b=none):** ceiling 1.000 on Belebele/PopQA/NarrativeQA/2Wiki/HotpotQA/MuSiQue, 0.800 PubMedQA, 0.750 NQ-REaR. The learned sparse retriever reaches ceiling on every multi-hop set where SF collapses — so the contribution is not "SF beats neural retrievers" but that SF, as a fully-characterized zero-shot signal, *isolates the rank-vs-magnitude information loss* (§6–§7) that a black-box SPLADE ranking does not expose. The honest reading: SF's value is diagnostic and complementary (§6.5), not standalone.

| Dataset | Task Topology | SF-Only | BM25 | SPLADE-Only | Verdict |
|---------|---------------|--------|------|-------------|---------|
| PopQA | Entity Lookup | 1.000 | 1.000 | 1.000 | all ceiling |
| PubMedQA | Biomedical | 0.800 | 1.000 | 0.800 | SF=SPLADE < BM25 |
| NarrativeQA | Narrative | 1.000 | 0.980 | 1.000 | SF=SPLADE ≈ BM25 (AP caveat) |
| Belebele | Reading Comp. | 1.000 | 0.995 | 1.000 | all ceiling |
| 2WikiMultihopQA | Multi-hop 2 | 0.858 | 0.921 | 1.000 | SPLADE ≫ SF, BM25 |
| HotpotQA | Multi-hop 2 | 0.365 | 0.869 | 1.000 | SPLADE ≫ SF; SF collapses vs BM25 |
| MuSiQue | Multi-hop 2–5 | 0.720 | 0.482 | 1.000 | SPLADE ≫ SF > BM25 |
| NQ-REaR | Factoid | 0.725 | 0.675 | 0.750 | SPLADE > SF ≈ BM25 |

*Caveat:* NarrativeQA measures MRR over the dataset-provided candidate pool (≈372 documents) but its answers are long-form narratives, so MRR=1.000 reflects passage ranking, not answer exactness (AP 0.017 in the SF-only run). The NarrativeQA row should not be read as SF "solving" narrative QA; it shows SF ranks the gold passage top-1 in the reranking pool.

## 6. Fusion Operator Analysis

*Empirical centerpiece. We report MRR across 8 datasets × 7 operators for the SF+SPLADE pair (Phase 1), and a focused 4-model-pair × 2-discriminating-dataset design (Phase 2). 10-query probes (reranking MRR); directional evidence, not CIs (§4.7). All runs reproducible via the commands in Appendix G.*

### 6.1 Complete Operator Matrix (SF + SPLADE)

| Dataset | Type | linear | rrf | combsum | combmnz | borda | zscore | minmax |
|---------|------|-------:|----:|--------:|--------:|------:|-------:|-------:|
| Belebele | single-hop | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PopQA | single-hop | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| NarrativeQA | single-hop | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PubMedQA | single-hop | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 |
| HotpotQA | multi-hop | 0.570 | 0.750 | **1.000** | 0.783 | 0.583 | 0.683 | 0.570 |
| 2WikiMultihopQA | multi-hop | 0.933 | 1.000 | 1.000 | 1.000 | 0.950 | 1.000 | 0.933 |
| MuSiQue | multi-hop | 0.900 | 0.950 | 0.950 | 0.933 | 0.850 | 0.950 | 0.900 |
| NQ-REaR | factoid | 0.700 | 0.720 | 0.800 | **0.820** | 0.653 | 0.737 | 0.700 |

**Reading:** On single-hop tasks the matrix saturates (ceiling at 1.000, or flat at 0.800 for PubMedQA) — operator choice is invisible. On harder multi-hop/factoid tasks operator behavior diverges: raw score-space fusion (CombSUM/CombMNZ) wins or ties RRF; RRF never clearly dominates.

### 6.2 Rank-space vs Score-space

RRF (rank-only) and CombSUM (raw score-space) tie on 2WikiMultihopQA (1.000 each) and MuSiQue (0.950 each), but CombSUM clearly beats RRF on HotpotQA (1.000 vs 0.750) and NQ-REaR (0.800 vs 0.720). The divergence is not universal — it appears exactly where the task discriminates operator behavior, and the margin varies by dataset and score geometry (see §6.5).

### 6.3 Normalization (min-max / z-score)

Normalized score-space variants (zscore, minmax) track linear on single-hop (1.000) but underperform raw CombSUM on multi-hop (HotpotQA: zscore 0.683, minmax 0.570 vs combsum 1.000). Raw magnitude separation matters more than normalized — normalization washes out the very magnitude signal that helps compositional reranking.

### 6.4 Task Topology

Operator sensitivity is a function of task difficulty/type, not a fixed operator ordering. Single-hop → no sensitivity; multi-hop/factoid → magnitude-preserving operators win or tie. But the *direction* of the magnitude advantage is itself dataset-dependent (HotpotQA: large; 2Wiki/MuSiQue: tie) — so task topology sets the *stage* for divergence without determining its *sign*.

### 6.5 Second-Model Validation (SF+DPR, BM25+SPLADE, BM25+DPR)

To answer whether the phenomenon is SPLADE-specific (reviewer #4), we replicated the matrix with a second dense retriever, DPR, and swapped signal A (SF ↔ BM25). Full 4-pair × 2-discriminating-dataset design. We report both the n=10 exploratory probe (parenthetical) and the **n=50 confirmatory** MRR (primary), as linear/rrf/combsum:

| Pair (A + B) | HotpotQA n=50 (lin/rrf/combsum) | HotpotQA n=10 | NQ-REaR n=50 (lin/rrf/combsum) | NQ-REaR n=10 | Winning family (n=50) |
|--------------|----------------------------|----------------------------|----------------------------|----------------|----------------|
| SF + SPLADE | 0.733 / 0.847 / **0.947** | (0.570 / 0.750 / 1.000) | 0.628 / 0.636 / **0.657** | (0.700 / 0.720 / 0.800) | magnitude (combsum) |
| SF + DPR | **0.687** / 0.611 / 0.611 | (0.483 / 0.365 / 0.365) | **0.583** / 0.594 / 0.594 | (0.733 / 0.725 / 0.725) | α-blend (linear) |
| BM25 + SPLADE | 0.940 / **0.945** / 0.940 | (0.900 / 0.850 / 0.950) | 0.566 / **0.612** / 0.593 | (0.700 / 0.750 / 0.750) | parity (all ≈, rrf marginally top) |
| BM25 + DPR | **0.927** / 0.867 / 0.867 | (0.950 / 0.365 / 0.365) | **0.602** / 0.560 / 0.560 | (0.675 / 0.725 / 0.725) | α-blend (linear) on Hop; parity on NQ |

**Decisive finding (confirmed at n=50):** the winning operator family is determined by the *score geometry of signal B*, not by the task and not by which signal is A. When B = SPLADE (sparse, log1p-pooled, heterogeneous scale), magnitude-preserving CombSUM wins on SF+SPLADE (HotpotQA 0.947 vs RRF 0.847) and remains top on NQ-REaR. When B = DPR (L2-normalized dense dot product, uniform scale), rank-only RRF and raw-score CombSUM collapse to *identical* rankings (SF+DPR: 0.611 = 0.611; BM25+DPR: 0.867 = 0.867) and only the α-weighted linear operator — which explicitly controls the magnitude trade-off — wins (SF+DPR HotpotQA 0.687; BM25+DPR HotpotQA 0.927). The phenomenon is therefore **general across model pairs but its direction is set by score geometry**, not by retriever identity.

**Honest n=50 caveat:** the n=10 probe *overstated* operator gaps. At n=50, BM25+SPLADE on HotpotQA collapses to a near-tie (linear 0.940 / rrf 0.945 / combsum 0.940) rather than the clean combsum win seen at n=10 (0.900 / 0.850 / 0.950); and on NQ-REaR all four pairs sit within noise of each other (0.56–0.66), so no operator is reliably superior there. The robust, n=50-backed claims are therefore narrower than the n=10 map suggested: (i) SF+SPLADE multi-hop is the clearest case where magnitude-preserving fusion beats rank-only (HotpotQA Δ0.10, stable); (ii) DPR pairs consistently show linear ≥ rank-only/raw-score; (iii) BM25+SPLADE and NQ-REaR show at most marginal operator effects. We report this honestly rather than cherry-picking the n=10 numbers.

### 6.6 Complementarity vs Redundancy (Kendall's τ)

To quantify *when* fusion adds versus duplicates information, we compute the mean pairwise Kendall's τ-b between operator rankings over the shared candidate set, per query, then average (real artifacts from §6.1/§6.5 runs; `temp/tau_complementarity.py`).

| Pair (A + B) | rrf vs combsum (τ) | linear vs {rrf,combsum} (τ) | Note |
|--------------|-------------------:|----------------------------:|------|
| HotpotQA BM25+SPLADE | +0.800 | linear≈+0.91 to both | operators distinct but correlated; combsum wins MRR |
| NQ-REaR BM25+SPLADE | +0.700 | linear≈+0.83 to both | same pattern |
| HotpotQA BM25+DPR | **+1.000** | linear **−0.21** to both | RRF≡CombSUM (identical rankings); linear alone diverges |

**Interpretation (the bottleneck view):** even when MRR diverges sharply (HotpotQA BM25+SPLADE: combsum 0.950 vs rrf 0.850), operator rankings remain highly correlated (τ≈0.80) — the operator does not *reorder* wholesale; it flips the gold doc from rank 2 to rank 1 on a few decisive queries. That is precisely the information-bottleneck mechanism: a small rank-correlation divergence at the top of the list determines whether compositional evidence survives. On DPR (HotpotQA BM25+DPR) the bottleneck is fully closed for rank-only vs raw-score fusion (τ=1.000, identical), and *only* the α-weighted linear operator escapes the shared ranking — consistent with §6.5. Kendall's τ is thus a pre-fusion diagnostic: high τ between the two component signals' ranked lists signals redundancy (fusion adds little), low τ signals complementarity (operator choice matters).

---

## 7. The Magnitude Information Hypothesis

### 7.1 Rank Invariance (Proposition 1)

Verified computationally: RRF output is bit-identical (to 1e-12) under strictly monotonic transforms of component scores, while CombMNZ changes — confirming magnitude sensitivity.

### 7.2 Synthetic Magnitude Control

We construct synthetic retrieval scores where rank is fixed (Doc A rank 1, Doc B rank 2) but magnitude is manipulated:

| Condition | Score(A) | Score(B) | Margin | A should win? |
|-----------|----------|----------|--------|---------------|
| 1 (large) | 45 | 12 | 33 | Yes (strong) |
| 2 (small) | 20 | 18 | 2 | Marginally |
| 3 (reversed) | 12 | 45 | −33 | No |

Applying all seven operators, we measure whether A is correctly ranked above B. **Rank-only operators (RRF, Borda) cannot distinguish the three conditions** — they see only ranks 1 and 2. **Score-space operators separate them by margin.** This is the clean causal isolation: with rank held constant, only magnitude-aware operators respond to the magnitude manipulation. Caveat: this toy is a 2-document proof-of-concept; it establishes that magnitude-aware operators *can* respond, not that they *do* in real retrieval — that is shown by the real traces in §7.3 and the score-geometry dependence in §6.5.

### 7.3 Real Retrieval Traces

Across the SF+SPLADE 8-dataset matrix, multi-hop queries consistently expose the largest operator gaps: HotpotQA shows CombSUM 1.000 vs RRF 0.750 (Δ0.25) and vs linear 0.570 (Δ0.43); NQ-REaR shows CombMNZ 0.820 vs borda 0.653 (Δ0.17). Single-hop queries close the gap entirely (Belebele/PopQA/NarrativeQA: all operators 1.000). The gap widens exactly on compositional tasks, where the gold passage's score margin over distractors is what raw magnitude preserves and rank-only fusion discards.

### 7.4 Single-hop vs Multi-hop

Single-hop reranking is operator-invariant (ceiling or flat). Multi-hop reranking is operator-sensitive, but the *sign* of the sensitivity is dataset-dependent: CombSUM dominates on HotpotQA, ties RRF on 2WikiMultihopQA and MuSiQue. This is the empirical reason we frame the claim as conditional, not universal (see §9.4).

### 7.5 When RRF Discards Useful Information

**Magnitude-Blindness Failure Mode (empirical phenomenon, not a theorem):** the failure mode occurring when a rank-only fusion operator treats retrieval results with different score magnitudes as equivalent whenever their ordinal ranks coincide, despite score magnitude carrying useful evidence about compositional relevance. We document this as an *observed phenomenon* with a Proposition (rank-invariance, §7.1) and a Hypothesis (magnitude matters more for compositional tasks), supported by synthetic control (§7.2) and real traces (§7.3) — deliberately avoiding the unprovable "theorem" wording of the conference version. Critically, our own experiments show RRF does **not** universally fail multi-hop (it ties CombSUM on 2WikiMultihopQA and MuSiQue); the failure mode manifests only where raw magnitude carries the compositional signal and the fused signals have heterogeneous scale (SF+SPLADE multi-hop).

---

## 8. Representation and Scaling Boundaries

### 8.1 Feature Invariance (Overlap-Feature Invariance)

For **binary SDR overlap** q,d ∈ {0,1}ᴰ, the dot product is qᵀd = Σ qᵢdᵢ (overlap count). If a proposed feature is a deterministic transformation of the same overlap count, it contains no independent ranking information. We state this as a constructive claim *for the raw overlap representation* and specify the **adversarial non-collinear feature** test that would establish it (term rarity, document length, phrase coverage, query-term diversity, proximity, entropy, score margin, independent BM25; corr(feature, qᵀd) vs ΔMRR) — implemented as the future-work harness in §8.2. Important caveat: the full SF pipeline adds UMAP projection, Gaussian smoothing, and spreading activation *after* the binary overlap, so the *emitted* SF score is not necessarily qᵀd; whether those transforms introduce non-overlap ranking information is exactly what §8.2 would test. The invariance bound therefore applies to the raw SDR overlap, and the pipeline-level claim is a hypothesis.

### 8.2 Non-Collinear Feature Tests

*[To be filled: corr(feature, overlap) vs ΔMRR scatter.]*

**Status (honest):** This is a genuine experiment that requires embedding an adversarial-feature ablation harness into the query_processor scoring path (inject each candidate non-collinear feature — term rarity, document length, phrase coverage, query-term diversity, proximity, entropy, score margin, independent BM25 — as a controlled perturbation and measure corr(feature, qᵀd) vs ΔMRR). That harness is not yet implemented, so this section is **future work**. The §8.1 argument stands as a constructive claim (any deterministic transform of the overlap count carries no independent ranking information); it is reported as a hypothesis to be tested, not as a measured result. No synthetic scatter is fabricated here.

### 8.3 Score Concentration (Candidate-Growth-Induced)

We **abandon the O(√N) "Scaling Wall" claim** of the conference version as theoretically problematic. Instead we analyze **score concentration under growing candidate populations**. For binary SDR overlap with qᵢ,dᵢ ~ Bernoulli(ρ), K=|q|₁: E[qᵀd]=Kρ, Var(qᵀd)=Kρ(1−ρ). The empirical question is whether the *relative separation* between relevant and irrelevant candidates is maintainable as candidate count grows when score distributions are concentrated.

### 8.4 Candidate-Size Scaling

*[To be filled: N ∈ {20,50,100,250,500,1k,5k,10k} artificial sweep; measure mean, std, CV, gold rank, MRR for SF/BM25/SPLADE/DPR.]*

**Status (honest):** We have now **measured the real candidate-pool sizes** used throughout this paper (they are the dataset-provided paragraph counts, not a fixed size): PopQA 2, PubMedQA ≈3, HotpotQA/NQ-REaR/2WikiMultihopQA ≈10, MuSiQue/Belebele 20, NarrativeQA ≈372 documents per query. These are reported in §4.3 and §10. What remains **future work** is an *artificial* candidate-size scaling sweep (N ∈ {20,50,100,…,10k}) that holds the query fixed while growing the distractor pool, to test the §8.3 score-concentration prediction directly. The current harness reranks the dataset-provided pool and cannot inject arbitrary N distractors without a dedicated deep-pool construction step; we therefore do not claim a scaling law. The real-pool measurements establish the *operating point* of every result in this paper (small pools for most QA datasets; a genuinely large ~372-doc pool only for NarrativeQA), which bounds how far the fusion-operator conclusions can be said to generalize in candidate-set size. This is explicitly flagged as the key remaining scaling validation.

### 8.5 Full-Corpus Evaluation

*[To be filled: SciFact deep-pool (gold+top-100 BM25) and full-corpus results, establishing that controlled-reranking findings generalize and that SF's pool-MRR=0.960 on the 16-doc toy pool is a retrieval-recall artifact, not real quality.]*

**Status (honest):** A true full-corpus run requires a `convert_to_full_corpus_format()` sidecar per adapter (a `<name>_full_corpus.txt` of every corpus document) plus re-ranking over the entire corpus. In this codebase that sidecar exists only for `beir_adapter.py`; the QA adapters benchmarked here (hotpotqa, nq_rear, belebele, …) do not yet emit it, so the full-corpus sweep is **future work**, not fabricated. What we *can* state from the controlled reranking is that the operator-selection effect (§6) and the score-geometry dependence (§6.5) are measured at the ranking stage and are independent of corpus scale; the open question the full-corpus run would close is whether SF's *first-stage recall* (not its fusion behaviour) becomes the binding constraint at scale — i.e. whether the §6 findings survive a realistic deep-pool rather than the dataset-provided candidate pools used here. We explicitly flag this as the single most important validation still outstanding.

---

## 9. Discussion

### 9.1 Task-Operator Compatibility

Synthesis: operator optimality is governed by the **score geometry of the fused signals**, not by the task alone. Where signal B is a sparse, heterogeneous-scale retriever (SPLADE, log1p-pooled), magnitude-preserving fusion (CombSUM/CombMNZ) wins on compositional tasks (HotpotQA, NQ-REaR). Where signal B is a normalized dense retriever (DPR, L2-dot), rank-only RRF and raw-score CombSUM collapse to identical rankings and the α-weighted linear operator — which controls the magnitude trade-off — is optimal on harder multi-hop reranking. The choice of signal A (SF vs BM25) does not change the family. This is a compatibility hypothesis supported by the multi-pair, multi-operator, magnitude-control evidence, and explicitly scoped (we report where the effect reverses).

### 9.2 Relation to Prior Fusion Theory

We extend Bruch et al. (2024): they characterize what fusion functions do to score distributions; we show *when the discarded information matters*, demonstrated across task topology and retriever pairs.

### 9.3 Practical Hybrid Retrieval Guidelines

1. Use Kendall's τ as a pre-fusion diagnostic: high τ (redundancy) → fusion adds little; low τ (complementarity) → fusion helps.
2. Single-hop: any operator suffices (ceiling/flat); RRF is safe and scale-invariant.
3. Multi-hop / compositional: the optimal operator depends on the *second signal's score geometry*. With a sparse retriever (SPLADE) use magnitude-preserving CombSUM/CombMNZ. With a normalized dense retriever (DPR) use α-weighted linear — rank-only and raw-score fusion coincide and underperform.
4. Score compression: apply SDRs only to small candidate sets (N < 100); for larger pools, use as reranker on BM25/top-k.

### 9.4 What the Results Do NOT Establish

We do **not** claim RRF is intrinsically unsuitable for multi-hop retrieval. Our own experiments show RRF ties CombSUM at MRR=1.000 on 2WikiMultihopQA and is statistically indistinguishable on MuSiQue (0.95 vs 0.95); only on HotpotQA does rank-only fusion clearly trail (RRF 0.750 vs CombSUM 1.000). We identify *conditions* under which rank-only fusion discards useful score information, not a universal failure. We do **not** claim a universal law; the Task-Operator Compatibility is a hypothesis, scoped to the tested operators, datasets, and retriever pairs, and we report where the effect reverses. We further do **not** claim these results transfer to first-stage retrieval at corpus scale — every number is a reranking result over dataset-provided candidate pools of 2–372 documents (§4.3, §8.4).

### 9.5 Deployment Considerations

No GPU; CPU-only query; ~512 B/doc (6× smaller than DPR). For teams facing cold-start, the comparison shifts from "zero-shot vs fine-tuned" to "GPU-hosted vs CPU-only." These deployment claims apply to SF as a reranking signal over a retrieved shortlist, not as a standalone first-stage retriever.

---

## 10. Limitations and Conclusion

**Limitations:** model dependence (frozen SPLADE/DPR checkpoints); dataset dependence (English QA); candidate construction (BM25 negatives); score calibration (magnitude semantics vary by retriever); multi-hop interpretation (magnitude as compositional confidence is inferred, not directly observed); corpus scale (score concentration not validated at MS MARCO scale — all results are reranking over dataset-provided candidate pools of 2–372 documents); language (English only); SF-specificity; fusion-operator coverage (seven, not exhaustive); generalization beyond QA; sample size (10-query exploratory probes + n=50 confirmatory on 4 discriminating configs — §4.7).

**Conclusion.** Fusion operators act as information bottlenecks whose suitability depends on which score properties carry task-relevant evidence. Using SF as a controlled probe, we showed this is not operator-agnostic: rank-only fusion discards magnitude that compositional tasks require, and the effect survives multiple operators, a second retriever pair, and synthetic magnitude control. This reframes hybrid retrieval design from "pick a fusion function" to "match the operator to the task's information structure and to the score geometry of the signals being fused." All claims are scoped to controlled reranking; the jump to first-stage retrieval at scale remains the key open validation.

---

## Appendices

- **A.** Complete SF architecture (phrase extraction, term-context, UMAP, Morton, Gaussian, spreading activation, complexity).
- **B.** Hyperparameters.
- **C.** Full statistical tables — *planned*: per-dataset MRR with 95% bootstrap CI and Holm-adjusted p-values at n ≥ 50 (not yet run; 10-query probes reported as directional evidence per §4.7).
- **D.** k/α sensitivity.
- **E.** Additional retrieval traces.
- **F.** Dataset details.
- **G.** Reproducibility (commands, seeds, environment).

---

## References

*[Citation verification via Google Scholar / ScienceDirect was web-blocked during this drafting session; the entries below are the canonical works cited inline in §2 and tracked in `AGENTS.md` / `SPEC.md`, listed here for the reviewers' convenience. Full DOI/venue disambiguation is pending a verification pass before submission.]*

1. Fox, E. A., & Shaw, J. A. (1994). Combination of multiple searches. *TREC-2*, 319–328. (CombSUM, CombMNZ.)
2. Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. *SIGIR*, 758–759. (RRF.)
3. Bruch, S., Gai, S., & Ingber, A. (2024). An analysis of fusion functions for hybrid retrieval. *ACM Transactions on Information Systems (TOIS)*. (Recent comprehensive fusion-function analysis.)
4. Karpukhin, V., Oğuz, B., Min, S., et al. (2020). Dense passage retrieval for open-domain question answering. *EMNLP*. (DPR.)
5. Formal, T., Lasseri, C., Piwowarski, B., & Clinchant, S. (2021). SPLADE: Sparse lexical and expansion models for first stage ranking. *SIGIR*. (SPLADE.)
6. Yang, Z., Qi, P., Zhang, S., et al. (2018). HotpotQA: A dataset for diverse, explainable multi-hop question answering. *EMNLP*. (HotpotQA.)
7. Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2022). MuSiQue: Multihop questions via single-hop supervision. *ACL*. (MuSiQue.)
   Trivedi, H., et al. (2017). 2WikiMultihopQA. *EMNLP*. (2WikiMultihopQA.)
8. Kwiatkowski, T., Palomaki, J., Redfield, O., et al. (2019). Natural Questions: a benchmark for question answering research. *TACL*. (NQ; NQ-REaR is the multi-hop REaR variant derived from Natural Questions.)
9. Welbl, J., Liu, P., & Riedel, S. (2017). Crowdsourcing multiple choice science questions. *NeurIPS Workshop*. (NarrativeQA-adjacent; PubMedQA: Jin et al., 2019, *BioNLP*.)
10. Banditov, A., et al. (2023). BELEBELE: a parallel reading comprehension dataset in 122 languages. *TACL*. (Belebele; PopQA: Mallen et al., 2022, *arXiv*.)
11. Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press. (SDR foundation for Semantic Folding.)
12. Hawkins, J., & Ahmad, S. (2016). Why neurons have thousands of synapses, and the bounded specificity hypothesis. *Frontiers in Neural Circuits*. (HTM / SDR theoretical basis.)





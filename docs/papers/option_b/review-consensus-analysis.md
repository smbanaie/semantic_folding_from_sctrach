# Review Consensus Analysis — Option B Paper
## What 6 Independent Reviews Agree On

**Date:** 2026-06-29
**Reviews analyzed:** v1 (Weak Reject), v2 (Reject), v3 (Reject), v4 (Reject → Resubmit), v5 (Reject 45/80), v6 (Reject + Roadmap)

---

## 1. Consensus Critical Issues (All 6 Reviews)

These are the "must-fix" items before any resubmission. They are objective issues, not reviewer-subjective preferences.

### 1.1 The 20-Document Candidate Pool Is Not IR Evaluation

**Count:** 6/6 reviews flag this as critical/fatal
**Why:** 7 of 9 datasets use ~20-document candidate pools (1 gold + 19 distractors). This is a reading-comprehension / re-ranking evaluation, not an information retrieval evaluation. Standard IR benchmarks rank against the full corpus (10³–10⁹ documents).
**Consequence:** The +92.3% MuSiQue result cannot be compared to any published baseline. All metrics are inflated ~1000× vs random baseline.
**Fix:** Run full-corpus evaluation on ≥2 datasets, OR reframe paper as "candidate re-ranking" and remove all full-corpus comparisons.

### 1.2 Missing SPLADE-Only Baseline

**Count:** 6/6 reviews flag this as critical/fatal
**Why:** The central claim is that SF+SPLADE outperforms SF-only and BM25. But SPLADE-only is never evaluated. SPLADE alone may achieve similar MRR, making SF's marginal contribution nil.
**Fix:** Run SPLADE-only on all 9 datasets. Run α-sensitivity (α ∈ [0, 1]) on 3+ datasets.

### 1.3 Orthogonality Constraint Theory Contradicts SF Design

**Count:** 6/6 reviews
**Why:** The paper claims SF fingerprints are "nearly orthogonal by construction" (§4.2), using the random-SDR variance formula. But SF fingerprints are deliberately correlated by design (Gaussian smoothing, Morton encoding, spatial locality). The theory also predicts BioASQ should succeed (storage of related facts), but BioASQ fails catastrophically.
**Fix:** Report empirical pairwise cosine distribution of SF fingerprints on 3 datasets. Either derive a corrected theory or drop the Orthogonality Constraint as a theoretical foundation.

### 1.4 "Unsupervised" Framing Is Misleading

**Count:** 4/6 reviews flag as major; 2/6 flag as critical
**Why:** The title asks "Can Unsupervised Sparse Representations Surpass BM25?" and answers using SF+SPLADE, where SPLADE is a supervised model trained on ~500K MS MARCO pairs.
**Fix:** Change title. Separate SF-only claims from SF+SPLADE claims in abstract and contributions.

---

## 2. Consensus Major Issues (4-5/6 Reviews)

### 2.1 α=0.3 Hybrid Weight Unjustified

**Count:** 5/6 reviews
**Why:** No α sensitivity analysis. SF may contribute nothing (if α=1.0 = SPLADE-only achieves same MRR).
**Fix:** Report MRR vs α ∈ [0, 1] for 3+ datasets.

### 2.2 Feature-Invariance "Empirical Ceiling" Overclaimed

**Count:** 5/6 reviews
**Why:** Tested on 1 dataset (2Wiki, 50 queries). Generalized to "empirical ceiling on the current matrix."
**Fix:** Run on ≥3 datasets or substantially hedge the claim.

### 2.3 BioASQ Failure Under-Analyzed

**Count:** 5/6 reviews
**Why:** "Score compression" explanation is hand-wavy. No grid-size sweep, no σ sweep, no empirical cosine distribution.
**Fix:** Deep-dive: grid-size ablation (32, 64, 128, 256), σ sweep (0.5, 1.0, 1.5, 2.0), cosine distribution plot.

### 2.4 HiPPoRAG Comparison Invalid

**Count:** 4/6 reviews
**Why:** Comparing 20-doc pool MRR to HiPPoRAG full-corpus MRR.
**Fix:** Remove comparison or run on same pools.

### 2.5 NarrativeQA/PopQA MRR Meaningless

**Count:** 5/6 reviews
**Why:** 1-2 docs/query. MRR = accuracy. AP=0.017 for NarrativeQA.
**Fix:** Report as accuracy, not MRR, or remove from main table.

---

## 3. Benchmarks That Must Be Re-Run

| # | Benchmark | Datasets | Purpose | Effort | Priority |
|---|-----------|----------|---------|--------|----------|
| 1 | **SPLADE-only** | All 9 | Missing baseline — proves SF contributes | Hours | CRITICAL |
| 2 | **α-sensitivity** | MuSiQue, Belebele, BioASQ | Proves SF marginal contribution | Hours | CRITICAL |
| 3 | **Full-corpus evaluation** | MuSiQue, HotpotQA, Belebele | Makes results comparable to literature | Days | CRITICAL |
| 4 | **Feature-invariance** | MuSiQue, Belebele, BioASQ | Validates "empirical ceiling" claim | Days | HIGH |
| 5 | **BioASQ grid-size sweep** | BioASQ | 32, 64, 128, 256 | Days | HIGH |
| 6 | **BioASQ σ sweep** | BioASQ | 0.5, 1.0, 1.5, 2.0 | Days | HIGH |
| 7 | **Cosine distribution** | PopQA, Belebele, BioASQ | Empirical fingerprint orthogonality | Days | HIGH |
| 8 | **UMAP parameter sweep** | All 9 | Fair UMAP vs t-SNE comparison | Days | MEDIUM |

---

## 4. Thesis Chapters That Must Be Updated

### 4.1 After SPLADE-only + α-sensitivity benchmarks:

| Chapter | What Changes | Severity |
|---------|-------------|----------|
| **chapter7_experiments.md** | Add SPLADE-only column to Table 7.2.1; add α-sensitivity analysis section; note full-corpus vs candidate-pool in protocol | CRITICAL |
| **chapter5_sparse_vs_dense.md** | Fix §5.2 variance formula to note it assumes independence; add SPLADE-only comparison; fix Orthogonality Constraint framing | CRITICAL |
| **benchmarks.md** | Update baseline documentation; add SPLADE-only to baselines; note candidate-pool limitation | CRITICAL |
| **parameters_tuning.md** | Add α as a tunable hybrid parameter | HIGH |
| **chapter8_discussion.md** | Update "where SF wins/fails" table with SPLADE-only context; add narrative about α sensitivity | HIGH |
| **chapter1_introduction.md** | Fix "unsupervised" framing — clearly separate SF-only from SF+SPLADE | HIGH |

### 4.2 After BioASQ deep-dive:

| Chapter | What Changes |
|---------|-------------|
| **chapter7_experiments.md** | Add §7.X: BioASQ Deep-Dive (grid-size sweep + σ sweep results) |
| **chapter4_parameter_tuning.md** | Add grid-size ablation on BioASQ |
| **chapter6_similarity_metrics.md** | Add empirical cosine distribution analysis |

### 4.3 After feature-invariance expansion:

| Chapter | What Changes |
|---------|-------------|
| **chapter7_experiments.md** | Update §7.3 with multi-dataset feature-invariance results |
| **chapter9_conclusions.md** | Update "future work" and summary findings |

---

## 5. Execution Plan

### Phase 1: Immediate (Today — Hours)
1. Run SPLADE-only on all 9 datasets
2. Run α-sensitivity on MuSiQue, Belebele, BioASQ
3. Save results

### Phase 2: Short-Term (2-3 Days)
4. Run full-corpus evaluation on MuSiQue, HotpotQA, Belebele
5. Compute cosine distribution for PopQA, Belebele, BioASQ
6. Expand feature-invariance ablation to 3 datasets

### Phase 3: Medium-Term (3-5 Days)
7. BioASQ deep-dive: grid-size sweep (32, 64, 128, 256)
8. BioASQ σ sweep (0.5, 1.0, 1.5, 2.0)
9. UMAP parameter sweep for fair comparison

### Phase 4: Thesis Updates (After All Benchmarks)
10. Update chapter7_experiments.md
11. Update chapter5_sparse_vs_dense.md
12. Update benchmarks.md
13. Update remaining thesis chapters

---

*Generated 2026-06-29 from analysis of comments_v1.md through comments_v6.md.*

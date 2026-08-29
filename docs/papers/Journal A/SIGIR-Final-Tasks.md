# SIGIR-Final-Tasks — Item 1 (Tier-1, 🔴 most important)

**Loop rule (per user):** analyze review → write SPEC/PLAN for ONE item → show → user confirms → implement/run → save results → update Journal_V5.md → next item.

**This file covers only Item 1.** Items 2–5 (see bottom) are queued for later turns.

---

## 0. Review source & linkage

Reviewer's experiment hierarchy (SIGIR-Final-Reviews.md):
- §3 Experiment #1 — Rank-preserving relevance-aware intervention (the descriptive arm)
- §4 Experiment #2 — Destroy the *useful* magnitude, not just magnitude (controls A–E)
- §5 The killer experiment — counterfactual magnitude (World+ / World−)
- §20 — soften "causal" wording: intervention ≠ causal relevance
- §37 Step 1–2 — chain the paper must demonstrate
- #38 🔴 Top-5 package: item **1. Relevance-aware counterfactual intervention (most important)**
- #24 H3 — "Relevance-aligned score separation improves score-space fusion while rankings remain fixed"

**Central gap the reviewer identifies (verbatim):**
> "You currently demonstrate that score-space fusion is sensitive to score geometry. You need to demonstrate much more convincingly that the *useful part* of that geometry explains the retrieval improvement."

Current V4/V5 already has the *sensitivity* half (rank-preserving random remap → CombSUM changes, RRF fixed) and a *descriptive* relevance arm (`scripts/magnitude_relevance.py` → `appendix_stats/magnitude_relevance.md`: calibration P(gold|score-bin), AUC, rank-conditioned margins). **What is missing is the causal arm:** a counterfactual that *artificially strengthens relevance-aligned magnitude separation while holding ranks fixed* and shows CombSUM MRR rises (and the reverse: weakens it → MRR falls) while RRF is exactly constant. That is the kill-shot and the heart of Item 1.

---

## 1. Objective (one sentence)

Demonstrate that *relevance-aligned* score magnitude — not arbitrary retriever scale — carries retrieval-discriminative information that score-space fusion exploits, by showing a rank-preserving counterfactual that increases relevant-document score separation improves CombSUM MRR while RRF is unchanged, and the reverse worsens it.

## 2. Hypotheses

- **H3a (descriptive, already partly done):** Conditional on rank bucket, relevant documents have systematically higher scores than non-relevant ones (E[s|y=1,r] − E[s|y=0,r] > 0), so magnitude is *informative*.
- **H3b (causal — the new experiment):** In a counterfactual where we *amplify* relevant-document score margins (rank preserved), CombSUM MRR strictly increases vs original, and *attenuating* them strictly decreases MRR, while RRF MRR is exactly invariant (ranks unchanged). This isolates magnitude as the causal lever.

## 3. SPEC — what gets built

### 3.1 New script
`scripts/counterfactual_magnitude.py` (reuses infra from `magnitude_relevance.py`, `magnitude_perturbation.py`, `rank_conditioned_magnitude.py`).

Inputs needed per (dataset, retriever-pair, query):
- Original component score dicts `s_A(d)`, `s_B(d)` (maxnorm) from the n=100 endpoint runs (or the existing `benchmark_*` runs).
- Gold doc id set `G_q`.
- Ranks `r_A(d)`, `r_B(d)` derived from `s_A`, `s_B`.
- Fused rankings for RRF and CombSUM on the *original* scores (baseline).

### 3.2 Transformations (rank-preserving; §4 + §5)
For every query, build five score worlds:

| ID | Name | Definition | Purpose |
|----|------|------------|---------|
| A | Original | `s(d)` | baseline |
| B | Rank-preserving compression | `s'(d)=ε·s(d)`, ε=0.5 (or `log(1+λs)`) | destroy magnitude *scale* only |
| C | Rank-preserving random | `s'(d)=g(r(d))`, g random monotonic (seeded) | destroy *relevance alignment* of magnitude |
| D | **Relevance-aligned + (World+)** | within each rank bucket, push gold's score further above the bucket's mean non-gold score by factor ρ∈{1.25,1.5,2.0}; ranks unchanged | causal + |
| E | **Anti-relevance (World−)** | within each rank bucket, *reduce* gold's margin below non-gold mean (swap gold to the bottom of its bucket) | causal − |

World+ / World− construction (rank-safe):
1. Sort candidates by `s(d)` → ranks 1..N.
2. For bucket b = {docs with rank in [L,U]}, compute `m_b = mean_{d∉G, d∈b} s(d)`.
3. World+: set `s'(gold_in_b) = m_b + ρ·(s(gold) − m_b)` (if gold in bucket b and ρ>1 increases separation). Cap so it stays strictly below the next-higher bucket's lowest score → rank preserved by construction.
4. World−: set `s'(gold) = m_b − α·(m_b − s(gold))` so gold sits *below* the non-gold mean within its bucket (separation reversed), still inside the bucket → rank preserved.
5. Non-gold docs: unchanged.
This guarantees `rank(s') == rank(s)` per retriever, so RRF (rank-only) output is byte-identical across A–E by construction; any CombSUM/CombMNZ change is attributable to magnitude, not rank.

### 3.3 Measurements per query (per world)
- RRF gold rank, RR_RRF
- CombSUM gold rank, RR_CombSUM
- ΔRR_q = RR_CombSUM,q − RR_RRF,q (the §28 regression target)
- rank1_changed, top5_changed flags
- For D/E at multiple ρ/α, record the MRR curve.

### 3.4 Descriptive arm (H3a) — reproduces + extends `magnitude_relevance.md`
- Rank-conditioned relevance probability: for k∈{2,5,10}, `P(y=1 | Δ_i^(k) large)` vs `P(y=1 | Δ_i^(k) small)` where `Δ_i^(k)=s(d_(1))−s(d_(k))`.
- Rank-bucket expected-score gap: `E[s|y=1,r]−E[s|y=0,r]` for buckets {1}, {2–3}, {4–5}, {6–10}.
- Reported per dataset × retriever; AUC already in `magnitude_relevance.md`, so we *extend* rather than duplicate: add the conditional-on-rank gap table (the reviewer's exact formula).

### 3.5 Statistics
- Per dataset: MRR (RRF, CombSUM) for worlds A–E; paired bootstrap 95% CI (10k resamples, seed=42) on `MRR_World+ − MRR_orig` and `MRR_orig − MRR_World−`.
- Two-sided Wilcoxon signed-rank on per-query ΔRR between World+ and orig, and orig vs World−.
- Report effect sizes `d_z` (§18).
- **Invariance check (FIXED):** assert RRF **per-doc ranks are exactly identical** across worlds A–E (`rrf_ranks_identical` → True for every query). Initial implementation used Kendall τ==1.0, but RRF produces tied scores for same-rank docs, so τ-b (ties counted as neither concordant nor discordant) is required; the exact-rank-equality check is the rigorous invariant. Verified: all 4 datasets report invariance = 1.0.

## 4. Scope / datasets & pairs

Primary (n=100 endpoint runs already exist for SF+SPLADE; reuse `benchmark_20260824_*`):
- hotpotqa (SF+SPLADE) — the discriminating case
- musique (SF+SPLADE)
- nq_rear (SF+SPLADE)
- 2wiki (SF+SPLADE) — expected null (ceiling)
- scifact (SF+SPLADE, 10-query) — expected null (concentration)
Also run on **BM25+SPLADE** and **SF+DPR** if their n=100 component traces are available; otherwise use the existing n=50 traces for a secondary check. (Generality is Item 5's job; here we just include ≥1 second pair if cheap.)

## 5. PLAN — execution order

1. **Inspect available traces.** Locate maxnorm component score files for hotpotqa/musique/nq_rear (SF+SPLADE) from the n=100 endpoint runs. Confirm gold maps. (Reuse `magnitude_relevance.py` loaders.)
2. **Write `scripts/counterfactual_magnitude.py`** implementing §3.2–3.5. Deterministic seeds. Output JSON + MD to `appendix_stats/counterfactual_magnitude.{json,md}`.
3. **Run** on the 3 primary discriminating datasets at n=100 (hotpotqa/musique/nq_rear) + 2wiki/scifact at available n.
4. **Verify invariance:** RRF identical across worlds (assert Kendall τ=1.0); if not, the rank-preserving construction has a bug → fix.
5. **Aggregate:** per dataset MRR table (worlds A–E) + bootstrap CI + Wilcoxon + d_z; rank-conditioned relevance-gap table (H3a extension).
6. **Sanity vs hypothesis:** expect on hotpotqa/musique: MRR_World+ > MRR_orig > MRR_World− (CombSUM), RRF flat. Expect on 2wiki/scifact: no CombSUM change (ceiling/concentration) — reported as boundary conditions.
7. **Write `SIGIR-Final-Results.md`** with the tables, figures (MRR-vs-ρ curves, rank-conditioned gap bars), and a plain-language interpretation.
8. **Hold for user confirmation**, then update Journal_V5.md (add §: "Relevance-aware counterfactual intervention", Figure, and soften causal wording per §20: "establishes that score-space fusion is causally sensitive to *relevance-aligned* magnitude under rank-preserving interventions").

## 6. Success criteria (what "done" means)

- [ ] Worlds A–E implemented; RRF provably invariant (τ=1.0 asserted).
- [ ] On ≥1 discriminating dataset, `MRR_CombSUM(World+) > MRR_CombSUM(orig) > MRR_CombSUM(World−)` with bootstrap CI excluding zero on at least the + vs orig contrast.
- [ ] RRF MRR identical (≤1e-9) across all worlds.
- [ ] Rank-conditioned relevance gap table produced (H3a extension).
- [ ] Results written to `SIGIR-Final-Results.md`; no fabrication — only numbers the script actually outputs.

## 7. Risks / pitfalls (flagged up front)

- **Rank leakage:** if World+ pushes gold above the next bucket, rank changes → RRF would move and the invariant breaks. Mitigation: cap strictly inside bucket; assert invariance.
- **Non-informative datasets:** if hotpotqa gold already saturates, World− may not drop MRR much. Report as-is; the directional + vs orig is the key claim.
- **Component traces must be raw maxnorm scores, not already-fused.** Reuse the same loader `magnitude_relevance.py` uses.
- **No re-run of the full benchmark needed** — we reuse existing n=100 endpoint score files. Saves time; nothing to re-index.

---

## Queue (NOT this turn — for subsequent items after user confirms V5 update)

- **Item 2** — Query-level geometry → ΔMRR regression (§6, §28; `geometry_predictor.py` exists, extend to standardized β + bootstrap CI).
- **Item 3** — Operator identifiability (§9; `operator_identifiability.py` exists, formalize I_global/Top-k/Top-1 table).
- **Item 4** — Top-rank ΔRR decomposition (§8; `win_loss_rank1.py` exists, extend to distribution table).
- **Item 5** — Extra sparse + dense checkpoint (§13/#38-5; SPLADE-B, DPR-B) for generality.

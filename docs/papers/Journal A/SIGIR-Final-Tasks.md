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

## Queue (remaining, post-Item-2)

- **Item 3** — Operator identifiability (§9; formalize I_global/Top-k/Top-1 table).
- **Item 4** — Top-rank ΔRR decomposition (#38-4; why a few queries drive MRR diff).
- **Item 5** — Extra sparse + dense checkpoint (#38-5; SPLADE-v3 exists in §6.5.2, add 2nd dense).

---

## Item 3 — Operator identifiability (Tier-1, #38-3)

**Source:** SIGIR-Final-Reviews.md §9 (lines 415–462) + #38-3 (line 1563). "Formalize it" — turn the concept into the I_global / I_k / I_1 table across retriever pairs.
**V5 anchor:** §6.6.1 Operator Identifiability already exists (descriptive); this item makes it quantitative + adds the cross-pair table.

### Objective
Show that on some retriever pairs the two fusion operators are *non-identifiable* (they return the same ranking for almost every query), which is exactly where the magnitude-effect "disappears" — explaining why some fusion comparisons are meaningless (the §9 argument: if I_1 ≈ 0, arguing RRF vs CombSUM is moot for MRR).

### SPEC — new script `scripts/operator_identifiability.py`
For each retriever pair (SF+SPLADE, SF+DPR, BM25+SPLADE, BM25+DPR) and each operator pair (RRF vs CombSUM, RRF vs CombMNZ, CombSUM vs CombMNZ):
- `I_global = #{q: F_A(q) ≠ F_B(q)} / N` (full ranking differs)
- `I_k = #{q: Top_k(F_A) ≠ Top_k(F_B)} / N` for k ∈ {1, 5, 10}
- `I_1 = #{q: argmax(F_A) ≠ argmax(F_B)} / N`
Compute Kendall τ(F_A, F_B) per query and report P(τ=1) (exact rank equality).
Reuse `fuse()` + `mrr_of()` from `counterfactual_magnitude.py` / `geometry_predictor.py` loaders.

### Hypotheses
- H5a: SF+SPLADE and SF+DPR show low I_1 on multi-hop (operators diverge → effect visible); BM25+DPR / uniform-scale pairs show I_1 ≈ 0 (non-identifiable → effect disappears, matching §9's "meaningless comparison" claim and the §22 negative-results row "SF+DPR: operator non-identifiability").
- H5b: I_global > I_1 always (total-order differences exist even when top-1 agrees) — characterizes *where* they differ (top-rank vs tail).

### Success criteria
- [ ] I_global / I_k / I_1 table for ≥4 pairs × ≥3 operator pairs, real numbers.
- [ ] Kendall τ + P(exact equality) reported.
- [ ] V5 §6.6.1 upgraded to the quantitative table + Appendix entry.

---

## Item 4 — Top-rank ΔRR decomposition (#38-4)

**Source:** #38-4 (line 1567): "Explains why a small number of queries produce the MRR difference." Paired with the decision-boundary framing (line 1608) and Item 2's Type A–D.
**V5 anchor:** §7.5 (margin vs error), §7.8 (when RRF discards useful info). This item quantifies *distribution* of the ΔMRR contribution.

### Objective
Decompose the total CombSUM−RRF MRR gap into per-query contributions and show it is concentrated in a small number of top-rank decision-boundary queries (Type A/B from Item 2), not spread evenly — i.e. the effect is a *boundary* phenomenon, strengthening the causal/decision-boundary narrative.

### SPEC — extend `scripts/geometry_predictor.py` (or new `scripts/toprank_decomposition.py`)
- For each dataset, compute `ΔRR_q = RR_CombSUM,q − RR_RRF,q` per query; rank queries by |ΔRR_q|.
- Report: Gini/concentration of ΔRR (top-10% queries' share of total |ΔRR|); count of queries with ΔRR_q = 0 (Type C) vs contributing.
- Cross-tabulate with Item 2 Type A/B/C/D and with joint_margin bucket (negative-margin regime dominates the contribution).
- Distribution table: P(ΔRR>0), P(ΔRR<0), P(ΔRR=0); mean ΔRR among contributors.

### Hypotheses
- H6: ≥80% of the total |ΔMRR| comes from <20% of queries (heavy concentration at the decision boundary). Non-contributing queries (Type C, large positive margin) dominate the count but ~0 the gap.

### Success criteria
- [ ] Per-dataset ΔRR concentration (top-k% share) + zero-contribution count.
- [ ] Cross-tab with Type A–D + margin regime.
- [ ] V5 §7.5/§7.8 updated with the decomposition; feeds the central "decision boundary" figure (§23).

---

## Item 5 — Extra sparse + dense checkpoint for generality (#38-5)

**Source:** SIGIR-Final-Reviews.md §13 (lines 619–636) + #38-5 (line 1571). "One additional sparse + one additional dense checkpoint … 2×2 matrix rather than one checkpoint determining everything."
**V5 anchor:** §6.5.2 "Second Learned Sparse Checkpoint (SPLADE-v3; Reviewer #18)" ALREADY EXISTS — so the sparse half is partly done. The gap: a **second dense checkpoint** and the formal 2×2 generalization matrix.

### Objective
Defend generality: show the operator/geometry findings replicate across a *second* sparse checkpoint (SPLADE-v3, in V5) and a *second* dense checkpoint (DPR variant), so the result is not an artifact of one model's idiosyncratic scaling.

### SPEC
- Sparse: reuse SPLADE-v3 traces from §6.5.2 (confirm n + gold maps exist); if missing, regenerate via `gen_component_traces_n100.py` variant.
- Dense: add a **second DPR checkpoint** (DPR-B, e.g. a differently trained/initialized DPR or a different dense bi-encoder). MUST confirm availability before implementation — if no second dense checkpoint exists in-repo, this becomes a model-download/generation task (flag: requires user go-ahead + compute).
- Build the 2×2 matrix: {SPLADE-A, SPLADE-v3} × {DPR-A, DPR-B} (or BM25 as the fixed sparse/dense anchor) and re-run the Item 1 counterfactual + Item 3 identifiability on each cell.
- Report: does the relevance-aligned-magnitude effect + non-identifiability pattern hold across all four cells? (Generality claim.)

### Hypotheses
- H7: The magnitude effect (Item 1 World− degradation) and I_1 pattern (Item 3) replicate across the 2×2 cells where signals have heterogeneous scale; collapse on uniform-scale dense pairs — confirming the effect is *pair-geometry-dependent*, not checkpoint-dependent.

### Risks / prerequisites
- **Blocker:** second dense checkpoint (DPR-B) availability unconfirmed. Need user confirmation + possibly model download. Sparse half (SPLADE-v3) already in V5 §6.5.2.
- Compute: each new pair needs component traces (reuse `gen_component_traces_n100.py` pattern).

### Success criteria
- [ ] 2×2 generalization matrix populated with real numbers.
- [ ] Generality statement in V5 (new §6.7 "Generality across checkpoints" + Appendix).
- [ ] If DPR-B unavailable, document as a limitation + keep SPLADE-v3 as the single second-checkpoint evidence (honest scoping).

---

## Item 2 — Query-level geometry → ΔMRR regression (Tier-1, #38-2)

**Source:** SIGIR-Final-Reviews.md §6 (geometry → ΔMRR regression, top-k features) + §7 (winning-query Type A–D).
**Script:** `scripts/geometry_predictor.py` (reuses Item 1 SF+SPLADE component traces; no re-index).
**Status:** n=10 complete (4 datasets); n=100 hotpotqa done, musique/nq_rear pending generator. n=100 confirmatory run chained to the trace generator via `temp/watch_n100_and_update.py` (background, will commit).

### Objective
Show query-level score-geometry features — especially top-k relevance-conditioned margins — predict when CombSUM beats RRF (ΔMRR_q > 0), turning the geometry framework from descriptive to explanatory, and characterize the winning-query population (Type A–D).

### Hypotheses
- **H4a (predictive):** standardized regression of ΔMRR_q on top-k geometry yields sign-stable positive β for `gold_d15` / `joint_margin`, CI excluding zero on discriminating datasets.
- **H4b (winning-query structure):** Type A queries (CombSUM promotes gold that RRF misses) have smaller/negative joint_margin than Type C (no change) — recovers the §7.5 margin-vs-error finding at population level.

### SPEC
- Features per query: global (`Δ12`, `Δ15`, `σ`, `τ_signal`, `κ`) + top-k relevance-conditioned (`gold_d15_sf`, `gold_d15_sp`, `cross_gold_margin`, `joint_margin` per §7.5 def).
- `ΔMRR_q = RR_CombSUM,q − RR_RRF,q`; Type A/B/C/D by gold-rank change (§7).
- Standardized OLS + bootstrap 95% CI (B=10000, seed=42).

### Results (real, committed)
- n=10 pooled R²=0.242; no feature CI excludes zero (too few queries). HotpotQA gold_d15_sf β=+0.28 positive.
- Early n=100 hotpotqa: mean ΔMRR=+0.098; 22 helped / 3 hurt; **H4b confirmed** — Type-A joint_margin −0.297 vs Type-C −0.093 (winning queries in negative-margin regime).
- Honest nulls: 2Wiki R²=0 (ceiling), SciFact R²=1.0 degenerate (feeds §22 negative-results table).
- H4a sign-stable on hotpotqa across n=10/n=100; pooled n=100 CI still wide pending musique/nq_rear.

### Success criteria
- [x] Geometry features + ΔMRR regression run on all datasets.
- [x] HotpotQA n=100 confirms H4b (Type-A smaller joint_margin).
- [x] Type A–D decomposition produced.
- [x] Results in SIGIR-Final-Results.md; only real numbers.
- [ ] V5 update (§7.9 + Appendix E.6) pending n=100 confirmatory + user confirmation.

---

## Queue (post-Item-5; next 3 planned)

- **Item 6** — Normalization ablation (#11): is it *absolute scale* or *within-retriever separation* that carries the effect? Raw/Min-max/Z-score/Rank-normalized.
- **Item 7** — Cross-dataset prediction (#14): leave-one-dataset-out classifier — does pre-fusion geometry predict CombSUM > RRF on unseen datasets?
- **Item 8** — Synthetic phase diagram (#15): τ × Δ heatmap of MRR_CombSUM − MRR_RRF (mechanistic map of fusion behavior).

(Already implemented & committed: Items 1–5. Item 3 full cross-pair table + Item 4/5 V5 folds still pending the background trace generator proc_73f6dfb204ff.)

---

## Item 6 — Normalization ablation (Tier-2, #11)

**Source:** SIGIR-Final-Reviews.md §11 (lines 520–566) + §12 (three-concept separation, lines 568–598).
**V5 anchor:** §6.5.3 already sweeps α (linear weight); this item isolates the *normalization* of each signal's magnitudes before fusion — the confound the reviewer flags ("is it absolute scale or within-retriever separation?").

### Objective
Disentangle two rival explanations of the magnitude effect: (a) absolute-score *scale* of signal B, vs (b) *within-retriever score separation* (the shape of the magnitude distribution). By re-normalizing each signal four ways *before* the same CombSUM/RRF fusion, we test which normalization regime preserves the relevance-aligned-magnitude advantage.

### SPEC — new script `scripts/normalization_ablation.py`
Reuse the existing SF+SPLADE n=100 component traces (`*_sf_splade_comp_*.json`). For each query, transform signal B (SPLADE) and/or signal A (SF) independently via four schemes, then recompute MRR under CombSUM and RRF:
- **Raw**: s as-is (current behaviour).
- **Min-max**: (s − min)/(max − min) per query.
- **Z-score**: (s − μ)/σ per query.
- **Rank-normalized**: replace score by percentile rank.
Hold the fusion operator fixed (CombSUM α=0.3, RRF k=60). Metrics per (normalization_A × normalization_B) cell:
- MRR (CombSUM, RRF), ΔMRR = MRR_CombSUM − MRR_RRF.
- top-1 change count; Kendall τ(CombSUM, RRF).
- Re-run the Item-1 World− intervention under each cell to see whether the magnitude effect (World− degradation) survives normalization.
Also report the three-concept framing (§12): rank information R(s) preserved by all; within-signal geometry G_within altered by normalization; cross-signal calibration G_cross changed.

### Hypotheses
- **H8a**: The effect (CombSUM > RRF, positive World− degradation) persists under Min-max and Z-score (these preserve *separation* but kill *absolute scale*) → confirms the effect is about **within-retriever separation**, not raw scale.
- **H8b**: Rank-normalized collapses CombSUM and RRF to near-identical rankings (I_1 ≈ 0) — because rank-normalization discards G_within entirely, leaving only R(s), which RRF already uses → predicts the §9 "non-identifiable" boundary from the normalization side.

### Risks / prerequisites
- No new traces needed (reuses SF+SPLADE n=100). Pure offline re-normalization + re-fusion → fast.
- Edge case: σ=0 queries (degenerate) → skip or add tiny epsilon.

### Success criteria
- [ ] 4×4 normalization cell matrix (or the 4 single-signal schemes) with real MRR/ΔMRR/τ.
- [ ] World− degradation reported per cell (effect survives vs dies).
- [ ] V5 new §6.8 "Normalization and the source of magnitude" + Appendix E.9; ties to §12 three-concept terminology.
- [ ] Answers the reviewer's core question explicitly: scale vs separation.

---

## Item 7 — Cross-dataset prediction (Tier-2, #14)

**Source:** SIGIR-Final-Reviews.md §14 (lines 640–704). "Elevate the paper considerably."
**V5 anchor:** §6.6.5 already builds geometry features + a (underpowered) leave-one-*dataset*-out logistic predictor on n=10. This item scales it to n=100, adds AUROC/AUPRC/CIs, and removes the "too few divergent queries" weakness.

### Objective
Test whether pre-fusion geometry *predicts* the winning operator family (CombSUM > RRF) on **unseen datasets** — the generalization claim the n=10 study admitted it couldn't support. Use leave-one-dataset-out (LODO), never random query split (avoids dataset-characteristic leakage).

### SPEC — extend `scripts/geometry_predictor.py` (new `cross_dataset_predict.py`)
- Label Y_q = 1 if RR_CombSUM,q > RR_RRF,q else 0, per query, over all n=100 traces (hotpotqa, musique, nq_rear; add SciFact/2Wiki if traces exist).
- Features: the 21 geometry features from §6.6.5 (9 per signal + 3 pair features).
- LODO: train on all datasets but one, test on the held-out; rotate. Classifiers: logistic regression + decision tree (keep simple per §17 — no XGBoost/LightGBM to avoid statistical-attack surface).
- Metrics: AUROC, AUPRC, accuracy, calibration (reliability curve), bootstrap 95% CI (B=10000, seed=42) on pooled LODO predictions.
- Report per-held-out-dataset AUROC + the pooled estimate.

### Hypotheses
- **H9**: LODO AUROC > 0.5 (geometry generalizes to unseen datasets); modest but significant AUROC (~0.6–0.7) is a *major* contribution vs the current "cannot support" admission.
- Null/limitation: if AUROC ≈ 0.5, report honestly as a negative result (feeds §22) and keep the per-dataset (not cross-dataset) finding.

### Risks / prerequisites
- Needs n=100 component traces for ≥3 datasets (hotpotqa/musique/nq_rear available from Items 1–5; SciFact/2Wiki need regen — optional).
- Underpowered if divergent queries (Y=1) are few per dataset → report base rate + AUPRC, not just accuracy.

### Success criteria
- [ ] LODO AUROC/AUPRC/accuracy + CIs for ≥3 datasets.
- [ ] Calibration reported; leakage explicitly ruled out (LODO, not random split).
- [ ] V5 §6.6.5 upgraded with the cross-dataset result + Appendix E.10; resolves the current "too few observations" honest weakness.

---

## Item 8 — Synthetic phase diagram (Tier-3, #15)

**Source:** SIGIR-Final-Reviews.md §15 (lines 706–753). "Relatively cheap and conceptually powerful."
**V5 anchor:** new; gives a *mechanistic map* of fusion behavior — the central figure the reviewer asks for in §23.

### Objective
Build two synthetic retrievers with *controlled* parameters — rank correlation τ ∈ [−1,1] and score-margin difference Δ — and map MRR_CombSUM − MRR_RRF across the (τ, Δ) plane. This isolates the regimes where magnitude-aware fusion helps, hurts, or is neutral, independent of any real corpus.

### SPEC — new script `scripts/synthetic_phase_diagram.py`
- Generate, per (τ, Δ) cell on a grid (τ ∈ {−0.9,…,0.9}, Δ ∈ {0, 0.25, 0.5, 1, 2}):
  - Retriever A scores: a relevance-aligned base (gold docs high) + noise.
  - Retriever B scores: constructed to have target Kendall τ with A's *ranking* and target score-margin Δ vs A.
  - A fixed gold set per query (e.g. top-1 or top-3 of a latent relevant ranking).
- Compute MRR under CombSUM and RRF; record ΔMRR = MRR_CombSUM − MRR_RRF.
- Aggregate over many synthetic queries per cell (seeded); produce a heatmap (τ × Δ) of mean ΔMRR.
- Validate against real data: overlay the empirical (τ, Δ) of the real SF+SPLADE vs SF+DPR pairs to show the diagram *explains* why SF+SPLADE helps and SF+DPR doesn't.

### Hypotheses
- **H10**: ΔMRR > 0 (CombSUM helps) requires BOTH high τ (retrievers agree on ranking) AND positive Δ (SF carries larger magnitude separation on gold) — i.e. the effect lives in a specific (τ, Δ) quadrant, matching the real SF+SPLADE location; low-τ or zero-Δ cells show ΔMRR ≈ 0 or negative.
- This mechanistically recovers Items 3/5: SF+DPR sits in a zero-Δ / non-identifiable cell → effect absent.

### Risks / prerequisites
- Synthetic generator must reproduce the *real* τ/Δ of SF+SPLADE & SF+DPR as a validation anchor (use `operator_identifiability.py` / `geometry_predictor.py` feature extractors to get empirical τ, Δ).
- Pure simulation → fast, no model loads, fully offline.

### Success criteria
- [ ] (τ, Δ) heatmap of ΔMRR with real-data anchor points overlaid.
- [ ] Regime description: where magnitude-aware fusion helps/hurts/neutral.
- [ ] V5 new §6.9 "A mechanistic phase diagram of fusion" + the heatmap as a central figure (feeds §23 one-central-figure request); Appendix E.11 with the grid values.

---

*Items 9+ candidates (not yet specced): #10 Candidate-set intervention (pool difficulty N∈{10,20,50,100,200,500} × distractor type → ΔMRR), #20 causal-language tightening, #21 SF-role clarification, #22 negative-results table. Available on request.*

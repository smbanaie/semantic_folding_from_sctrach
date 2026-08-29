# SIGIR-Final-Tasks — Item 2 (Tier-1, #38-2)

**Loop rule (per user):** analyze review → write SPEC/PLAN for ONE item → show → user confirms → implement/run → save results → update Journal_V5.md → next item.

**This file covers only Item 2.** Item 1 is done (§7.6.1 + Appendix E.5 committed). Items 3–5 are queued.

---

## 0. Review source & linkage

- SIGIR-Final-Reviews.md **§6 Experiment #3 — Relevance-conditioned score geometry** (lines 249–319): "Which geometry variables actually predict fusion success? ... regress ΔMRR_q on top-k geometry features."
- **§7 Even better: explain the actual winning queries** (lines 321–360): classify each query into Type A/B/C/D by how CombSUM vs RRF changes rank, then ask "What distinguishes A from C?"
- #38 Top-5 package item **2. Relevance-aware margin analysis** (closest mapping: the magnitude/margin geometry that predicts fusion success).
- Pairs with Item 1: both reuse the same SF+SPLADE component traces; Item 1 proves the magnitude is relevance-aligned *causally*, Item 2 shows *which geometry features predict* where fusion wins.

**Central gap the reviewer identifies (verbatim §6):** the geometry framework `G(s)=(R,μ,σ,Δ12,Δ15,ρ,κ)` is "currently more descriptive than explanatory." The fix: regress `ΔMRR_q = RR_CombSUM,q − RR_RRF,q` on geometry features, and prefer **top-k geometry** over global.

---

## 1. Objective (one sentence)

Show that query-level score-geometry features — especially *top-k relevance-conditioned margins* — predict when CombSUM beats RRF (ΔMRR_q > 0), turning the geometry framework from descriptive to explanatory, and characterize the winning-query population (Type A–D).

## 2. Hypotheses

- **H4a (predictive):** A standardized regression of ΔMRR_q on top-k geometry features yields a non-zero, sign-stable β for the gold-vs-rank-k margin `Δ^gold_{1,k}` and the cross-signal gold margin `Δ^cross_1`, with bootstrap CIs that exclude zero on the discriminating datasets (HotpotQA, MuSiQue).
- **H4b (winning-query structure):** Type A queries (CombSUM moves gold 3→1 or 5→2) are characterized by a *small/negative* joint margin at baseline (the §7.5 regime), whereas Type C (no change) sit at *large positive* margin (gold already dominates both signals). This recovers the §7.5 margin-vs-error finding at population level.

## 3. SPEC — what gets built

### 3.1 New script
`scripts/geometry_predictor.py` (reuses loaders from `magnitude_relevance.py` / `counterfactual_magnitude.py`).

Inputs per (dataset, query): maxnorm(SF), maxnorm(SPLADE) component score dicts + gold id set (same traces Item 1 uses; n=10 now, n=100 when ready).

### 3.2 Geometry features per query

Global (per signal, then cross):
- `Δ12` = s(d1) − s(d2); `Δ15` = s(d1) − s(d5) (within-signal top gaps)
- `σ`, `μ` of the score distribution; `ρ` = Pearson(SF rank, SPLADE rank); `κ` = top-1 score mass fraction
- `τ_signal` = Kendall τ between the two signals' score orders (already defined in §6.6.3)

Top-k relevance-conditioned (the §6 recommendation — these are the explanatory features):
- `Δ^gold_{1,3}`, `Δ^gold_{1,5}` = s(gold) − s(d3), s(gold) − s(d5) within the gold's own signal
- `Δ^cross_1` = s_SPLADE(gold) − s_SF(gold) (cross-signal gold margin)
- `joint_margin` = mean over signals of (gold score − best non-gold score)/max|score| (reuses §7.5 def)
- `gold_rank_sf`, `gold_rank_sp` = ordinal rank of gold in each signal

### 3.3 Measurements per query
- `ΔMRR_q = RR_CombSUM,q − RR_RRF,q` (the §6 target), also `ΔRR_q` for CombSUM vs RRF.
- Winning-query class:
  - **Type A** CombSUM moves gold rank 3→1 or 5→2 (or any promotion into top-1 that RRF misses)
  - **Type B** CombSUM moves gold 5→2 (mid promotion)
  - **Type C** no change in gold rank
  - **Type D** CombSUM *hurts* (gold rank worse than RRF)
- Standardized features (z-score across queries per dataset) so β are comparable.

### 3.4 Statistics
- Per dataset: OLS `ΔMRR_q ~ β·[Δ^gold_{1,5}, Δ^cross_1, joint_margin, τ_signal, Δ12, σ]` on standardized features.
- Bootstrap 95% CI (B=10000, seed=42) on each β; report sign-stability across datasets.
- Partial-R² / standardized effect for the top-k margin feature specifically.
- Type A–D counts per dataset; for A vs C, report mean joint_margin (predicts A has smaller margin).
- Wilcoxon on ΔMRR_q vs 0 (does CombSUM systematically beat RRF?).

## 4. Scope / datasets & pairs

Same trace set as Item 1: hotpotqa, musique, 2wiki, scifact (n=10 now); nq_rear + n=100 hotpotqa/musique when Item 1's generator finishes. BM25+SPLADE / SF+DPR reused from §6.5 if available (generality already partially covered there).

## 5. PLAN — execution order

1. **Inspect loaders** from `counterfactual_magnitude.py` — reuse `load_component_traces` (SF+SPLADE, gold map). No new I/O.
2. **Write `scripts/geometry_predictor.py`** implementing §3.2–3.4. Deterministic seed. Output JSON + MD to `appendix_stats/geometry_predictor.{json,md}`.
3. **Run** on the n=10 set (hotpotqa/musique/2wiki/scifact) — fast, no re-index.
4. **Verify** regression runs; check sign of `Δ^gold_{1,5}` β matches H4a on discriminating datasets.
5. **Aggregate** per-dataset β table (with bootstrap CI) + Type A–D counts + A-vs-C margin comparison.
6. **Sanity** vs hypothesis: expect on hotpotqa/musique a positive, CI-excluding-zero β for the top-k margin; Type A queries have smaller joint_margin than Type C. On 2wiki/scifact expect null (ceiling/concentration) — report as boundary conditions (feeds the §22 negative-results table).
7. **Write `SIGIR-Final-Results-Item2.md`** with tables + a top-k-geometry scatter (ΔMRR_q vs joint_margin).
8. **Hold for user confirmation**, then update Journal_V5.md: new §7.9 "Predicting operator suitability from query geometry" (extends existing §6.6.5) + Appendix entry E.6; and fold the Type A–D decomposition into §7.5/§7.8.

## 6. Success criteria

- [ ] Geometry features computed; `ΔMRR_q` regression runs on all datasets.
- [ ] On ≥1 discriminating dataset, top-k margin β sign-stable positive with bootstrap CI excluding zero.
- [ ] Type A–D decomposition produced; A-vs-C mean joint_margin difference reported.
- [ ] Results written to `SIGIR-Final-Results-Item2.md`; only real numbers.
- [ ] V5 updated after user confirmation.

## 7. Risks / pitfalls

- **n=10 is small** for regression — bootstrap CI will be wide; report as directional + flag n=100 confirmatory (same caveat as Item 1).
- **Multicollinearity** among geometry features (Δ12/Δ15/σ correlated) — use standardized features + report per-feature CI, not a single joint R²; consider dropping redundant global features if VIF high.
- **Ceiling datasets (2wiki) saturate** ΔMRR_q=0 → regression undefined there; exclude from β inference, keep as negative-result row.
- **Reuse Item 1 traces** — no new benchmark run needed.

---

## Queue (NOT this turn)
- **Item 3** — Operator identifiability (§9; formalize I_global/Top-k/Top-1 table).
- **Item 4** — Top-rank ΔRR decomposition (§8; extend to distribution table).
- **Item 5** — Extra sparse + dense checkpoint (§13/#38-5; SPLADE-B, DPR-B) for generality.

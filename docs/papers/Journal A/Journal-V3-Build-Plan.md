# Journal V3 Build Plan — Salvage + Full SIGIR_REVIEW Audit

## Verdict on the two source files

| File | Identity | Reliability |
|------|----------|-------------|
| `*_journal-V1.md` | Byte-content == committed `journal.md` @ `3d1f5ce` (only CRLF differs). Contains ALL Aug-24 work: n=100 core, p_Holm=0.0007, compress/amplify/magswap, §6.6.3/6.6.4, §7.6 H3, 20-entry refs | Data-complete; a few stale phrasings |
| `*_journal-V2.md` | Older branch (n=50 era: "Larger-n studies remain future work", corrupted refs — "Banditov" for Belebele, Welbl mislabeled NarrativeQA-adjacent, web-blocked drafting note) | Stale data; some phrasings better |

The "totally invalid content" report is explained: V2 looks newer by mtime (12:24 > 10:14) but is an **older text generation**. Anyone reading V2 as latest sees outdated stats + mangled references.

## Hunk adjudication (34 diff hunks)

**Take from V1 (data):** everything with n=100 / battery / τ split / H3 / factorial / refs.
**Take from V2 (phrasing only), applied onto V1 base:**

1. Abstract contribution (2): "as the mechanism by which rank-only fusion can degrade compositional reranking" → replace with review's own wording "a mechanism that can degrade compositional reranking under specific score-geometry conditions"
2. Abstract feature-invariance parenthetical: V2's compact "(SF scores are a deterministic function of term-co-occurrence overlap…)" replaces V1's longer double-clause
3. §1.1 central claim: V2's "For single-hop matching, rank is often sufficient" (removes ceiling-effect hedging)
4. Contribution 4 (mechanistic): V2's tighter sentence
5. §3.2 SF-probe: "fully characterized probe whose score construction is fully characterized" (V2) over "comparatively transparent" ×2 sites + SPLADE-only paragraph "isolates"
6. §6.6 heading back to "(Kendall's τ)" — but keep body τ_signal/τ_operator distinction (review item 19 satisfied inside)
7. §7.1 synthetic design: V2's honest "hold RANK fixed … by construction" replacing V1's convoluted intended-vs-induced ordering note
8. §8.4 heading "Complete-Collection Evaluation" + §9.5 "Deployment Considerations" + V2 deployment paragraph
9. C.4 CombMNZ sentence: V2's "multiplicity weighting adding value when evidence is distributed over a 385-document pool" (concrete) over V1's hedge
10. Conclusion closer: V2's bold bottleneck sentence

## Audit result vs SIGIR_REVIEW.md (38 items)

### APPLIED (evidence in V1)
0.2 exploratory framing (9 hits) · 0.3 n=100 runs+stats (14) · 0.4 causal→sensitivity (only epistemics sentence) · 0.5 signal-B gone (0 hits, joint geometry everywhere) · 0.9→22 feature-invariance [To be filled] removed, harness scoped as future work (Option B+ partial A) · 0.10 complete-collection/constructed-collection terminology · 1 title kept · 2 abstract conditional · 3 intro hop-may-correlate · 4 RQ4 boundaries · 5 framework strong · 6 Prop-1 as foundation · 8 Regime A/B · 10 seven operators · 13 SF-as-probe trimmed · 14 matrix kept · 16 joint geometry rewrite · 17 two-checkpoints scoping · 19 τ_signal/τ_operator split + fusion-gain fit reported (§6.6.3) · 20 synthetic control kept · 21 real perturbation battery · 23 pool sweeps N=20–494 · 25 renamed · 26 conditional compatibility hypothesis · 27 Table 2.1 positioning · 30 limitations expanded (ρ calibration note) · 31 refs rebuilt, no pending-notes except 2 honest [venue verify] → now to be resolved in V3 · 36 removals done · 37 phases 1–3 largely executed.

### PARTIAL (needs strengthening in V3)
- **0.3**: add win/tie/loss + rank-1-change counts (RRF→CombSUM) — reviewer's "most useful statistic"; power-analysis sentence
- **12**: statistics table needs Win% column
- **18**: α-sweep needs CI-over-α statement (response-surface framing)
- **28**: "descriptive pre-fusion diagnostic" wording not yet present
- **31**: remove last 2 `[venue verify]` flags (resolve or drop claims)
- **34**: three-level claim hierarchy not explicit → add claims-classification block
- **35**: contribution list partially aligned; tighten to reviewer's four

### MISSING (new work required for V3)
- **0.6** score-geometry predictor experiment (features → predict winning operator)
- **0.8** calibration baselines (L2/rank-Gaussian/sigmoid/quantile) to separate magnitude from calibration
- **0.11/24** SciFact-5183 component investigation (SF-only/SPLADE-only/BM25-only + gold-rank histogram + CV)
- **7.1** query-sampling protocol statement (+ multi-seed where feasible)

## Execution order for V3
1. Build V3 = V1 + 10 V2 phrasing adoptions (scripted, deterministic)
2. Run new experiments: `scripts/calibration_baselines.py`, `scripts/scifact_deep_investigation.py`, `scripts/geometry_predictor.py`
3. Insert results into V3 (new §7.x calibration subsection, §8.4 investigation block, §6.6.5 predictor, §4.1 sampling protocol)
4. Statistics upgrades (win/tie/loss, rank-1 changes, power analysis), α response-surface sentence, three-level claims block
5. Resolve [venue verify] flags
6. Final gates + verify PASS → deliver as final version

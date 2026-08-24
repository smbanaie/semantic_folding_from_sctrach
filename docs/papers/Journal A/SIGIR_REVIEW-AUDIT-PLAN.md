# SIGIR_REVIEW.md → Journal_V3.md Full Audit Plan

Goal: item-by-item verification of every action in `SIGIR_REVIEW.md` against
`Journal_V3.md`, fixing all PARTIAL/MISSING items, delivering the final
polished Journal_V3.md. Every fix must trace to a committed artifact or script.

## Phase A — Audit scaffold
- **A1** Extract all 38 numbered items (+ sub-items) from SIGIR_REVIEW.md into
  `SIGIR_REVIEW-AUDIT.md` checklist: item id, reviewer demand, target section,
  status (APPLIED / PARTIAL / MISSING), evidence pointer.
- **A2** Baseline gates on current V3 (banned terms, causal count, heading
  sequence, data markers) so later deltas are measurable.

## Phase B — Verification sweep (no edits; classify each item)
- **B1** Paper-wide 0.1–0.11 (protocol coherence, n=10 language, sample size,
  causal wording, signal-B, geometry predictor, calibration, feature
  invariance, full-corpus terminology, SciFact investigation).
- **B2** Items 1–6: title, abstract rewrite, intro/RQs, related-work refs,
  framework, Proposition-1 framing.
- **B3** Items 7–14: dataset table rigor, query-sampling protocol + multi-seed,
  candidate regimes A/B/C, retrieval models coverage, fusion operators +
  learned baseline, α-tuning framing, statistics completeness, exploratory/
  confirmatory separation.
- **B4** Items 15–25: CombSUM/CombMNZ explanation evidence, second-model
  validation wording, SPLADE-v3 scoping, α CI, Kendall-τ treatment, synthetic
  magnitude grid, real perturbation battery, feature invariance, scaling
  analysis, SciFact decomposition, complete-collection naming.
- **B5** Items 26–31: discussion framing vs Bruch, practical guidelines τ,
  deployment brevity, limitations list, reference audit.
- **B6** Items 32–38: required figures inventory, central diagnostic table,
  three-level claims, contribution wording, removal list, execution order
  cross-check, final assessment criteria.

## Phase C — Fixes & new experiments (from B classifications)
- **C1** Textual fixes for every PARTIAL item found in B1–B6.
- **C2** NEW EXPERIMENT — feature-invariance harness (items 0.9/22, reviewer
  "strongly prefers A"): compute per-doc raw-overlap proxy + candidate features
  (doc length, lexical overlap, term rarity); regress SF score on overlap +
  features; report partial contributions. Script `scripts/feature_invariance.py`
  → `appendix_stats/feature_invariance.{md,json}`; new §8.1 paragraph.
- **C3** NEW ANALYSIS — split-half stability (item 7.1 multi-seed equivalent):
  partition n=100 confirmatory queries into disjoint halves ×10 random splits;
  report per-operator mean±std and CombSUM-vs-RRF sign stability.
  Script `scripts/split_half_stability.py` → appendix artifact; §4.x sentence.
- **C4** Central diagnostic table (item 33): assemble per-task τ/CV/margin/
  best-operator/evidence row from existing artifacts into §6.1 area.
- **C5** Figures (item 32): generate missing central figures from committed
  data — operator-map heatmap, perturbation battery chart, pool-growth curve,
  phase-diagram heatmap → `figures/`; reference them from relevant sections.
- **C6** Statistics presentation (item 12): render win/tie/loss + effect size
  as a paper table (not only prose), citing `win_loss_rank1_n100.*`.

## Phase D — Final gates & delivery
- **D1** Re-run all grep gates + numerical spot-checks on updated V3.
- **D2** Heading-sequence + cross-ref integrity after insertions.
- **D3** Fresh ad-hoc verifier over changed scripts/artifacts (PASS required).
- **D4** Update REPORTS/BENCHMARK_RESULTS pointers if new artifacts warrant;
  commit (`docs(journal-a): ...`) and push.

## Known open risks entering Phase B
- Item 9 (second dense retriever): compute-heavy; reviewer marks lower priority
  than calibration → plan to scope explicitly in limitations rather than run.
- Item 23 pool sizes up to 10k: SciFact-5183 covers >5k point; HotpotQA capped
  at constructed 494 → state infra boundary honestly.
- Item 20 heatmap: synthetic_operator_phase.py already yields the grid; needs
  rendering as Figure + explicit reference.

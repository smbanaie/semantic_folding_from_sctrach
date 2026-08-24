# Journal_V4 Plan — SIGIR-Final-Reviews.md (42 items) Implementation

Source: `SIGIR-Final-Reviews.md` (964 lines, 42 numbered recommendations +
must-fix/strongly-recommended/optional checklist). Target: `Journal_V4.md`
built from Journal_V3, then chief-reviewer scoring loop to ≥9/10.

## Phase A — Build V4 base + audit scaffold
- **A1** Copy Journal_V3.md → Journal_V4.md.
- **A2** Create SIGIR-Final-AUDIT.md checklist: 42 rows, status/evidence,
  refreshed after each wave.

## Phase B — Must-fix wave (reviewer's own "Must fix" list)
- **B1** Central claim softened everywhere (item 1): replace
  "rank-only fusion discards magnitude that compositional tasks require"
  framing with the 9/10-safe claim in Abstract/Intro/Contributions/
  Discussion/Conclusion (near-verbatim).
- **B2** Title → Option A: "When Does Score Magnitude Matter? Task- and
  Retriever-Pair Dependence in Hybrid Retrieval Fusion" (item 2).
- **B3** SF-as-probe rewrite of Introduction with the suggested sentence
  (item 3).
- **B4** Contributions → exactly three, reviewer's wording + the explicit
  non-claims sentence (item 4).
- **B5** n=50/n=100 separation statement in Appendix C intro + "all primary
  statistical claims based on n=100" (item 10); statistical protocol
  paragraph (item 11); no bare "significant" — every instance names the
  Holm family (item 12).
- **B6** Operator × Retriever-Pair Interaction Screen terminology + paired
  difference-of-differences sentence (item 13); remove "central claim in its
  most direct form" self-praise (item 33).
- **B7** shufflescores interpretation corrected per item 40 checklist;
  N<100 recommendation removed (item 21); SciFact scaling softened to
  "unidentifiable when candidate generation fails..." (item 20);
  "first measured sweep" removed (item 32); divergence-rate sentence deleted
  or reduced (item 31).
- **B8** §7.6 circularity guard: rank-conditioned magnitude analysis
  (NEW BENCHMARK/ANALYSIS — logistic R ~ rank + magnitude; incremental ΔAUC /
  partial contribution of magnitude beyond rank) + rename section to
  "Relevance-bearing score magnitude" (items 18, 19).
- **B9** Causal hierarchy sentence added (item 28): §7.1 property / §7.2
  manipulation / §7.4 causal sensitivity on real outputs / §7.6 relevance
  association.

## Phase C — Strongly recommended wave
- **C1** Benchmark taxonomy table (item 5) + operational single-hop/multi-hop
  definitions (item 6) + Tier 1/2/3 benchmark hierarchy (item 7).
- **C2** Visual-hierarchy labels on every table: "Exploratory diagnostic;
  n=10" vs "Confirmatory evaluation; n=100" (item 8).
- **C3** Master benchmark table (item 9): Dataset|Pair|n|Pool|Best op|MRR|
  RRF|Δ|p_Holm from appendix_c_*_n100 artifacts.
- **C4** Manipulation→inference table with Primary inference column (item 35)
  + three-magnitudes definitions raw/relative/cross-signal mapped to
  conditions (item 34) + operator taxonomy table (item 36).
- **C5** Retriever-pair matrix table (items 37–38) + BM25+SPLADE generality
  sentence; SPLADE-v3 "qualitative ordering" softening (item 39).
- **C6** Signature figure: conceptual 2D phase map (magnitude informativeness ×
  disagreement) with datasets overlaid, labeled conceptual not validated
  (item 14). Extend fig3 into 3-panel causal centerpiece Original/
  Rank-preserving/Rank-destroying with MRR+τ (item 15).
- **C7** Practical guidance rewritten per items 22/23/24; falsification
  paragraph added (item 25).

## Phase D — Structure & presentation
- **D1** Limitations reorganized into Internal/External/Deployment validity +
  Threats to Validity subsection (items 26–27).
- **D2** §7.4 compressed to Question/Design/Prediction/Result/Interpretation
  skeleton; big tables already in Appendix E (item 16).
- **D3** Abstract replaced with the reviewer's 5-paragraph structure (item 29);
  conclusion replaced with the information-preservation version ending in the
  bold design principle (item 30).
- **D4** Main-paper slimming pass: claim→experiment→result→interpretation;
  parameter/raw detail stays in appendix (item 17).

## Phase E — Verification & scoring loop
- **E1** Consistency check: all numbers cross-checked against committed
  artifacts (numerical audit script), banned terms, heading sequence,
  figure/table refs resolve, claim-evidence alignment spot checks.
- **E2** Chief-reviewer scoring rubric (claim alignment, statistics rigor,
  benchmark protocol clarity, narrative economy, presentation, reproducibility)
  scored 1–10; iterate refinements until ≥9.0 overall with no dimension <8.5.
- **E3** If any dimension blocked by missing evidence → new benchmark run
  (only rank-conditioned analysis anticipated; no broad new experiments per
  item 41).
- **E4** Final gates + fresh ad-hoc verifier PASS → commit/push as V4 final.

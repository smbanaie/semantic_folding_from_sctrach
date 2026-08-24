# Final Improvements V2 — Execution / Refactor / Rewrite Plan

**Sources:**
- `SIGIR_REVIEW-V2.md` — chief-reviewer re-review of current manuscript. Verdict **6.2–6.7/10 Major Revision / Weak Reject**; 32 numbered findings; "fixable, not a rejection of the idea"; explicit 8-item P0/P1 revision list (§31).
- `SIGIR-RESOLVER-V2.md` — researcher implementation program: hypothesis freeze (H1–H4), frozen-trace experimental framework, 18 experiments, priority table (§26).

**Scope discipline (reviewer §32):** no new datasets, no new operators, no new theory. Every change must make an existing claim match its actual evidence.

---

## PART 0 — RECONNAISSANCE RESULTS (verified on disk before planning)

| Asset | Status |
|-------|--------|
| Real component scores (SF+SPLADE, n=10, HotpotQA/MuSiQue/SciFact) | ✅ `appendix_alpha/*_comp_{1.0,0.0}.json` |
| Gold labels for those queries | ✅ three `query_gold.json` files |
| Hop annotations: HotpotQA raw `supporting_facts` | ✅ 1,000 items |
| Hop annotations: MuSiQue converted `is_supporting` + raw decomposition (`question_decomposition`, `paragraph_support_idx`) | ✅ 2,417 dev queries |
| 2Wiki raw supporting_facts | ✅ 1,000 items |
| n=50 runs w/ per-query fused results | HotpotQA SF+SPLADE ✅; MuSiQue SF+SPLADE ✅; NQ-REaR SF+SPLADE + SF+DPR + BM25 pairs (Aug-22) ✅; DPR runs Aug-22 ✅ |
| Query headroom for confirmatory core | HotpotQA converted only 50 (raw 1,000 → can reconvert); MuSiQue 2,417 available (n=100 cheap); NQ-REaR exactly 100 |
| Compute | CPU-only torch (DPR n=50 ≈ 287 s/batch → n=100 ≈ ~10 min/op-set); SPLADE batch n=50 ≈ 93–397 s |
| Existing scripts to extend | magnitude_perturbation.py (x2/log1p/pow05/rpr/shufflescores), margin_vs_error.py, operator_identifiability.py, learned_fusion_baseline.py, appendix_c_stats.py |
| τ terminology split | ❌ not yet in paper |
| Factorial analysis of 4-pair matrix | ❌ absent (paper reports 4 separate tables) |
| Magnitude-relevance correlation test (H3) | ❌ absent |
| Calibration test | ❌ absent (word appears only in unrelated contexts) |
| References section | ⚠️ contains forbidden drafting note (line 671) + known-bad entries (2Wiki 2017, NarrativeQA/PubMedQA merged entry, Belebele author "Banditov", NQ-REaR provenance) |
| "fully characterized" phrasing | 3 sites remain |
| Deployment §9.x | present as full subsection |

---

## PART 1 — NEW BENCHMARKS & ANALYSES (the experiments)

### B1. Expand confirmatory core to n=100 (P0, reviewer #11)
The single largest empirical weakness. Designate the confirmatory core: {HotpotQA, MuSiQue, NQ-REaR} × {RRF, CombSUM} × {SF+SPLADE, SF+DPR}, target **n=100** per cell.
- MuSiQue: rerun benchmark phase at `query_end=100` on existing index run `run_20260822_191925` (corpus already covers 828 docs from first 50 queries; verify pool coverage for queries 50–99 — if pools are built per query from converted JSONL this works; else rebuild index at max_queries=200). ~10 min/op × 7 ops.
- HotpotQA: reconversion needed (converted JSONL has only 50; raw has 1,000). Reconvert with max_queries=150 → fresh index → benchmark n=100. Index cost ≈ prior index time (~15 min).
- NQ-REaR: exactly 100 available → n=100 is the full set. Reuse Aug-22 run dirs if pool-compatible.
- Deliverable: `benchmark_*_n100/` runs + refreshed Appendix C tables (C.1–C.3) + Holm-corrected stats at n=100.
- **Decision gate:** if any n=10-era headline gap (e.g., CombSUM vs RRF on HotpotQA) fails to persist directionally at n=100, the paper's central claim is *weakened in writing* accordingly — pre-committed here.

### B2. H3 magnitude-relevance correlation test (P0, reviewer #7 + resolver Exp 3/4/5)
New script `scripts/magnitude_relevance.py`. For every candidate doc in every real trace (SF+SPLADE endpoint artifacts + n≥50 runs where component scores exist):
- Per-retriever margin stats: Δscore = score_gold − best_negative; P(Δ>0), mean, median, AUC(score) — split by task type (single-hop vs multi-hop vs factoid-large-pool).
- Hop-coverage analysis (HotpotQA/MuSiQue only): Spearman(SPLADE score, hop_coverage) where hop_coverage = fraction of supporting docs' titles/terms matched by the candidate (proxy at doc level: candidate ∈ supporting set ⇒ 1.0; partial title-term overlap otherwise). Regression: score ~ hop_coverage + lexical_overlap + doc_length.
- **Outcome branches:** β_hop > 0 significant → paper may say "magnitude correlates with compositional evidence, controlled for length/overlap." Otherwise → all "compositional confidence" language replaced by "score-separation information" and H2 downgraded. Pre-committed.

### B3. Score calibration curve (P2→promoted, resolver Exp 14)
Same script family as B2: bin real scores per retriever, plot/report P(gold | score-bin). If flat → magnitude has no semantic value in that retriever → stated plainly. Cheap (reuses B2 traces).

### B4. τ_signal vs τ_operator split + local disagreement (P0 text + P1 experiment, reviewer #15/#16)
- New computation in `scripts/tau_analysis.py`: τ_signal = Kendall(SF ranking, SPLADE ranking) per query; τ_operator = Kendall(RRF output, CombSUM output) per query; top-1/top-3 agreement; gold-rank difference. Fusion Gain = MRR(fused) − max(MRR(A), MRR(B)).
- Correlate Fusion Gain against τ_signal AND top-k disagreement separately. Expected finding (per reviewer): global τ weak predictor, local disagreement strong. Either result is reportable.

### B5. Factorial analysis of the four-pair matrix (P0, reviewer #28 + resolver Exp 6)
Promote 4-pair comparison to centerpiece. With per-query MRR arrays from n=100 confirmatory cells (+ existing n=50 for the remaining pairs), fit:
- Permutation-based Operator × RetrieverPair interaction test (no statsmodels dependency: shuffle operator labels within query strata, 10k resamples, seed=42) on {HotpotQA, NQ-REaR}.
- Report interaction effect size + p_perm alongside the existing tables.
- Script: `scripts/factorial_interaction.py`.

### B6. Rank-preserving intervention battery upgrade (P0, reviewer #4 Experiment A/B/C + resolver's non-negotiable experiment)
Extend `scripts/magnitude_perturbation.py` with two conditions on **real traces**:
- `compress`: rank-preserving squash toward equal magnitudes (resolver Condition B/D: [43,12]→[1.0,0.99]).
- `amplify`: rank-preserving spread ([43,12]→[1000,1]).
- `magswap`: move the top distractor's magnitude onto the gold doc without changing ranks (Experiment B semantics).
Run on HotpotQA+MuSiQue real component scores; verify RRF/Borda bit-invariance under compress/amplify/rpr; quantify CombSUM-family decision flips. This becomes §7.4's primary evidence.

### B7. Complementarity 4-cell table (P1, resolver Exp 12)
Classify each confirmatory-core query: (A✓B✓ / A✓B✗ / A✗B✓ / A✗B✗) using top-1 correctness of each signal alone; report fused MRR per cell per operator. Shows where fusion actually matters. Script extension inside `tau_analysis.py`.

### Explicitly NOT done (per reviewer §32 + resolver §26): new datasets, new operators, more fusion families, web-scale retrieval, additional checkpoints.

---

## PART 2 — PAPER REWRITE / REFACTOR

### R1. Kill residual causal language (P0, reviewer #3/#18)
- Contribution 3 rewrite (exact sentence provided by reviewer, adopted verbatim): *"Through controlled magnitude perturbations, we demonstrate that magnitude-sensitive fusion operators respond to score magnitude even when ordinal information is held fixed, and connect this controlled behavior to observed retrieval traces."*
- Abstract: "we isolate magnitude information loss as the mechanism" → *"we identify magnitude information loss as a mechanism underlying several observed multi-hop reranking failures"*.
- Global replace: "causal factor/control/separation" → "controlled evidence for the functional role of magnitude" / "causal sensitivity of the fusion operator (not semantic validity of magnitude)". Add one explicit epistemics sentence in §7: *"We establish causal sensitivity of the fusion operators to magnitude, not causal validity of magnitude as a relevance signal; the latter is addressed by the correlation analyses in §7.x."*

### R2. Fix the 2-doc toy inconsistency (P0, reviewer #13)
The large/small/reversed table claims "rank held fixed" while reversed makes B rank-1. Rewrite §7.2 lead-in: *"document identities are fixed and the intended ordering is fixed; score magnitudes are manipulated"*; describe the rev row as the control where intended ordering contradicts score-induced ordering. Keep monotone-transform invariance checks as the true rank-fixed condition (they exist: log/sqrt/exp/sigmoid bit-identity claim).

### R3. Replace "signal B determines operator" residue with joint-interaction framing (P0, reviewer #5)
Current text already moved to "joint score geometry" (Wave-1a), but reviewer quotes the old form — verify zero occurrences of the old claim pattern and strengthen §6.5 synthesis into the reviewer's exact conceptual model: **task × signal-A geometry × signal-B geometry × operator → outcome**, with the 4-pair table as the factorial evidence (links to B5).

### R4. τ terminology split throughout (P0, reviewer #15)
Rename every occurrence: complementarity discussion uses **τ_signal**; operator-agreement tables use **τ_operator**. §6.6 gets both definitions side by side. Remove "complementarity" wording from any τ_operator context.

### R5. Verb-discipline pass (P0, reviewer #12)
Three-tier verb policy applied globally: establish/demonstrate/prove ONLY for mathematically or statistically supported claims; observe/find/consistently observe for empirical patterns at n≥50; suggest/indicate/provide exploratory evidence for n=10 probes. Audit abstract + contributions + §6.1 headline (the 0.750-vs-1.000 line gains "exploratory probe" scoping).

### R6. Single-hop ceiling phrasing (reviewer #20)
"For single-hop matching rank is often sufficient" → scoped form: *"In our single-hop candidate-reranking conditions, operator sensitivity is largely masked by ceiling effects."*

### R7. Feature-invariance precision (reviewer #8)
Replace compressed premise with: *"raw SDR overlap is determined by binary bit intersection; the final SF score is a deterministic transformation of the encoded spatial representation."* Distinguish raw-overlap → spatial-transform → emitted-score chain explicitly in §8.1.

### R8. Score-concentration interpretation softened (reviewer #9)
§8.2/§8.3 status lines rewritten to: *"consistent with, but not establishing, score concentration as the underlying mechanism."* The weird RRF non-monotonicity across N is reported as such.

### R9. "Full corpus" final rename (reviewer #10)
Regime B → **Full-dataset reranking**; every instance glossed as "full-corpus with respect to the constructed 494-document HotpotQA collection." (Supersedes the Wave-1 "complete-collection" choice — reviewer prefers this term.)

### R10. "Fully characterized" → "controlled and comparatively transparent" (reviewer #25)
All 3 sites.

### R11. CombMNZ multiplicity caution (reviewer #14)
Adopt reviewer's safer sentence wherever CombMNZ agreement-as-evidence is implied (§6.1 NQ-REaR observation, §6.6.1).

### R12. NQ-REaR reclassification (reviewer #19)
Task-topology table entry: "Magnitude/separation" assumption removed → "**large-pool factoid reranking**"; text notes its operator differences are not attributed to multi-hop magnitude semantics.

### R13. References audit + remove drafting note (P0 hard stop, reviewer #23)
- Delete line-671 blockquote (web-blocked/pending disclaimer) entirely.
- Fix entries: 2WikiMultihopQA → Ho et al., NAACL 2018 (not Trivedi 2017 EMNLP); NarrativeQA → Kočiský et al. 2018 TACL (own entry); PubMedQA → Jin et al. 2019 BioNLP workshop (own entry); Belebele → Bandarkar et al. 2023 TACL (typo "Banditov"); PopQA → Malen et al. 2022 ACL (own entry, not arXiv footnote); NQ-REaR provenance → stated as HippoRAG2 redistribution of NQ-derived REaR set with original NQ citation (Kwiatkowski 2019 TACL) + Sciavolino-style provenance note; SciFact → Wadden et al. 2020 EMNLP Findings; COVID-QA → Möller et al. 2020 (already correct); SPLADE-v3 → Lassance et al. 2023; BM25 → Robertson & Zaragoza 2009; t-SNE → van der Maaten & Hinton 2008; UMAP → McInnes et al. 2018; Morton → Morton 1966. Each verified against adapter headers/arXiv IDs where available offline.

### R14. Deployment demotion (reviewer #24)
Collapse deployment subsection (512 B/doc, CPU/GPU) to ≤3 sentences inside Discussion; delete as standalone section.

### R15. Unfinished-material sweep verification (reviewer #21/#22 — mostly done in Wave 1)
Verify zero `[To be filled]`, zero "planned appendix" strings; confirm every cited appendix (C/D/E + stats files) exists on disk with real content. Appendix C tables refresh to n=100 after B1.

### R16. Conclusion adoption (resolver §25)
Adopt the resolver's defensible-conclusion template nearly verbatim as the closing statement (it matches our current conclusion but tighter).

---

## PART 3 — EXECUTION ORDER

| Step | Work | Depends on |
|------|------|-----------|
| S1 | B1 HotpotQA reconversion + n=100 index | — |
| S2 | B1 MuSiQue + NQ-REaR benchmark n=100 | — (parallel with S1) |
| S3 | B2+B3 magnitude_relevance.py + calibration (runs on existing traces immediately) | — |
| S4 | B4+B7 tau_analysis.py (τ split, fusion gain, 4-cell) | — |
| S5 | B6 perturbation battery extension (compress/amplify/magswap) | — |
| S6 | B5 factorial_interaction.py (needs S2 outputs for max power) | S2 |
| S7 | Paper rewrite waves R1–R16 (text) incorporating S3/S4/S5 numbers | S3–S5 |
| S8 | Appendix C refresh at n=100 + verb pass final + reference audit completion | S1–S2, R13 |
| S9 | Full grep gate v2 + numerical audit v2 + commit/push | all |

---

## PART 4 — CHIEF-RESEARCHER SCORING LOOP

### Round 1

| Dimension | Score | Notes |
|-----------|-------|-------|
| Coverage of review findings | 9.5 | All 32 findings mapped; P0s all addressed |
| Scientific soundness of new benchmarks | 8.5 | B1/B2/B6 directly answer the two biggest criticisms |
| Feasibility (compute/data on hand) | 8.0 | HotpotQA reconversion risk; DPR CPU cost bounded |
| Claim-evidence alignment mechanics | 9.0 | Pre-committed weakening gates (B1/B2 outcome branches) |
| Effort proportionality (reviewer §32 warning) | 9.0 | No breadth added; everything serves existing story |
| Reproducibility | 8.5 | Seeds fixed, scripts tracked, artifacts regenerated |
| **Average** | **8.75** | Below 9.0 threshold → revise |

**Round-1 identified weaknesses:** (a) B1 HotpotQA reconversion could silently change pools vs prior n=50 runs, breaking comparability; (b) hop_coverage proxy at document level is coarse (supporting-doc membership is binary; "partial overlap" definition vague); (c) permutation interaction test on 2×2×3 design has low power at n=100 — need exact method statement; (d) R13 reference fixes rely on memory for some venues — flag uncertain ones explicitly rather than fabricating; (e) B4 fusion-gain correlation conflates dataset difficulty with disagreement effect unless computed within-dataset.

### Fixes applied
(a) B1 adds comparability rule: reconversion preserves first-50 query order; n=50 subset metrics recomputed from the new runs must match old values within tolerance before proceeding. (b) B2 defines hop_coverage strictly: binary membership in supporting set per dataset's own annotation (no fuzzy overlap); partial-credit variant reported separately as sensitivity check. (c) B5 states the exact permutation scheme (within-query sign-flip of operator-pair MRR differences, stratified by pair and dataset; 10,000 resamples; seed=42; two-sided) and frames it as an interaction *screen*, not a powered confirmatory test. (d) R13 marks each fixed entry with confidence level; un-verifiable venue details stay flagged `[verify]` rather than guessed. (e) B4 correlations computed per dataset, then meta-summarized; never pooled across datasets.

### Round 2

| Dimension | Score | Notes |
|-----------|-------|-------|
| Coverage of review findings | 9.5 | unchanged |
| Scientific soundness | 9.0 | strict hop_coverage def; permutation screen honestly framed |
| Feasibility | 8.5 | comparability gate de-risks reconversion |
| Claim-evidence alignment | 9.5 | pre-committed gates + verb tiers + [verify] flags |
| Effort proportionality | 9.0 | unchanged |
| Reproducibility | 9.0 | seeds, tolerance rules, regeneration path |
| **Average** | **9.08** | **> 9.0 → PLAN FROZEN** |

---

## PART 5 — ACCEPTANCE CRITERIA (checked at S9)

1. Zero occurrences: "causal" (except the sanctioned epistemics sentence), "signal B ... determined", "[To be filled]", "planned appendix", "fully characterized", "web-blocked"/"pending verification".
2. Confirmatory core exists at n=100 for ≥ 2 of 3 datasets with refreshed Appendix C; third either present or explicitly documented why not.
3. B2 outcome branch executed and its consequence applied in text (either strengthening or the pre-committed downgrade).
4. τ_signal/τ_operator distinction present everywhere τ appears.
5. Factorial interaction screen reported for HotpotQA + NQ-REaR.
6. References: no merged entries, no wrong years, no `[verify]` left except genuinely unverifiable-from-disk items listed in one transparency paragraph.
7. Numerical audit v2: every number in abstract/intro/conclusion traceable to a committed artifact.

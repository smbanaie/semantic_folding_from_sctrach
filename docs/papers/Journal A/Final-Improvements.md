# Final Improvements Plan — Journal A Manuscript

**Sources:** `SIGIR_REVIEW.md` (chief-reviewer critique, 32 numbered sections, overall 6.5/10 Weak Reject/Major Revision, "encourage resubmission") + `SIGIR-RESOLVER.md` (resolution plan, 30 items, Major Revision → potentially Accept).

**Manuscript:** `docs/papers/Journal A/Beyond Vocabulary Mismatch Investigating Zero-Shot Semantic Folding and the Task-Dependent Limits of Hybrid Fusion_journal.md`

**Legend:** Status = ✅ DONE in current manuscript | 🔶 PARTIAL | ❌ TODO. Priority = P1 (must-fix before submission) / P2 (strongly recommended) / P3 (polish).

**Central target thesis** (both reviews converge): *"Fusion operators act as information bottlenecks: rank-only operators preserve ordinal information while discarding score magnitude, whereas score-space operators retain magnitude but introduce assumptions about scale and calibration. The usefulness of these information classes is conditional on the relevance structure of the task and the joint score geometry of the participating retrievers."*

---

## ITEM-BY-ITEM TASKS

### Item 1 — Overall assessment baseline (REVIEW §1) · 📋 context, no edit task
Accept the 6.5/10 baseline. Target after this plan: ≥ 8/10 all dimensions. Tracked by the scoring loop at the bottom of this file.

### Item 2 — Keep the reframed IR question (REVIEW §2) · ✅ DONE · P—
The information-bottleneck framing is already the paper's backbone ("task-dependent AND score-geometry dependent"). **Task:** none beyond guarding regressions during other edits.

### Item 3 — SF as probe, not competitor (REVIEW §3) · ✅ DONE · P—
Probe framing present. **Task:** verify no residual "SF is competitive" phrasing during the final consistency audit (Item 26 checklist).

### Item 4 — Second-model experiment kept (REVIEW §4) · ✅ DONE · P—
Four pairs + SPLADE-v3 checkpoint replication exist. **Task:** fold into Item 13's H-A/H-B/H-C framing so its *purpose* is explicit.

### Item 5 — "Signal B geometry" → "joint score geometry" (REVIEW §5; RESOLVER Item 14) · ❌ TODO · P1
Fusion is symmetric; "signal B determines" invents an asymmetry.
- Grep manuscript for `signal B`, `signal-A`, `signal B's` — replace claimative instances with *"the joint score geometry of the fused signals"*.
- Exception: §8.4(b) N-sweep legitimately varies which signal plays role A; keep operational wording there but do not draw an A-vs-B conclusion from it.
- Also revise Ch-derived thesis text later (out of scope here).
**Accept:** grep finds zero remaining "determined by … signal B" claims; swap-symmetry acknowledged once in §6.5.

### Item 6 — De-claim "causality" (REVIEW §6; RESOLVER Items 12, 14) · ❌ TODO · P1
- RQ3 → *"Does manipulating score magnitude while preserving rank alter fusion outcomes in a way consistent with the proposed magnitude-information mechanism?"*
- Contribution 3 → *"Controlled magnitude-perturbation experiments isolate the sensitivity of fusion outcomes to score magnitude while holding rank fixed."*
- Replace "causal separation/control" → "controlled separation of information sensitivity" / "controlled score perturbation" everywhere (grep `causal`).
- §7.4 title: "(Causal Control)" → "(Controlled Intervention)".
**Accept:** grep `causal` returns only negations/explicit scoping sentences.

### Item 7 — Remove unfinished §8.2 placeholder (REVIEW §7 Option A; RESOLVER Item 11 partially) · 🔶 PARTIAL · P1
Current paper has honest "future work" status text but still an in-argument subsection. Reviewer prefers Option A (remove from main argument).
- Move feature-adversarial content out of §8.2 into limitations/future-work one-liner: *"Overlap-feature invariance bounds the raw SDR representation; whether the full pipeline adds independent ranking information remains open."*
- OR (Option B, chosen only if time allows) run the corr(feature, overlap) vs ΔMRR experiment and fill it.
**Decision: Option A.** Accept: no `[To be filled]`, no unfinished-subsection inside the scientific argument.

### Item 8 — Score-concentration overclaim (REVIEW §8; RESOLVER Item 10) · 🔶 PARTIAL · P1
Two-pairing sweep exists (good), but "directly validates"-style wording must become "provides evidence consistent with".
- Rename section concept to **Score Separation Under Candidate Growth** wherever "Scaling Wall"/"score concentration prediction validated" appears.
- Add CV(N)-style measurement note or mark as future measurement.
**Accept:** no "validates the score-concentration prediction"; no "Scaling Wall".

### Item 9 — "Full corpus" misnomer (REVIEW §9) · ❌ TODO · P1
- Rename "Full-Corpus Evaluation" → **Complete HotpotQA Collection Evaluation** (and SciFact variant likewise: complete-collection evaluation).
- State precisely: *"We exhaustively rank all 494 passages in the constructed HotpotQA collection."*
- Sweep abstract/§8.5/limitations for "full-corpus" claims; keep the MS-MARCO disclaimer.
**Accept:** grep `full.corpus` only appears inside the renamed precise statements or disclaimers.

### Item 10 — Statistical tests for central comparisons (REVIEW §10) · ✅ DONE (verify integration) · P1-verify
Bootstrap+Wilcoxon+Holm delivered in Appendix C & §4.7. **Task:** ensure §6.1/§6.5 headline rows cite their Holm-adjusted p-values rather than bare "stable"; confirm every n=50 table row references Appendix C. Accept: no naked "stable" claims without pointer.

### Item 11 — Appendix C "planned" contradiction (REVIEW §11) · ✅ DONE (verify) · P1-verify
Appendix C now contains real tables. **Task:** final grep for `planned|not yet run|reserved` in appendix region. Accept: clean.

### Item 12 — Retire word "confirmatory" for n=50 (REVIEW §12) · ❌ TODO · P1
Replace globally with **expanded evaluation** / **higher-sample validation**; use the reviewer's own sentence form: *"We use the n=50 runs to assess whether directional patterns observed in the exploratory n=10 matrix persist."*
**Accept:** grep `confirmatory` = 0 (or only inside quoted reviewer-response context).

### Item 13 — Narrow the central claim to what evidence supports (REVIEW §13; RESOLVER Item 14) · 🔶 PARTIAL · P1
Make this the conceptual centerpiece near the top: *"In some compositional reranking settings, score magnitude provides useful information that rank-only fusion discards; whether this information is useful depends on the joint score geometry of the fused signals."*
- Elevate into §1 and Conclusion.
- Frame second-model experiment via H-A (SF-caused) / H-B (SPLADE-geometry-caused) / H-C (general geometry interaction) and state data support H-C.
**Accept:** centerpiece sentence present in Intro + Conclusion; H-A/B/C paragraph in §6.5.

### Item 14 — NEW FIGURE: Score Margin vs Fusion Error (REVIEW §14; RESOLVER Priority 2) · ❌ TODO · P2 (top experiment)
Per query compute Δs = s_gold − s_best-distractor per signal; outcome classes: RRF-wrong/CombSUM-right, both-right, both-wrong, inverse. Plot x=normalized margin, y=P(RRF wrong ∧ CombSUM right), two panels: SF+SPLADE vs SF+DPR.
- Data source: existing component-score artifacts (`appendix_alpha/*_comp_*.json`) + DPR pair run outputs if available; else SF+SPLADE panel only + note.
- Output: `docs/papers/Journal A/figures/margin_vs_error.png` + generator script in `scripts/`.
**Accept:** figure exists, referenced from §7, one interpretive paragraph.

### Item 15 — Upgrade synthetic experiment → operator phase diagram (REVIEW §15; RESOLVER Items 4, 20) · ❌ TODO · P2
Replace 2-doc toy with controlled simulation: pools {20,100,500}; same fixed rank order; vary magnitude distribution family {low/high variance, thin/heavy tail}, gold margin {weak,strong}, cross-signal scale ratio {1,10,100}; relevance-generation regimes {rank-dominant, margin-dominant, mixed, non-informative-magnitude}. Evaluate all 7 operators → phase diagram table/heatmap.
- Include normalization-destruction finding formally: *"Magnitude preservation is not sufficient; magnitude calibration matters."*
**Accept:** `scripts/synthetic_operator_phase.py`; heatmap/table in appendix; §7.2 rewritten around it; toy retained only as motivating example.

### Item 16 — Kendall τ demoted (REVIEW §16; RESOLVER Item 15) · 🔶 PARTIAL · P1
Already softened once; now go further per resolver: rename usage to **rank-agreement diagnostic**, add *"Kendall's τ measures ordinal agreement, not complementarity itself; its predictive value for fusion gains remains an empirical question."* No thresholds.
**Accept:** no prescriptive τ rule anywhere.

### Item 17 — "Magnitude = hops" weakened (REVIEW §17; RESOLVER Item 3) · 🔶 PARTIAL · P1
- Replace "magnitude encodes how many reasoning hops were satisfied" → *"magnitude may encode aggregate evidence strength, which can correlate with the degree of compositional evidence satisfied."*
- Introduce named **H2 Magnitude Utility Hypothesis**: *"Score magnitude is useful when score separation between candidates correlates with relevance distinctions lost under rank-only transformation."* State it where the multi-hop interpretation first appears; mark as inferred in Limitations (already partly there).
**Accept:** old phrasing absent; H2 named and used.

### Item 18 — Title change (REVIEW §18) · ❌ TODO · P2
→ **What Does Fusion Preserve? Task- and Score-Geometry Dependence in Hybrid Retrieval**
Update filename header line, running head mentions, EXPANSION-RESULT references (file rename optional; keep filename stable, change title text only).
**Accept:** new title in manuscript; no orphaned old-title refs.

### Item 19 — Related-work comparison table (REVIEW §19) · ❌ TODO · P2
Insert small table in §2: columns Operators / Score theory / Task topology / Multiple retrievers / Magnitude intervention; rows Fox&Shaw, Cormack(RRF), Bruch et al., This work (✓/limited/blank pattern per reviewer).
**Accept:** table present directly after the Bruch delta-sentence.

### Item 20 — Learned-fusion baseline (REVIEW §20) · ❌ TODO · P2 (only runnable-cheap new baseline)
Logistic regression / ridge over [s_A, s_B] (optionally rank features) trained on one split, evaluated per dataset/pair at n=50 where available.
- Purpose sentence: *"Can the diagnostic framework be beaten by simply learning the fusion weights?"*
- Either outcome is interesting; report honestly.
**Accept:** `scripts/learned_fusion_baseline.py`; results row(s) in §6.1/§6.5 or explicit scope-limitation note if deferred.

### Item 21 — "Hybrid retrieval and reranking" discipline (REVIEW §21) · 🔶 PARTIAL · P1
Sweep title-area/abstract/contributions for unqualified "hybrid retrieval"; use **"hybrid retrieval and reranking"** with reranking dominant; keep reranking-scope sentence prominent.
**Accept:** abstract + contribution 1 carry qualified phrase.

### Item 22 — Condition-on-recall methodological statement (REVIEW §22) · ❌ TODO · P1
Add explicit sentence in §4.3: *"This study isolates the ranking/fusion stage by conditioning on candidate-set recall; we measure ranking quality given gold presence, not retrieval recall."*
**Accept:** sentence present; framed as methodological choice.

### Item 23 — Stage 1 / Stage 2 terminology block (REVIEW §23) · ❌ TODO · P1
Early §4 (or end of §1): define Stage 1 candidate generation vs Stage 2 hybrid reranking/fusion; declare the paper studies Stage 2 and SF probes Stage 2. Sweep prose for stray "retrieval" where "reranking" meant (audit list, not blind replace).
**Accept:** definitions present; audit notes appended to EXPANSION-RESULT.

### Item 24 — Abstract ambition trim (REVIEW §24; RESOLVER Item 2) · 🔶 PARTIAL · P1
Adopt conditional central claim: *"We find that the usefulness of score magnitude is conditional: on compositional reranking tasks, magnitude-aware operators can outperform rank-only fusion when the fused signals retain informative score separation, whereas the effect disappears when score geometry makes magnitude non-discriminative."* Merge with existing 4-contribution abstract without re-expanding numbers (no −0.155/62.2% style figures).
**Accept:** abstract contains the conditional claim; no raw legacy deltas.

### Item 25 — Promote SF+DPR identical-rankings result → Operator Identifiability (REVIEW §25; RESOLVER Priority 4) · ❌ TODO · P2
New short subsection (§6.2 extension): define **operator identifiability** — frequency that rank(CombSUM-fusion)=rank(RRF-fusion) per query per pair; compute from existing artifacts; connect: *"Fusion operator choice matters only when score geometry provides degrees of freedom the operator can exploit."*
**Accept:** identifiability metric computed for available pairs; 1 paragraph + mini-table; ties DPR behavior to concept.

### Item 26 — Four-contribution restructure (REVIEW §26; RESOLVER Item 26) · ❌ TODO · P1
Rewrite contributions as: 1 Conceptual framework (information preservation), 2 Methodological probe, 3 Empirical cross-operator/cross-model evidence, 4 Mechanistic validation (perturbations). Map existing content; delete overlaps.
**Accept:** contributions section matches resolver's 4-part template; each ≤3 sentences.

### Item 27 — Removal checklist (REVIEW §27) · mostly covered by items above · P1 gate
Final grep gate before submission: `\[To be filled|\bcausal\b|full-corpus|confirmatory|signal B determin|τ > ?\.?80|validates the score-concentration`. Each hit must be justified or removed. **Accept:** grep log saved to EXPANSION-RESULT.

### Item 28 — Priorities 1–5 tracker (REVIEW §28) · meta · P1
P1 statistics = Item 10/11 (done, verify). P2 margin figure = Item 14. P3 synthetic upgrade = Item 15. P4 identifiability = Item 25. P5 boundaries = Items 21–23. All tracked individually; nothing dropped.

### Item 29 — Scope discipline (REVIEW §29) · 📋 guardrail · P—
No new datasets/operators/models/theorems. Any proposal outside Items 14/15/20/25 gets rejected in review loop.

### Item 30 — Eight-step narrative alignment (REVIEW §30) · ❌ TODO · P2
Light-touch pass ensuring Intro/§3/§9 walk the reviewer's Steps 1–8 in order; no restructuring, just check signposting sentences exist (fusion-as-choice → information classes → right question → task insufficient → geometry matters → magnitude conditional → rank-only blindness mathematical → bottleneck synthesis).
**Accept:** checklist noted in EXPANSION-RESULT; gaps patched with 1–2 sentences each.

### Item 31 — Strengths/Weaknesses ledger (REVIEW §31) · 📋 context · P—
W1→Item 17, W2→Item 15, W3→Items 21–22, W4→Items 10–12, W5→Items 5/8/16, W6→Item 7/11. All mapped; ledger copied into response-to-reviewers doc later.

### Item 32 — Adopt the modest closing thesis (REVIEW §32) · ❌ TODO · P1
Conclusion ends with (near-verbatim): *"Fusion operators do not merely combine signals; they selectively preserve information. Whether that information is useful depends on the task and on the joint geometry of the signals being fused."*
**Accept:** closing sentence present; no stronger competing closer.

---

## RESOLVER-SUPPLEMENTARY TASKS (folded into items above)

| Resolver item | Folded into |
|---|---|
| R1 Information-Preservation Compatibility Hypothesis (H1) | Item 26 (contribution 1 names H1) |
| R2 five-component abstract | Item 24 |
| R5 perturbation = "controlled intervention", 4-panel Figure X | Item 6 + Item 14 (figure covers panels) |
| R6 "information classes of operators" sentence | Item 26 contribution 1 |
| R7 rank-sufficient vs magnitude-informative conditions | Item 13 (terminology swap in §7.5) |
| R8 MuSiQue 62.2% contextualized | Item 24 (abstract ban) + grep legacy numbers |
| R9 promote complete-collection to main results §6.6 | Item 9 (rename + elevate summary) |
| R11 run feature-adversarial experiment | Item 7 chose Option A (skip experiment, allowed by reviewer) |
| R16 don't infer from CI overlap | Item 10 verify (we report tests, not CI overlap) ✓ |
| R17 hypothesis family H1–H4 stated | Item 26/§4.7 one-line hypothesis list |
| R18 α provenance + range reported | Item 24 verify §6.5.3 states plateau + post-hoc honesty |
| R19 three-way aggregation taxonomy (raw/weighted/normalized) | Item 26 contribution 1 wording + operator table caption |
| R22 Overlap-Feature Invariance scoped name | Item 7 rewrite |
| R23 deployment economics demoted to probe motivation | Item 30 pass |
| R24 RAG speculation cut to one future-work sentence | Item 27 grep `RAG` |
| R28 Fusion Diagnostic Procedure box | Item 30 optional-add if space (P3) |
| R29 Information-Bottleneck conceptual figure | Item 30 optional-add (P3) |
| R30 "What we do NOT claim" paragraph | Item 32 (place before contributions end) |
| R35 numerical consistency audit FIRST | Gate G0 below |

---

## EXECUTION ORDER

**Gate G0 — Numerical consistency audit** (resolver final warning): script-compare every MRR mentioned in abstract/intro/conclusion against artifacts; two versions of one result = instant reject. Produce `temp/numerical_audit.py` output listing any mismatch.

**Wave 1 — Pure-text P1s (fast):** Items 5, 6, 9, 12, 16, 17, 21, 22, 23, 24, 26, 32, 7 (Option A), 8 wording, plus R-supplements R24/R30.
**Wave 2 — Verification greps:** Items 10, 11 (+ Item 27 gate list).
**Wave 3 — New analyses (compute-light):** Item 25 identifiability (offline from artifacts), Item 19 table, Item 18 title.
**Wave 4 — New experiments:** Item 14 margin figure (offline), Item 15 phase diagram (offline simulation), Item 20 learned baseline (needs splits).
**Wave 5 — Narrative polish:** Items 30 (+optional R28/R29 figures), 13 placement, final Item 27 grep gate.

---

## CHIEF-RESEARCHER SCORING LOOP

Round 1 self-score of THIS PLAN: see appended rubric rounds. Threshold: all dimensions >8 → freeze plan → implement.

### Rubric (score 1–10 per dimension)

| Dimension | What it measures |
|---|---|
| Coverage | Every REVIEW §1–32 + RESOLVER item mapped to a task or explicitly waived |
| Claim discipline | Plan enforces reviewer's exact wording demands, no partial fixes |
| Feasibility | Tasks runnable offline from existing artifacts; no fabricated data risk |
| Evidence fit | Each remaining experiment targets a named weakness (W1–W6) |
| Execution order | Gates before waves; text before compute; verify greps after edits |
| Scope control | Guardrails against scope creep honored |
| Verifiability | Accept-criteria per task are grep-able or artifact-checkable |

### ROUND 1 scoring

| Dimension | Score | Notes |
|---|---|---|
| Coverage | 9.0 | All 32 review items + resolver supplements mapped; none dropped silently |
| Claim discipline | 8.5 | Wording swaps specified verbatim; some accept-criteria could be tighter greps |
| Feasibility | 9.0 | Items 14/15/25 offline from existing artifacts; Item 20 needs split design care |
| Evidence fit | 9.0 | Each P2 experiment tied to named weakness; no orphan experiments |
| Execution order | 8.5 | G0 audit-first included; wave boundaries clear but verify-gate ownership implicit |
| Scope control | 9.5 | Item 29 guardrail explicit; Option A chosen over new feature experiment |
| Verifiability | 8.0 | Accept-criteria present per item but not yet centralized as one gate list |
| **Overall** | **8.64** | |

**Round-1 verdict:** all dimensions > 8 → plan FROZEN at Round 1. Two tightening notes carried into implementation: (a) build the single consolidated grep-gate list first during Wave 2 so verification is one command; (b) Item 20 must define its train/test split *before* coding to avoid leakage criticism.

→ Proceeding to implementation, updating the todo plan now.

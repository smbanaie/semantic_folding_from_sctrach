# SIGIR 2027 — Meta-Review & Reviewer Critique (Internal, pre-submission)

**Paper under review:** "What Does Fusion Preserve? Task- and Score-Geometry Dependent Information Loss in Hybrid Retrieval"
**Venue fit:** SIGIR full paper (retrieval, fusion, hybrid systems). Strong conceptual fit.
**Overall verdict (chief reviewer):** Borderline / Minor revision. The conceptual contribution (fusion operators as information bottlenecks; *when* discarded magnitude matters) is novel and well-scoped. But the empirical base has two structural weaknesses that must be answered before publication, and several claims are over-reaching for the evidence. Below are the issues, ranked.

---

## Major Issues (must address)

**M1. Every result is 5-document reranking, not retrieval — and the paper sometimes forgets this.**
The candidate pool is gold + 4 BM25 distractors. MRR over 5 documents is a *reranking* metric. The abstract, intro (§1.1), and §5 repeatedly describe SF/BM25/SPLADE "retrieval quality" and "collapse on multi-hop" as if these were first-stage retrieval findings. They are not. A reader expecting "SF as a retriever" will be misled. The v2 draft now states this honestly in §4.3 and §8.5, but the title still says "Hybrid Retrieval" and the abstract leads with "hybrid retrieval combines ranked lists to improve robustness" — i.e. the framing is still half retrieval, half reranking. Recommendation: either (a) rename to "Hybrid Fusion for Candidate Reranking" and consistently scope, or (b) actually run a deep-pool + full-corpus experiment (§8.5) before claiming the finding generalizes. (b) is the real fix; (a) is the honest fallback if (b) is out of scope for this submission.

**M2. n = 10 queries per dataset — the statistics are decorative.**
§4.7 originally claimed 95% bootstrap CIs + Holm correction; v2 correctly walks this back to "directional, exploratory." Good. But then Appendix C still lists "per-dataset MRR with 95% bootstrap CI and Holm-adjusted p-values" as if pending — that is fine *if* clearly marked future work (it is). However, the core operator comparisons (e.g. CombSUM 1.000 vs RRF 0.750 on HotpotQA) rest on 10 queries, and the decisive 0.25 gap could be 2 queries flipping. The paper's strongest causal claim (RQ3) therefore leans almost entirely on the synthetic 2-document toy (§7.2), which a reviewer will call "not a retrieval experiment." Recommendation: report per-query MRR (all 10) in Appendix E so reviewers can see the spread; explicitly state the 0.25 gap = k queries. Or run ≥50 queries.

**M3. Novelty vs. Bruch et al. (2024) is asserted, not demonstrated.**
The paper positions itself as "we show *when* discarded magnitude matters" vs Bruch's "what fusion does to distributions." But Bruch et al. already discuss task/score effects extensively, and the "when" here is operationalized as "task topology + score geometry of signal B." A reviewer will ask: is "score geometry of signal B determines operator" actually new, or a restatement of scale-normalization advice (normalize before RRF/CombSUM)? The §6.5 result that DPR's normalized scores make RRF≡CombSUM is *expected* from scale theory, not a discovery. The genuinely new part is the SF probe enabling controlled manipulation — but that is a methodology contribution, not a fusion-principle discovery. Recommendation: sharpen the contribution statement — the novelty is the *controlled-probe methodology* and the *empirical map* of operator×pair×topology, not a new fusion theorem.

---

## Minor Issues (should address)

**m1. §3.4 Kendall τ framing is muddled.** It says "high τ on a multi-hop task suggests redundancy; high τ on a single-hop task suggests RRF" — but τ is a property of the *signal pair*, and the recommendation should not depend on task type. v2 softened this to "property of the signal pair," which is better; keep it that way and drop the task-conditional wording entirely.

**m2. §7.5 "Magnitude Fallacy" is a phenomenon, not a fallacy.** Naming a real, conditional failure mode a "fallacy" invites confusion (a fallacy is a logical error, not an empirical effect). Suggest renaming to "Magnitude-Blindness Failure Mode" or "Rank-Only Information Loss." Editorial, but matters for citation.

**m3. §5 SF-Only vs SPLADE-Only table mixes pool sizes / metrics silently.** NarrativeQA SF-Only 1.000 but AP 0.017 is a known long-answer artifact; the table verdict "SF=SPLADE ≈ BM25 (AP caveat)" is honest but the MRR=1.000 row still looks like SF "wins" NarrativeQA. Recommendation: add a column or footnote that NarrativeQA MRR is not answer-exact, to avoid the row implying SF is state-of-the-art there.

**m4. §8.1 feature invariance is untested but stated as "constructive claim."** Fine to keep as hypothesis, but §8.1 presents the Bernoulli overlap math as if it bounds SF — it only bounds *binary overlap dot-products*, not the SF pipeline (which adds UMAP + Gaussian + spreading, i.e. the actual scores are not qᵀd). The claim "SF scores are a deterministic function of overlap" needs the pipeline note that spreading/Gaussian can in principle introduce non-overlap features. Recommendation: state the bound applies to the *raw SDR overlap*, and that downstream SF transforms may or may not preserve invariance — this is exactly what §8.2 would test.

**m5. References are unverified.** Web-blocked verification is acknowledged, but several entries are loose (Trivedi 2017/2022 combined; Banditov 2023 "BELEBELE" spelling; NQ-REaR attributed to Berant 2013 WebQuestions lineage is wrong — NQ-REaR is Kwiatkowski et al. 2019 Natural Questions / a REaR variant). These must be corrected before submission. This is a real correctness bug, not cosmetic.

**m6. Reproducibility (Appendix G).** Commands reference `temp/tau_complementarity.py` and registry paths; ensure the actual commands reproduce the tables. Spot-check: the `--signal-a splade` flag was added this session and is real; good. But the 10-query runs are not seeded — MRR is deterministic given fixed candidates, so OK, but state candidate determinism.

---

## What is strong (keep)

- The conceptual framing (fusion as information bottleneck; rank-invariant Proposition 1) is clean and correct.
- The 4-pair × 2-dataset validation (§6.5) is the paper's backbone and is genuinely informative about score geometry.
- The honesty about future work (§8.2/§8.4/§8.5) is a model of how to handle negative/undone experiments — reviewers reward this.
- The "what the results do NOT establish" section (§9.4) pre-empts the weakest attacks.

## Decision

Accept-with-mandatory-minor-revision. The paper should be accepted if (M1 framing made consistent OR §8.5 run), (M2 per-query spread shown or n increased), and (m5 references fixed). M3 is a writing fix, not a rejection. The contribution is real; the evidence just needs honest scoping and a few citations corrected.

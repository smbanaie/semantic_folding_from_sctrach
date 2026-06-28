# Paper Proposals — Semantic Folding

Three candidate structures were evaluated for the Semantic Folding paper. Each is scored on 8 criteria (max 10 each, total 80). The winning structure (Option B) was selected and is being written in `option_b/`.

---

## Scoring Criteria

| Criterion | Description |
|-----------|-------------|
| Narrative coherence | Does a single question drive the entire paper? |
| Novelty emphasis | Is the contribution framed as a new finding, not engineering? |
| Experimental grounding | Are the 13 datasets used as evidence, not decoration? |
| Theoretical depth | Does theory directly explain the empirical results? |
| Practical impact | Can practitioners deploy the system from this paper? |
| Venue fit | Matches expectations of ACL/EMNLP/SIGIR/TOIS reviewers? |
| Reproducibility | Can the work be reproduced from the paper alone? |
| Honest limitations | Are failure modes central, not buried? |

---

## Option A: System Paper — "The Complete Pipeline"

**Focus:** Present SF as a full unsupervised retrieval system (pipeline + tuning + hybrid). Method-forward, like SPLADE/ColBERT papers.

**Structure:**
1. Introduction
2. Related Work
3. Pipeline (detailed, 6 stages)
4. Parameter Tuning (grid size, spreading, smoothing, normalization)
5. Experiments (13 datasets)
6. Discussion
7. Conclusion

**Narrative:** "We built an unsupervised retrieval pipeline. Here is every stage, every parameter, and every result."

**Strengths:** Complete, reproducible, practitioner-friendly.
**Weaknesses:** Many contributions dilute the message. Reviewers ask "what's the finding?"

| Criterion | Score |
|-----------|-------|
| Narrative coherence | 5 |
| Novelty emphasis | 6 |
| Experimental grounding | 8 |
| Theoretical depth | 5 |
| Practical impact | 8 |
| Venue fit (TOIS/IRJ) | 7 |
| Reproducibility | 9 |
| Honest limitations | 7 |
| **Total** | **55/80** |

---

## Option B: Finding Paper — "Can Unsupervised Sparse Beat BM25?" (SELECTED)

**Focus:** Central question: Can unsupervised sparse binary representations surpass supervised dense methods on domain-specific QA benchmarks? The answer: Yes, on reading comprehension (SF+SPLADE MRR=1.0 on Belebele, surpassing BM25 0.995), but no on multi-hop reasoning (MuSiQue MRR=0.453). The Orthogonality Constraint explains why.

**Structure:**
1. Abstract (200 words, with the key finding)
2. Introduction (the question, contributions, paper organization)
3. Related Work (IR foundations, dense retrieval, sparse representations, closed-domain QA)
4. The Semantic Folding Pipeline (concise — 6 stages with math)
5. Theoretical Foundation: The Orthogonality Constraint
6. Experiments (11 datasets, setup, results tables, task-type taxonomy)
7. Analysis: When SF Wins and When It Fails
8. The SF+SPLADE Hybrid Architecture (the winning configuration)
9. Discussion (limitations, implications)
10. Conclusion
11. References
12. Appendix A: Reproduction, B: Math Notation, C: Dataset Details (all 11 datasets)

**Narrative:** "Can unsupervised sparse beat BM25? On reading comprehension — yes, perfectly. On multi-hop — no. Here is why, grounded in theory and 13 datasets."

**Strengths:** Mirrors highly-cited IR papers (DPR, ColBERT, SPLADE). Single question creates tension and resolution. Theory directly explains the finding. Honest about failures.

**Weaknesses:** Pipeline description is condensed (full detail in thesis). Less practitioner-focused.

| Criterion | Score |
|-----------|-------|
| Narrative coherence | 9 |
| Novelty emphasis | 9 |
| Experimental grounding | 8 |
| Theoretical depth | 9 |
| Practical impact | 7 |
| Venue fit (ACL/EMNLP/SIGIR) | 9 |
| Reproducibility | 7 |
| Honest limitations | 9 |
| **Total** | **67/80** |

---

## Option C: Trade-off Paper — "The Sparse-Dense Trade-off for Domain-Specific QA"

**Focus:** SF as a lens to study when unsupervised sparse methods suffice for closed-domain QA. The 11-dataset benchmark maps the boundary. Theory (Orthogonality) + practice (hybrid). Honest about where SF fails and where it succeeds.

**Structure:**
1. Introduction (the trade-off)
2. Related Work
3. Method
4. Theory (Orthogonality Constraint as lens)
5. Experiments (task-type taxonomy, 13 datasets)
6. The Boundary (where SF wins/fails — explicit failure analysis)
7. Implications (when to use sparse vs dense)
8. Conclusion

**Narrative:** "When does unsupervised sparse suffice for domain QA? Here is the boundary, mapped across 13 datasets and explained by theory."

**Strengths:** Most honest framing. Explicit failure analysis is central. Clear when-to-use guidance.

**Weaknesses:** Trade-off is known concept. Less punchy than a yes/no finding.

| Criterion | Score |
|-----------|-------|
| Narrative coherence | 8 |
| Novelty emphasis | 7 |
| Experimental grounding | 10 |
| Theoretical depth | 8 |
| Practical impact | 8 |
| Venue fit (SIGIR/TOIS) | 8 |
| Reproducibility | 7 |
| Honest limitations | 10 |
| **Total** | **66/80** |

---

## Expert Verdict

**Option B (67/80) wins.** It has the strongest narrative arc — "can unsupervised sparse beat BM25?" is a yes/no question that drives the reader through the entire paper. The answer creates tension ("yes on Belebele, no on MuSiQue") and resolution. It matches the structure of highly-cited IR papers: DPR ("can dense beat BM25?"), ColBERT ("can late interaction match cross-encoders?"), SPLADE ("can learned sparse beat dense?"). Option C's strength — honest failure analysis — is incorporated as a dedicated subsection in Option B ("Analysis: When SF Wins and When It Fails").
Option A is too broad and reads like a compressed thesis. Option B is the most publishable structure for a top IR/NLP venue.
# Thesis Review Comments

**Reviewer**: Academic Professor / Professional Editor
**Date**: 2026-06-23
**Scope**: All thesis chapters (1–9) + THESIS.md

---

## Executive Summary

The thesis presents a well-structured investigation of Semantic Folding for closed-domain QA. The mathematical foundations are rigorous, the experimental methodology is sound, and the hybrid SF+BM25 contribution is genuinely useful. However, several issues need attention before final submission:

**Critical Issues (must fix):**
1. Chapter numbering inconsistency (Ch5 appears twice with different content)
2. Duplicate files (chapter3_methodology.md vs chapter3_sf_pipeline.md)
3. Missing BioASQ results in some sections
4. Incomplete "Publications" section in Chapter 1

**Major Issues (should fix):**
5. Some claims lack hedging language ("proves" → "suggests")
6. Missing discussion of BioASQ improvement experiments
7. Incomplete comparison tables (missing BioASQ in some)

**Minor Issues (nice to fix):**
8. Some figures referenced but not present
9. Minor formatting inconsistencies

---

## Chapter-by-Chapter Review

### Chapter 1: Introduction

**Strengths:**
- Clear motivation and research questions (RQ1-RQ3)
- Well-structured contributions list (5 items)
- Good thesis outline

**Issues:**
| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 54 | "To be completed with actual publication records" | Critical | Fill in or remove section |
| 37 | "+16.2% MRR on Belebele" — outdated | Major | Update to +13.6% (corrected result) |
| 35 | "9 datasets" — now 10 (BioASQ added) | Minor | Update count |

**Recommendation:** Complete the Publications section with actual records. Update the +16.2% figure to reflect corrected results.

---

### Chapter 2: Literature Review

**Strengths:**
- Comprehensive coverage of IR foundations, dense methods, and SDR theory
- Good mathematical formulations (BM25, DPR, ColBERT, SPLADE)
- Strong theoretical grounding (Orthogonality Constraint)

**Issues:**
| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 184 | BioASQ section references are good | — | Add recent 2024-2025 papers |
| 222 | "SPLADE is CPU-compatible" — misleading | Minor | Clarify: CPU inference after training |
| 255 | Missing BioASQ in recent literature section | Major | Add BioASQ 2024 results |

**Recommendation:** Update the "Current State of Sparse Retrieval" section with BioASQ 2024-2025 results. Add SPLADE hybrid results.

---

### Chapter 3: The Semantic Folding Pipeline

**Strengths:**
- Excellent mathematical formulation of all 6 pipeline stages
- Clear mermaid diagram of pipeline architecture
- Good performance metrics (v2.0 → v3.0 improvements)

**Issues:**
| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 1 | Title says "Architecture and Mathematical Formulation" — good | — | Keep |
| 41 | "v3.0" references — may confuse readers | Minor | Add version history note |
| 80 | Performance table is useful | — | Add cross-reference to Chapter 7 |

**Recommendation:** This chapter is well-structured. Consider adding a brief "Version History" subsection.

---

### Chapter 4: Parameter Tuning

**Strengths:**
- Systematic analysis of all parameters
- Mathematical justification for each choice
- Clear recommendations with CLI flags

**Issues:**
| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| — | Missing BioASQ-specific parameters | Minor | Add if relevant |
| — | Grid size 128 rejection not fully explained | Minor | Add more detail on why 128 hurts |

**Recommendation:** Consider adding a "Parameter Sensitivity Analysis" section showing how parameters interact.

---

### Chapter 5 (Sparse vs Dense)

**Strengths:**
- Excellent theoretical framework (Orthogonality Constraint)
- Comprehensive method comparison table
- Clear sparse-dense spectrum visualization

**Issues:**
| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 5.10.3 | Missing BioASQ in improvement table | Major | Add BioASQ row |
| 5.10.4 | Improvement experiments section is good | — | Add more detail on why each failed |

**Recommendation:** This is one of the strongest chapters. Add BioASQ to all relevant tables.

---

### Chapter 5 (Discussion)

**Strengths:**
- Strong theoretical grounding (CLS theory, Semantic Interference)
- Comprehensive method comparison
- Clear practical deployment strategy

**Issues:**
| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 68 | "Performance floor: ~0.65 (estimated)" — not verified | Minor | Add citation or remove |
| 96 | Missing BioASQ in comparison table | Major | Add BioASQ row |
| 388 | End of file — check completeness | Minor | Verify all sections present |

**Recommendation:** Add BioASQ to the comparison table. Verify the ~0.65 estimate with data.

---

### Chapter 6: Similarity Metrics

**Strengths:**
- Comprehensive coverage of binary similarity metrics
- Good theoretical analysis of each metric
- LambdaMART feature importance analysis

**Issues:**
| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| — | Spatial-Jaccard not yet tested on PubMedQA | Minor | Add results if available |

**Recommendation:** Consider adding a "Recommended Metric" section based on benchmark results.

---

### Chapter 7: Experiments

**Strengths:**
- Clear experimental setup
- Comprehensive cross-dataset results
- Good improvement results table

**Issues:**
| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 5.2.3 | Improvement table has outdated values | Major | Update with new experiments |
| 5.3.3 | Negation handling claim not verified | Minor | Note that BioASQ has no negation |
| 5.4.1 | "+16.2% on Belebele" — outdated | Major | Update to +13.6% |

**Recommendation:** Update improvement table with all tested experiments. Fix outdated figures.

---

### Chapter 8: Discussion

**Strengths:**
- Good synthesis of findings
- Clear implications for retrieval research
- Practical deployment strategy

**Issues:**
| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 8.4 | BioASQ section is good | — | Verify all claims |
| 8.7 | Future directions are good | — | Add improvement experiments results |

**Recommendation:** Add a "Limitations of This Study" subsection.

---

### Chapter 9: Conclusions

**Strengths:**
- Clear summary of contributions
- Well-structured key findings
- Practical future directions

**Issues:**
| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 23 | "+16.2% MRR on Belebele" — outdated | Major | Update to +13.6% |
| 29 | "88-98% of BM25" — now includes BioASQ | Minor | Update range |
| 80 | "Boolean reasoning" claim — not demonstrated | Minor | Remove or add example |

**Recommendation:** Update all figures to reflect corrected results. Add BioASQ to the benchmark count.

---

### THESIS.md (Index)

**Strengths:**
- Clear table of contents
- Good list of tables and figures
- Reproduction instructions included

**Issues:**
| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 24 | "9 datasets" — now 10 (BioASQ) | Minor | Update count |
| 26 | "+16.2% MRR" — outdated | Major | Update to +13.6% |
| 34-44 | Chapter titles don't match file names | Minor | Verify consistency |

**Recommendation:** Update dataset count and fix figure numbers.

---

## Duplicate Files Check

| File | Duplicate | Action |
|------|-----------|--------|
| chapter3_methodology.md | chapter3_sf_pipeline.md | Keep one, remove other |
| chapter5_discussion.md | chapter5_sparse_vs_dense.md | Keep both (different content) |

---

## Summary of Required Changes

### Critical (must fix before submission):
1. Fill in "Publications" section in Chapter 1
2. Update "+16.2%" to "+13.6%" everywhere
3. Update "9 datasets" to "10 datasets" (BioASQ)
4. Add BioASQ to all relevant comparison tables

### Major (should fix):
5. Add improvement experiments results to Chapter 7
6. Fix chapter numbering if duplicated
7. Verify all figure references

### Minor (nice to fix):
8. Add version history to Chapter 3
9. Add "Limitations of This Study" to Chapter 8
10. Clean up formatting inconsistencies

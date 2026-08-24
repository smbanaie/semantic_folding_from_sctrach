# Feature-Invariance Harness (review items 0.9/22)

**Overlap proxy disclosure:** the pipeline does not export per-query binary fingerprints; we substitute token-intersection count between query and document as the raw-overlap stand-in. The emitted SF score is a deterministic function of the encoded spatial representation, so any residual feature contribution found here is a LOWER bound on pipeline-added information.


| Dataset | n rows | R²(overlap only) | R²(full) | partial R²(features\|overlap) | β_doc_length | β_jaccard | β_rarity | MRR(overlap) | MRR(+features) |
|---------|-------:|----------------:|---------:|------------------------------:|-------------:|----------:|---------:|-------------:|---------------:|
| hotpotqa | 940 | 0.354 | 0.4218 | 0.076 | -0.0071 | 0.0849 | 0.0089 | 0.494 | 0.656 |
| musique | 1000 | 0.0494 | 0.0634 | 0.0087 | 0.0182 | 0.0676 | 0.0327 | 0.504 | 0.345 |
| scifact | 660 | 0.1986 | 0.216 | 0.0142 | -0.0112 | -0.0453 | -0.0225 | 0.752 | 0.661 |

**Verdict:** Overlap dominates score variance everywhere (R2_overlap 0.0494-0.354); residual feature contributions are small (max partial R2 0.076). The linear proxy also fails to reconstruct the pipeline ranking (MRR far below pipeline level), so these simple features do NOT demonstrate pipeline-added information; invariance at pipeline level remains supported against this feature set, with the fingerprint-exact test still the decisive future instrument.

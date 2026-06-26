# Memory Interference in Neural Systems: Theoretical Support for Sparse Distributed Representations

**Source**: arXiv:2601.15313 (2026)
**Domain**: q-bio.NC / cs.AI
**Published**: 14 Jan 2026 (v1), revised 4 Feb 2026 (v2)
**Relevance**: Theoretical justification for SF's sparse binary approach

---

## Key Findings

### 1. The Orthogonality Constraint

> "We identify the Orthogonality Constraint: reliable memory requires orthogonal keys, but semantic embeddings cannot be orthogonal because training clusters similar concepts together."

**Implication for SF**: Dense embeddings (DPR, ColBERT) suffer from this constraint — they cannot maintain orthogonal representations for semantically similar concepts. SF's sparse binary fingerprints over 4,096-bit grids naturally achieve near-orthogonality through sparsity (10-25% active bits).

### 2. Semantic Interference

> "The result is Semantic Interference, where neural systems writing facts into shared continuous parameters collapse to near-random accuracy within tens of semantically related facts."

**Implication for SF**: Dense embeddings store facts in shared continuous parameters, leading to interference. SF's binary fingerprints use discrete grid positions — each concept maps to specific cells, not shared continuous values.

### 3. Collapse Thresholds

> "Through semantic density (rho), the mean pairwise cosine similarity, we show collapse occurs at N=5 facts (rho > 0.6) or N ~ 20-75 (moderate rho)."

**Implication for SF**: Dense embeddings fail when semantic density is high. SF avoids this by using binary representations where similarity is measured by set overlap, not cosine distance in continuous space.

### 4. Complementary Learning Systems

> "Complementary Learning Systems theory explains this through two subsystems — a fast hippocampal system using sparse, pattern-separated representations for episodes, and a slow neocortical system using distributed representations for statistical regularities."

**Implication for SF**: SF directly implements the hippocampal subsystem: sparse activation patterns (1-2% of grid cells), pattern separation (distinct concepts → distinct grid regions), rapid encoding (no training).

### 5. Hash-Based vs Neural Retrieval

> "On Wikipedia facts, KO retrieval achieves 45.7% where Modern Hopfield Networks collapse to near-zero; hash-based retrieval maintains 100%."

**Implication for SF**: Morton Z-order encoding is a form of locality-sensitive hashing — maps similar concepts to nearby cells while maintaining distinct representations. Combines hash reliability with semantic flexibility.

---

## Connection to Semantic Folding

| CLS Theory Principle | SF Implementation |
|---------------------|-------------------|
| Sparse pattern-separated representations | Binary fingerprints with 10-25% active bits |
| Fast hippocampal encoding | Unsupervised, no training required |
| Pattern separation | Distinct concepts → distinct grid regions |
| Orthogonal keys | High-dimensional sparsity ensures near-orthogonality |
| No semantic interference | Discrete grid positions, not shared continuous parameters |

---

## Why This Matters for Dataset Selection

The Orthogonality Constraint and Semantic Interference explain why SF excels on datasets with **high semantic density** (many semantically similar concepts):

1. **Biomedical QA**: Dense embeddings would interfere on similar medical terms (myocardial infarction, heart attack, MI). SF's sparse binary encoding separates them on the grid.

2. **Narrative comprehension**: Dense embeddings would interfere on paraphrases ("said" vs "stated" vs "uttered"). SF's grid places them in nearby but distinct regions.

3. **Entity lookup**: Dense embeddings would interfere on entity variants. SF's binary encoding maintains distinct representations.

Conversely, SF struggles on datasets requiring **compositional reasoning** because the Orthogonality Constraint only helps with individual concept storage, not with composing facts across passages.

---

## References

1. Webber, F. D. S. (2015). Semantic Folding Theory. arXiv:1511.08855.
2. Kanerva, P. (1988). Sparse Distributed Memory. MIT Press.
3. Memory Interference in Neural Systems. (2026). arXiv:2601.15313.

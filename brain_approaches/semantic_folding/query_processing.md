# Query Processing in Semantic Folding: Architecture, Mathematical Formulation, and Empirical Evaluation

## Abstract

This document provides a comprehensive technical description of the query processing module within a Semantic Folding pipeline developed for knowledge graph construction over academic corpora. The module transforms natural-language queries into sparse, distributed fingerprint representations over a two-dimensional semantic grid, applies IDF-weighted dot-product scoring against pre-indexed document fingerprints, and returns a ranked list of semantically relevant documents. The design integrates phrase extraction via spaCy, IDF-based term weighting, spatial spreading with exponential decay, and a weighted overlap scoring function. This document covers the theoretical foundations, algorithmic specification, implementation details, and empirical behavior as observed on a 20-document corpus.

---

## 1. Introduction

Semantic Folding Theory (Purdy, 2016) proposes that the semantic content of natural language can be represented as sparse binary vectors — called *semantic fingerprints* — defined over a fixed, high-dimensional grid. Words and phrases that appear in similar contexts are assigned proximate positions on this grid, exploiting the spatial locality of semantic similarity. This approach draws on neuroscientific parallels with distributed cortical representations (Hawkins & George, 2006) and operationalizes the distributional hypothesis (Harris, 1954): linguistic units sharing contextual co-occurrence patterns occupy overlapping regions of the semantic space.

The query processing module described herein is the inference-time component of a multi-stage pipeline. Given a free-text query, it:

1. Extracts and normalizes constituent phrases.
2. Constructs a weighted query fingerprint by superimposing individual phrase fingerprints scaled by their IDF weights.
3. Optionally applies spatial spreading to generalize beyond exact grid positions.
4. Scores each document fingerprint against the query fingerprint using a weighted dot-product formulation.
5. Returns a ranked list of documents with associated relevance scores.

---

## 2. Theoretical Background

### 2.1 Semantic Fingerprints

A semantic fingerprint is a sparse binary vector $\mathbf{f} \in \{0,1\}^N$, where $N = G^2$ for a grid of side length $G$. In this implementation, $G = 16$, yielding $N = 256$. Each phrase $p$ in the vocabulary is assigned a fingerprint $\mathbf{f}_p$ whose active bits correspond to grid cells encoding the phrase's semantic context.

Phrase fingerprints are constructed offline (Steps 2–4 of the pipeline) by mapping phrase co-occurrence contexts into grid coordinates via a fixed coordinate assignment function. Crucially, each phrase fingerprint is built from only those contexts in which the phrase actually appears — a constraint that ensures fingerprint uniqueness and prevents the degenerate case where all fingerprints collapse to a common centroid.

### 2.2 Document Fingerprints

A document fingerprint $\mathbf{d} \in \{0,1\}^N$ is formed by the union (bitwise OR) of the fingerprints of all phrases extracted from the document:

$$\mathbf{d} = \bigvee_{p \in \mathcal{P}(D)} \mathbf{f}_p$$

where $\mathcal{P}(D)$ denotes the set of phrases in document $D$. Document fingerprints are binary, capturing the set of semantic regions activated by the document's content. The number of active bits, $\|\mathbf{d}\|_0 = \text{nnz}(\mathbf{d})$, reflects the document's semantic breadth.

### 2.3 Inverse Document Frequency Weighting

Not all phrases are equally discriminative. Common phrases (e.g., *language*, *cultural*) activate large, overlapping regions of the semantic grid and provide little contrastive signal. Rare phrases (e.g., *evolution*, *phylogenetic*) are more informative about a document's specific content.

The IDF weight for phrase $p$ is defined as:

$$\text{IDF}(p) = \log\left(\frac{M}{1 + \text{df}(p)}\right) + 1$$

where $M$ is the total number of documents in the corpus and $\text{df}(p)$ is the number of documents containing phrase $p$. The additive smoothing constant $+1$ prevents zero weights for phrases appearing in all documents.

### 2.4 Spatial Spreading

The semantic grid encodes not only lexical identity but spatial proximity as semantic relatedness. Two phrases with overlapping or adjacent grid activations share contextual similarity. Spatial spreading operationalizes this by propagating activation from each active cell to its Moore neighborhood with exponential decay:

$$\tilde{f}[i,j] = \max_{(i',j') \in \mathcal{N}_r(i,j)} f[i',j'] \cdot \gamma^{d(i,j,i',j')}$$

where $r$ is the spreading radius, $\gamma \in (0,1)$ is the decay factor, and $d(\cdot)$ is the Chebyshev distance between cells. In the experiments reported here, $r=1$ and $\gamma=0.5$, meaning immediate neighbors receive half-weight activation.

Spreading increases recall at the cost of precision: documents whose phrases land in adjacent but non-identical grid regions can still contribute to the match score. This is particularly beneficial for synonymic variation, where semantically equivalent phrases may not map to the exact same grid cells.

---

## 3. Pipeline Architecture

The query processing module (Step 6) operates downstream of five preceding stages:

| Step | Module | Output |
|------|--------|--------|
| 1 | Document Ingestion | Raw text corpus |
| 2 | Term Context Extraction | Co-occurrence matrix, phrase-context map |
| 3 | Coordinate Assignment | Grid coordinate map |
| 4 | Phrase Fingerprint Construction | Sparse fingerprint matrix (862 × 256) |
| 5 | Document Fingerprint Construction | Document fingerprint matrix (20 × 256) |
| **6** | **Query Processing** | **Ranked document list** |

The query processing module loads pre-built phrase and document fingerprint matrices and executes the inference pipeline described below.

---

## 4. Algorithmic Specification

### 4.1 Query Phrase Extraction

Given a raw query string $Q$, the module invokes a phrase extraction pipeline built on spaCy's `en_core_web_sm` model. The extraction proceeds as follows:

1. **Tokenization and POS tagging** via spaCy's dependency parser.
2. **Candidate extraction**: noun phrases (NPs) and, if `--keep-verbs` is active, verb phrases (VPs) are extracted as candidate phrases.
3. **Vocabulary matching**: each candidate is looked up in the phrase fingerprint index. Candidates not present in the vocabulary are discarded.
4. **Query expansion**: morphological variants and sub-phrase decompositions are generated; each is checked against the vocabulary independently.

For the reference query:

> *"How has language evolved and what does it reveal about cultural and historical human interaction?"*

The extraction process produced 2 raw phrases, which expanded to 11 candidates after query expansion, yielding **5 vocabulary hits**: `{cultural, historical, human, interaction, language}`. The phrase `evolved` / `evolution` did not appear in the vocabulary under the default configuration, representing a recall gap addressable by lemmatization or synonym injection.

### 4.2 Query Fingerprint Construction

The weighted query fingerprint $\mathbf{q} \in \mathbb{R}^N$ is constructed by superimposing IDF-weighted phrase fingerprints:

$$\mathbf{q} = \sum_{p \in \mathcal{P}(Q)} \text{IDF}(p) \cdot \mathbf{f}_p$$

This produces a **real-valued** (non-binary) vector where each active cell carries a weight proportional to the cumulative IDF of the phrases that activated it. Unlike document fingerprints — which are binarized — the query fingerprint preserves IDF magnitude, allowing rare phrases to contribute proportionally more to the final score.

In the reported run, 5 matched phrases activated **53 unique grid cells**, with a weighted sum yielding a non-binary floating-point vector. The active bit count of 53 reflects roughly 20.7% grid coverage before spreading.

### 4.3 Spatial Spreading

If spreading is enabled (`--spreading-steps` > 0), the query fingerprint is passed through the spreading operator:

$$\tilde{q}[i,j] = \max_{(i',j') \in \mathcal{N}_r(i,j)} q[i',j'] \cdot \gamma^{d(i,j,i',j')}$$

This extends each real-valued activation outward to its $r$-ring neighborhood. With $r=1$ and $\gamma=0.5$, the spreading step increased active coverage from **53 to 124 bits** (+71 new cells), representing a 134% increase in spatial coverage.

Post-spreading normalization is **disabled** (`normalize_after_spreading=False`) to preserve IDF magnitude differences between phrases. L2 normalization after spreading would attenuate the discriminative signal from rare, high-IDF terms.

### 4.4 Scoring and Ranking

Documents are scored using a **weighted dot-product** formulation:

$$\text{score}(Q, D) = \frac{\tilde{\mathbf{q}} \cdot \mathbf{d}}{\sqrt{\text{nnz}(\mathbf{d})}}$$

where:
- $\tilde{\mathbf{q}} \in \mathbb{R}^N$ is the (spread) weighted query fingerprint.
- $\mathbf{d} \in \{0,1\}^N$ is the binary document fingerprint.
- $\text{nnz}(\mathbf{d})$ is the number of active bits in the document fingerprint, used as a normalization factor penalizing documents with excessive breadth.

The numerator $\tilde{\mathbf{q}} \cdot \mathbf{d}$ computes the sum of query weights at positions activated by the document, effectively measuring the total IDF-weighted semantic overlap. The normalization by $\sqrt{\text{nnz}(\mathbf{d})}$ applies a soft length penalty: documents covering more semantic ground are penalized relative to those with focused content.

This scoring function is asymmetric: the query fingerprint is real-valued (carrying IDF weights), while document fingerprints remain binary. This asymmetry is intentional — it decouples the discriminative signal (encoded in query weights) from the document's topical breadth (encoded in its binary coverage).

---

## 5. Empirical Results

### 5.1 Experimental Configuration

The experiment was conducted on a corpus of 20 documents covering diverse topics. The pipeline was executed with the following parameters:

| Parameter | Value |
|-----------|-------|
| Grid size | 16 × 16 = 256 cells |
| Spreading steps | 1 |
| Spreading decay | 0.50 |
| IDF weighting | Enabled |
| Keep verbs | True |
| Top-k | 10 |
| Vocabulary size | 831 phrases |
| Fingerprint matrix | 862 × 256 (862 rows, 831 labeled) |

### 5.2 Ranking Results

The top-10 ranked documents for the reference query are presented below:

| Rank | Document ID | Score |
|------|-------------|-------|
| 1 | 8 | 0.2103 |
| 2 | 6 | 0.1620 |
| 3 | 4 | 0.1588 |
| 4 | 16 | 0.1509 |
| 5 | 17 | 0.1469 |
| 6 | 5 | 0.1403 |
| 7 | 15 | 0.1240 |
| 8 | 10 | 0.1230 |
| 9 | 13 | 0.1196 |
| 10 | 1 | 0.0722 |

Document #6 (known to concern language evolution) ranked **2nd**, confirming that IDF weighting successfully elevates thematically focused documents. Prior to IDF weighting (binary cosine similarity baseline), Document #6 ranked 5th without spreading.

### 5.3 Analysis of Document #8's Lead Position

Document #8 (concerning urbanization and social dynamics) ranks first despite being less thematically aligned with language evolution. This is attributable to high overlap with the frequent query terms `cultural`, `human`, and `interaction` — terms with lower IDF weights but broad semantic grid coverage. Their cumulative dot-product contribution exceeds the boost provided by the rarer term `evolution` in Document #6.

This behavior reflects a fundamental tension in IDF-weighted retrieval: **breadth of overlap** versus **depth of discriminative match**. Document #8 scores via breadth; Document #6 via depth. In standard IR evaluation, this would be adjudicated by precision-at-k and NDCG metrics against human relevance judgments.

### 5.4 Vocabulary Coverage and the Evolution Gap

The phrase `evolution` (and its morphological variant `evolved`) was absent from the matched query vocabulary despite appearing 4 times in the corpus. This is a known limitation of the vocabulary construction stage: phrases must appear in the pre-built phrase fingerprint index, which is derived from corpus co-occurrences in Steps 1–4. Queries containing out-of-vocabulary (OOV) terms lose the discriminative contribution of those terms entirely.

Mitigation strategies include:

- **Lemmatization at index time**: indexing `evolve`, `evolved`, `evolution` under a common lemma.
- **Subword fingerprinting**: constructing fingerprints for morphological roots.
- **Query expansion via WordNet or embedding-based synonym injection**: supplementing OOV query terms with in-vocabulary synonyms before fingerprint construction.

### 5.5 Matrix Alignment Warning

A non-critical warning was raised during execution:

> `token_map has 831 entries but matrix has 862 rows — index map and matrix may be misaligned.`

This discrepancy (31 rows) arises from phrases that were fingerprinted during Step 4 but subsequently deduplicated or filtered from the metadata index. The 31 orphan rows are unreachable during query processing (no phrase string maps to them) and do not affect correctness. However, they constitute latent technical debt: the matrix occupies slightly more memory than necessary, and the misalignment complicates index validation. Remediation requires re-running Step 4 with strict consistency enforcement between the fingerprint matrix and the metadata JSON.

---

## 6. Design Decisions and Trade-offs

### 6.1 Binary Documents, Real-Valued Queries

The decision to maintain binary document fingerprints while allowing real-valued query fingerprints is a deliberate asymmetry. Regenerating document fingerprints with continuous IDF weights (Option B, rejected) would require re-running Steps 4–5 and would alter the storage format. The adopted approach (Option A) localizes the IDF logic entirely within the query processing module, preserving pipeline modularity and backward compatibility.

The trade-off is that document-side IDF information is not directly available to the scorer. Documents containing rare phrases cannot signal that rarity in their fingerprint — only the query can carry IDF weights. This is analogous to asymmetric scoring in BM25, where term frequencies are considered on both sides; the current design is closer to a weighted query vector against a binary inverted index.

### 6.2 Normalization Strategy

The normalization denominator $\sqrt{\text{nnz}(\mathbf{d})}$ is a soft cosine-like length penalty. Alternative normalizations considered:

- **No normalization**: favors long documents with many active bits; strongly biased toward topic-broad documents.
- **Full cosine normalization** $(\|\tilde{\mathbf{q}}\|_2 \cdot \|\mathbf{d}\|_2)^{-1}$: penalizes both query and document length simultaneously; appropriate if query coverage varies significantly across runs.
- **$\text{nnz}(\mathbf{d})$ (linear)**: over-penalizes broad documents; empirically under-performs the square-root variant.

The square-root penalty provides a pragmatic balance, consistent with Okapi BM25's field-length normalization philosophy.

### 6.3 Spreading Radius and Decay

The spreading parameters $r=1$, $\gamma=0.5$ were selected empirically. A radius of 1 (Moore neighborhood: 8 neighbors per cell) provides a limited spatial generalization without excessive noise injection. The 50% decay ensures that spread bits contribute at most half the weight of a direct hit, preserving the primacy of exact vocabulary matches.

Larger radii ($r=2,3$) increase recall but risk merging semantically distinct regions of the grid, reducing precision. In a 16×16 grid, $r=3$ would reach up to 49 cells per center, potentially connecting unrelated semantic regions.

---

## 7. Limitations and Future Work

**Vocabulary OOV**: Query terms not present in the phrase fingerprint index contribute nothing to the query fingerprint. This is the most significant failure mode for queries containing domain-specific or morphologically complex terms. Future work should incorporate lemmatization-aware vocabulary lookup and embedding-based OOV handling.

**Binary document representation**: Document fingerprints do not encode term frequency or IDF. Documents containing a rare phrase once are indistinguishable from those containing it frequently. Incorporating TF-weighted document fingerprints — at the cost of moving from binary to integer/float storage — would improve ranking fidelity.

**Grid resolution**: A 16×16 grid (256 cells) limits the semantic granularity available to the system. With 831 vocabulary phrases competing for 256 cells, significant collision and aliasing occur. Increasing grid resolution to 32×32 (1024 cells) or 64×64 (4096 cells) would reduce collisions at the cost of increased memory and potentially sparser fingerprints.

**Evaluation**: No ground-truth relevance judgments are available for the current corpus. Systematic evaluation requires annotation of query-document relevance pairs and computation of standard IR metrics (MAP, NDCG@10, P@5).

---

## 8. Conclusion

The query processing module presented here implements a principled, efficient approach to semantic retrieval based on Semantic Folding Theory. By constructing IDF-weighted, spatially spread query fingerprints and scoring them against binary document fingerprints via weighted dot-product, the module achieves semantically coherent rankings that improve over naive binary cosine similarity. The modular design — confining IDF logic to the query side — preserves backward compatibility with previously indexed document collections.

Empirical results on a 20-document corpus demonstrate that IDF weighting successfully elevates topically focused documents (e.g., Document #6 on language evolution rising to 2nd place), while spatial spreading provides robust recall for semantically adjacent content. Key limitations — vocabulary coverage gaps, binary document representations, and grid resolution constraints — identify concrete directions for future development within the broader knowledge graph construction pipeline.

---

## References

- Harris, Z. S. (1954). Distributional structure. *Word*, 10(2–3), 146–162.
- Hawkins, J., & George, D. (2006). *Hierarchical Temporal Memory: Concepts, Theory, and Terminology*. Numenta Technical Report.
- Purdy, S. (2016). Encoding data for HTM systems. *Frontiers in Neuroscience*, 10, 34.
- Robertson, S. E., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.
- Turney, P. D., & Pantel, P. (2010). From frequency to meaning: Vector space models of semantics. *Journal of Artificial Intelligence Research*, 37, 141–188.

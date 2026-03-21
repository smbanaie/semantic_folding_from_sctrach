# Semantic Fingerprint Generation in a Closed-Domain Question Answering Pipeline

## Abstract

This chapter presents a technically rigorous account of fingerprint generation within a six-stage Semantic Folding pipeline designed for closed-domain question answering. Grounded in the distributional hypothesis (Harris, 1954; Firth, 1957) and the neuroscientific theory of Sparse Distributed Representations (Hawkins & Ahmad, 2016; Ahmad & Hawkins, 2016), the pipeline transforms raw text into binary spatial encodings — termed *semantic fingerprints* — over a two-dimensional cortical grid. We provide formal treatment of two central contributions: (1) Morton Z-order curve linearisation (Morton, 1966) for locality-preserving fingerprint indexing, and (2) TF-IDF weighted aggregation (Salton et al., 1975) for document-level SDR construction. The implementation is grounded in two production scripts — `phrase_fingerprints.py` (Step 4) and `doc_fingerprints.py` (Step 5) — whose design decisions are analysed in detail. The approach is contrasted with dense embedding methods such as Word2Vec (Mikolov et al., 2013), GloVe (Pennington et al., 2014), and BERT (Devlin et al., 2019) to position Semantic Folding within the broader NLP landscape.

---

## 1 Introduction

The fundamental question of how meaning can be computationally represented has occupied computational linguistics for over half a century. The *distributional hypothesis* — the principle that linguistic items appearing in similar contexts carry similar meanings — provides the theoretical foundation for most modern semantic models (Harris, 1954; Firth, 1957). This hypothesis underpins both classical vector space models (Salton et al., 1975; Turney & Pantel, 2010) and contemporary neural embedding approaches (Mikolov et al., 2013; Pennington et al., 2014; Devlin et al., 2019).

Semantic Folding Theory (Lofthouse, 2019) operationalises the distributional hypothesis through a biologically inspired framework drawn directly from the neocortical architecture of the human brain. Rather than producing dense continuous vectors, Semantic Folding encodes meaning as *sparse distributed patterns* over a two-dimensional semantic grid — patterns whose structure mirrors the topographic organisation of the neocortex, where semantically related concepts activate spatially proximate cortical regions.

The central artefact of this encoding is the **semantic fingerprint**: a sparse binary matrix of size $G \times G$ (where $G$ is the grid side length, typically $G = 32$) in which active cells correspond to the positions of semantically relevant contexts on the grid. Fingerprints are defined at two levels of granularity:

- **Phrase fingerprints** (Step 4): one fingerprint per unique phrase token, encoding the set of grid cells at which that phrase's associated contexts appear.
- **Document/context fingerprints** (Step 5): one fingerprint per document or context, constructed by TF-IDF weighted aggregation of its constituent phrase fingerprints, followed by sparsification.

This chapter provides a complete technical and theoretical account of both levels, grounded in the implementation documented in `phrase_fingerprints.py` and `doc_fingerprints.py`.

---

## 2 Related Work

### 2.1 Dense Embedding Approaches

The dominant paradigm in modern NLP represents semantic content as dense continuous vectors in $\mathbb{R}^d$. Word2Vec (Mikolov et al., 2013) learns such representations via shallow neural networks trained on co-occurrence prediction. GloVe (Pennington et al., 2014) derives vectors from global co-occurrence statistics. Contextual models such as BERT (Devlin et al., 2019) produce token-level representations conditioned on full sentential context via transformer self-attention.

While highly effective, these approaches share a common limitation for large-scale retrieval: similarity computation requires $O(d)$ floating-point operations per pair, and nearest-neighbour search over large corpora necessitates approximate indexing structures such as HNSW or FAISS.

### 2.2 Sparse Representations

An alternative tradition, rooted in neuroscience rather than gradient-based optimisation, favours **sparse distributed representations** (Hawkins & Ahmad, 2016; Ahmad & Hawkins, 2016). In an SDR of dimensionality $n$ with $k$ active bits, the probability that two randomly chosen SDRs share any active bit is:

$$P(\text{overlap} \geq 1) = 1 - \left(\frac{\binom{n-k}{k}}{\binom{n}{k}}\right) \approx 1 - e^{-k^2/n}$$

For typical parameters ($n = 1024$, $k \approx 20$), this probability is negligibly small, making non-zero overlap a statistically reliable indicator of semantic similarity (Ahmad & Hawkins, 2016). Retrieval reduces to **Hamming distance** computation, which is executable in a single CPU instruction on modern hardware via `popcount`.

### 2.3 Locality-Sensitive Methods

The locality-sensitive hashing (LSH) framework (Indyk & Motwani, 1998) provides a theoretical basis for hash functions that map similar objects to nearby codes. Morton Z-order encoding (Morton, 1966), as employed in this pipeline, achieves an analogous goal in two dimensions: it maps a 2D grid coordinate $(x, y)$ to a 1D index in a manner that preserves spatial proximity, ensuring that grid-adjacent cells produce nearby linear indices.

### 2.4 Weighting Schemes

The TF-IDF weighting scheme (Salton et al., 1975) and its theoretical grounding in pointwise mutual information (Church & Hanks, 1990; Bullinaria & Levy, 2007) have long provided the standard approach to term salience estimation. Probabilistic retrieval models such as BM25 (Robertson & Zaragoza, 2009) and language model-based approaches (Ponte & Croft, 1998) extend this foundation, and the aggregation mechanism in Step 5 draws conceptually from this tradition.

---

## 3 The Semantic Folding Pipeline

The pipeline implemented in this work comprises six sequential stages, transforming raw text corpora into queryable semantic fingerprint indices. Table 3.1 summarises each stage.

| Step | Module | Input | Output |
|---|---|---|---|
| 0 | Corpus Preparation | Raw text files | Labelled contexts with unique `context_id` |
| 1 | Phrase Extraction | Labelled corpus | Phrase records with `phrase_id`, `token`, `context_id` |
| 2 | Phrase–Context Matrix | Phrase records | Co-occurrence matrix $M \in \mathbb{R}^{P \times C}$ |
| 3 | Semantic Space Mapping | Co-occurrence matrix | Context coordinates $\{(x_c, y_c)\}$ on $G \times G$ grid |
| 4 | Phrase Fingerprints | Coordinates + phrase metadata | $N \times G^2$ fingerprint matrix |
| 5 | Document Fingerprints | Phrase fingerprints + corpus | $D \times G^2$ document SDR matrix |
| 6 | Query Processing | Query text + fingerprint index | Ranked context list |

**Table 3.1:** The six-stage Semantic Folding pipeline.

The phrase–context matrix (Step 2) is a direct instantiation of the classical vector space model (Salton et al., 1975), where each context constitutes a dimension and each phrase a vector in that space. The semantic space mapping (Step 3) reduces this high-dimensional space to a $G \times G$ grid such that semantically similar contexts occupy proximate grid positions — a 2D analogue of the topographic organisation described by Lofthouse (2019).

---

## 4 Theoretical Background: Sparse Distributed Representations on a 2D Grid

### 4.1 The Semantic Grid

Let $\mathcal{G} = \{0, 1, \ldots, G-1\}^2$ denote the $G \times G$ semantic grid. Each context $c$ in the corpus is assigned a unique coordinate $(x_c, y_c) \in \mathcal{G}$ by the semantic space mapping step (Step 3), such that the spatial distance between two context coordinates approximates their semantic dissimilarity:

$$d_{\text{semantic}}(c_i, c_j) \propto \|(x_{c_i}, y_{c_i}) - (x_{c_j}, y_{c_j})\|_2$$

### 4.2 Fingerprint Definition

The **phrase fingerprint** of token $t$ is a function $\mathbf{f}_t : \mathcal{G} \to \mathbb{R}_{\geq 0}$ defined as:

$$\mathbf{f}_t(x, y) = \sum_{c \in \mathcal{C}_t} \mathbf{1}[(x_c, y_c) = (x, y)]$$

where $\mathcal{C}_t = \{c : \text{phrase } t \text{ appears in context } c\}$ is the set of contexts containing phrase $t$, and $\mathbf{1}[\cdot]$ is the indicator function. In implementation, the grid is linearised to a vector $\mathbf{f}_t \in \mathbb{R}^{G^2}$.

### 4.3 Sparsity

For a phrase appearing in $|\mathcal{C}_t|$ distinct grid cells, the sparsity of its fingerprint is:

$$s_t = 1 - \frac{|\mathcal{C}_t|}{G^2}$$

For $G = 32$ and typical phrase frequencies, $s_t \geq 0.95$, yielding SDRs consistent with the sparsity regime identified by Ahmad & Hawkins (2016) as optimal for reliable similarity discrimination.

---

## 5 Phrase Fingerprint Generation (Step 4)

### 5.1 Inputs and Outputs

The `phrase_fingerprints.py` module consumes two JSON artefacts produced by upstream pipeline stages:

- **`context_coordinates.json`** — a mapping $\{\texttt{context\_id} \to \{x, y\}\}$ produced by `semantic_space.py` (Step 3).
- **`phrase_metadata.json`** — a list of phrase records, each containing at minimum `phrase_id`, `token`, and `context_id`.

It produces three output files:

- **`phrase_fingerprints.npz`** — a compressed NumPy archive containing the $N \times G^2$ fingerprint matrix (dtype `float32`), where $N$ is the number of unique phrase tokens.
- **`phrase_fingerprints_meta.json`** — a mapping $\{\texttt{token} \to \text{row index}\}$ enabling efficient lookup.
- **`phrase_fingerprints_stats.json`** — aggregate statistics including per-fingerprint sparsity, mean maximum activation, and skip counts.

### 5.2 Morton Z-Order Linearisation

The central design decision in phrase fingerprint generation is the choice of **linearisation scheme** for mapping 2D grid coordinates to 1D vector indices. Two schemes are supported:

**Row-major linearisation** computes:

$$\text{index}_{\text{row}}(x, y) = y \cdot G + x$$

This is computationally trivial but does not preserve 2D spatial locality: grid cells that are adjacent in 2D may be far apart in the 1D index (e.g., cells at $(G-1, 0)$ and $(0, 1)$ are adjacent in the grid but separated by $G - 1$ positions in the 1D index).

**Morton Z-order linearisation** (Morton, 1966) interleaves the binary representations of $x$ and $y$ coordinates to produce the Morton code:

$$z = \text{Morton}(x, y) = \sum_{i=0}^{B-1} \left( x_i \cdot 2^{2i} + y_i \cdot 2^{2i+1} \right)$$

where $x_i$ and $y_i$ denote the $i$-th bits of $x$ and $y$ respectively, and $B = \lceil \log_2 G \rceil$ is the bit depth. The resulting Z-order curve traverses the grid in a recursive $\mathsf{Z}$-shaped pattern, ensuring that spatially proximate cells map to nearby 1D indices.

This property is formally analogous to the locality-sensitive hashing framework (Indyk & Motwani, 1998): the Morton encoding constitutes a locality-sensitive hash for 2D grid coordinates under the $\ell_\infty$ metric. As a consequence, the Hamming distance between two Morton-encoded fingerprints provides a tighter approximation to the spatial (and hence semantic) distance between their constituent active cells than row-major encoding.

> **Implementation note:** When Morton encoding is enabled (`--use-morton`), the grid size $G$ should be a power of two to ensure complete bit interleaving without boundary effects. The `validate_inputs` function emits a warning if this condition is not satisfied.

### 5.3 Fingerprint Construction

For each phrase record $\{$`phrase_id`, `token` $t$, `context_id` $c\}$, the construction procedure is:

1. Retrieve $(x_c, y_c)$ from `context_coordinates`.
2. Compute the linear index: $\text{idx} = \text{Morton}(x_c, y_c)$ or $y_c \cdot G + x_c$.
3. Set $\mathbf{f}_t[\text{idx}] \mathrel{+}= 1.0$.

Multiple occurrences of phrase $t$ in different contexts mapping to the same grid cell accumulate additively. The full procedure is formalised in Algorithm 3.1.

---

**Algorithm 3.1: Phrase Fingerprint Generation**
```bash
Input:  context_coordinates C, phrase_metadata P, grid_size G,
        use_morton ∈ {True, False}, smooth ∈ {True, False}, σ ∈ ℝ₊
Output: fingerprint matrix F ∈ ℝᴺˣᴳ², token_index_map M

1.  validate_inputs(args)
2.  C ← load_context_coordinates(coordinates_path)
3.  P ← load_phrase_metadata(metadata_path)
4.  validate_grid_bounds(C, G)
5.  Initialise F ← 0 ∈ ℝᴺˣᴳ², M ← {}
6.  for each record r ∈ P do
7.      t ← r.token;  c ← r.context_id
8.      if c ∉ C then skip; end if
9.      (x, y) ← C[c]
10.     if use_morton then idx ← Morton(x, y)
11.     else idx ← y × G + x
12.     if t ∉ M then M[t] ← next_row(); F[M[t]] ← 0 end if
13.     F[M[t], idx] += 1.0
14. end for
15. if smooth then
16.     for each row i do F[i] ← gaussian_filter1d(F[i], σ) end for
17. end if
18. return F, M
```
---

### 5.4 Gaussian Smoothing

An optional post-processing step applies a 1D Gaussian kernel to each fingerprint vector:

$$\tilde{\mathbf{f}}_t[i] = \sum_{j} \mathbf{f}_t[j] \cdot \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(i-j)^2}{2\sigma^2}\right)$$

Smoothing is motivated by the observation that the Morton-indexed fingerprint encodes spatial proximity: neighbouring cells in the 1D index are semantically related, so Gaussian diffusion of activation mass across adjacent cells generalises the fingerprint to nearby semantic regions. The parameter $\sigma$ (controlled via `--sigma`) governs the spatial extent of this generalisation. When $\sigma \to 0$, the smoothed fingerprint converges to the original binary representation; when $\sigma \gg 1$, it approaches a uniform distribution over the grid.

### 5.5 Statistical Characterisation

The `phrase_fingerprints_stats.json` output records the following statistics over all $N$ fingerprints:

- **Mean sparsity** $\bar{s} = \frac{1}{N}\sum_{t} \left(1 - \frac{\|\mathbf{f}_t\|_0}{G^2}\right)$
- **Mean maximum activation** $\frac{1}{N}\sum_{t} \max_i \mathbf{f}_t[i]$
- **Skip count**: number of phrase records whose `context_id` was absent from `context_coordinates`, indicating upstream pipeline inconsistencies.

---

## 6 Document Fingerprint Generation (Step 5)

### 6.1 Motivation and Design

While phrase fingerprints encode the semantic signature of individual tokens, downstream retrieval requires document-level representations that capture the aggregate semantic content of a context. The `doc_fingerprints.py` module (Step 5 of the pipeline) constructs such representations by weighted combination of phrase fingerprints, following the principle that term salience within a document should modulate the contribution of each phrase's fingerprint to the document-level SDR (Salton et al., 1975; Church & Hanks, 1990).

### 6.2 IDF Weight Computation

Let $D$ denote the total number of documents in the corpus and $d_t$ the number of documents containing phrase $t$. The inverse document frequency is computed as:

$$\text{IDF}(t) = \log\!\left(\frac{D}{d_t + 1}\right) + 1$$

This smoothed formulation prevents zero weights for universally occurring phrases and is consistent with standard IDF implementations (Salton et al., 1975; Robertson & Zaragoza, 2009). IDF weights are computed corpus-wide by `compute_idf_weights` before document processing begins.

### 6.3 TF-IDF Weighted Fingerprint Aggregation

For a document $d$ with term frequency $\text{TF}(t, d)$ for phrase $t$, the raw document fingerprint accumulator is:

$$\mathbf{A}_d = \sum_{t \in \mathcal{V}_d} \text{TF}(t, d) \cdot \text{IDF}(t) \cdot \mathbf{f}_t$$

where $\mathcal{V}_d$ is the set of phrases in document $d$ that appear in the phrase fingerprint inventory. The accumulator $\mathbf{A}_d \in \mathbb{R}^{G^2}$ is stored as a sparse `csr_matrix` for memory efficiency.

In implementation, phrase fingerprints are stored as sets of active $(row, col)$ coordinate pairs (loaded by `load_phrase_fingerprints_sparse`), and the flat index is computed using row-major order as a neutral dense carrier. Morton re-ordering is deferred to the sparsification step:

```python
def build_document_fingerprint(doc_text, phrase_fingerprints,
                                idf_weights, grid_size,
                                remove_verbs=True):
    accumulator = lil_matrix((1, grid_size**2), dtype=np.float32)
    tf = {}
    for token in doc_text.split():
        normed = normalize_phrase(token, remove_verbs=remove_verbs)
        if normed is not None:
            tf[normed] = tf.get(normed, 0) + 1
    for phrase, freq in tf.items():
        if phrase not in phrase_fingerprints:
            continue
        idf   = idf_weights.get(phrase, 1.0)
        weight = freq * idf
        for (row, col) in phrase_fingerprints[phrase]:
            flat_idx = row * grid_size + col
            accumulator[0, flat_idx] += weight
        matched += 1
    return accumulator.tocsr() if matched > 0 else None
```

### 6.4 Sparsification via Morton Ordering

The raw accumulator $\mathbf{A}_d$ is a dense weighted sum and does not yet constitute an SDR. Sparsification converts it to a binary SDR by retaining only the top-$k$ active cells, where $k$ is determined by a target sparsity percentage $\rho$:

$$k = \max\!\left(1,\ \left\lfloor \rho \cdot G^2 \right\rceil\right)$$

The `sparsify_to_sdr` function wraps `lib.sparsify_fingerprint`, which applies Morton re-ordering (`use_zorder=True`) before thresholding. This ensures that the final binary SDR respects the locality structure of the Z-order curve, aligning the document fingerprint's spatial semantics with those of the phrase fingerprints from Step 4:

```python
def sparsify_to_sdr(fingerprint, top_percent, grid_size):
    total_bits = grid_size * grid_size
    top_k = max(1, int(round(top_percent * total_bits)))
    return sparsify_fingerprint(fingerprint, top_k=top_k,
                                use_zorder=True, grid_size=grid_size)
```

### 6.5 Optional Normalisation and Diversity Metrics

Following sparsification, an optional $\ell_2$ normalisation step (`normalize_fingerprint`) scales each document fingerprint to unit norm, enabling cosine-equivalent similarity computation via dot product. The pipeline also computes **fingerprint diversity** (`compute_fingerprint_diversity`) — a corpus-level metric measuring the mean pairwise Hamming distance between document SDRs, providing an indicator of the discriminative capacity of the fingerprint space.

---

## 7 Query Processing (Step 6)

Query processing mirrors the document fingerprint construction pipeline in miniature:

1. **Phrase extraction:** The query string is tokenised and normalised using `normalize_phrase`, producing a set of query phrase tokens $\mathcal{Q}$.
2. **Fingerprint construction:** For each $t \in \mathcal{Q}$, the corresponding phrase fingerprint $\mathbf{f}_t$ is retrieved from the phrase fingerprint index (loaded from `phrase_fingerprints.npz`).
3. **Aggregation:** Query phrase fingerprints are combined (with uniform weights, since IDF-based weighting requires corpus-level statistics not generally available at query time) to produce a query fingerprint $\mathbf{f}_q$.
4. **Retrieval:** The query SDR $\mathbf{f}_q$ is compared against all document SDRs $\{\mathbf{f}_d\}$ using **Hamming similarity**:

$$\text{sim}(\mathbf{f}_q, \mathbf{f}_d) = \frac{|\mathbf{f}_q \cap \mathbf{f}_d|}{|\mathbf{f}_q \cup \mathbf{f}_d|}$$

where $|\mathbf{f}_q \cap \mathbf{f}_d|$ denotes the number of bits active in both fingerprints (bitwise AND popcount) and $|\mathbf{f}_q \cup \mathbf{f}_d|$ the number active in either (bitwise OR popcount). This is equivalent to the **Jaccard similarity** over the active bit sets, which is closely related to the MinHash framework for set similarity estimation.

The retrieval step draws conceptually from probabilistic relevance models (Robertson & Zaragoza, 2009; Ponte & Croft, 1998), replacing dense scoring functions with computationally efficient SDR overlap comparison. Evaluation of retrieval quality can be performed against semantic similarity benchmarks such as SimLex-999 (Hill et al., 2015).

---

## 8 Implementation Details

### 8.1 Dependencies

The implementation relies on the following core libraries:

- **NumPy**: fingerprint matrix storage and arithmetic.
- **SciPy** (`csr_matrix`, `lil_matrix`): sparse matrix accumulation during document fingerprint construction.
- **SciPy** (`gaussian_filter1d`): optional Gaussian smoothing in Step 4.
- **Loguru** (`logger`): structured logging with `INFO`, `WARNING`, and `SUCCESS` levels.

### 8.2 CLI Interface

The `phrase_fingerprints.py` module exposes a complete command-line interface:

```bash
python phrase_fingerprints.py \
    --coordinates  data/context_coordinates.json \
    --metadata     data/phrase_metadata.json \
    --output-dir   data/fingerprints/ \
    --grid-size    32 \
    --use-morton \
    --smooth \
    --sigma        1.0
```

The `doc_fingerprints.py` module is invoked as:

```bash
python doc_fingerprints.py \
    --corpus        data/corpus.jsonl \
    --phrases       data/phrases.txt \
    --fingerprints  data/phrase_fingerprints.npz \
    --output-dir    data/doc_fingerprints/
```

### 8.3 Validation and Error Handling

Both modules implement strict input validation before file I/O begins (`validate_inputs`), checking:

- Existence and format (`.json`) of all input paths.
- Grid size positivity and power-of-two alignment with Morton encoding.
- Non-negativity of $\sigma$.
- Coordinate bounds consistency (`validate_grid_bounds`), with `SystemExit(1)` on violation.

Malformed individual records (missing keys, type errors) are skipped with a warning rather than aborting execution, ensuring robustness on noisy corpus data.

---

## 9 Summary

This chapter has presented a complete technical account of the Semantic Folding fingerprint generation pipeline, covering Steps 4 and 5 in detail and Step 6 in outline. The key contributions are:

1. **Morton-indexed SDRs**: The use of Z-order curve linearisation (Morton, 1966) ensures that spatial proximity on the semantic grid is reflected in the linear fingerprint index, grounding the approach in the locality-sensitive hashing framework (Indyk & Motwani, 1998) and providing theoretically justified Hamming distance semantics.

2. **Gaussian smoothing**: Optional Gaussian diffusion of phrase fingerprints generalises single-cell activations to semantically proximate grid regions, producing smoother sparse representations with improved recall at retrieval time.

3. **TF-IDF weighted aggregation**: Document fingerprints are constructed by TF-IDF weighted superposition of phrase SDRs (Salton et al., 1975; Church & Hanks, 1990), providing term-salient document representations that are theoretically grounded in both classical IR (Robertson & Zaragoza, 2009) and distributional semantics (Turney & Pantel, 2010).

4. **Efficient sparse retrieval**: By encoding all semantic content as binary SDRs, the pipeline reduces document retrieval to Hamming distance computation — a single CPU instruction — contrasting sharply with the $O(d)$ floating-point cost of dense embedding models (Mikolov et al., 2013; Pennington et al., 2014; Devlin et al., 2019).

The pipeline is grounded in Semantic Folding Theory (Lofthouse, 2019), which draws its biological inspiration from the established neuroscience of SDRs (Hawkins & Ahmad, 2016; Ahmad & Hawkins, 2016; Räsänen & Saarinen, 2016) and the distributional semantics tradition (Harris, 1954; Firth, 1957).

---

## References

- Ahmad, S., & Hawkins, J. (2016). *How do neurons operate on sparse distributed representations? A mathematical theory of sparsity, neurons and active dendrites.* arXiv:1601.00720.
- Bullinaria, J. A., & Levy, J. P. (2007). Extracting semantic representations from word co-occurrence statistics: A computational study. *Behavior Research Methods, 39*(3), 510–526.
- Church, K. W., & Hanks, P. (1990). Word association norms, mutual information, and lexicography. *Computational Linguistics, 16*(1), 22–29.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*, 4171–4186.
- Firth, J. R. (1957). A synopsis of linguistic theory, 1930–1955. In *Studies in Linguistic Analysis*, 1–32. Blackwell.
- Harris, Z. S. (1954). Distributional structure. *Word, 10*(2–3), 146–162.
- Hawkins, J., & Ahmad, S. (2016). Why neurons have thousands of synapses, a theory of sequence memory in neocortex. *Frontiers in Neural Circuits, 10*, 23.
- Hill, F., Reichart, R., & Korhonen, A. (2015). SimLex-999: Evaluating semantic models with genuine similarity estimation. *Computational Linguistics, 41*(4), 665–695.
- Indyk, P., & Motwani, R. (1998). Approximate nearest neighbors: Towards removing the curse of dimensionality. *Proceedings of STOC 1998*, 604–613.
- Lofthouse, T. (2019). *Semantic Folding Theory and its Application in Text Analysis.* Cortical.io White Paper.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *Proceedings of ICLR 2013*.
- Morton, G. M. (1966). *A computer oriented geodetic data base and a new technique in file sequencing.* IBM Technical Report.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *Proceedings of EMNLP 2014*, 1532–1543.
- Ponte, J. M., & Croft, W. B. (1998). A language modeling approach to information retrieval. *Proceedings of SIGIR 1998*, 275–281.
- Räsänen, O., & Saarinen, J. (2016). Sequence prediction with sparse distributed hyperdimensional coding applied to the analysis of mobile phone use patterns. *IEEE Transactions on Neural Networks and Learning Systems, 27*(9), 1878–1889.
- Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333–389.
- Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing. *Communications of the ACM, 18*(11), 613–620.
- Turney, P. D., & Pantel, P. (2010). From frequency to meaning: Vector space models of semantics. *Journal of Artificial Intelligence Research, 37*, 141–188.
- Wu, Z., & Palmer, M. (1994). Verb semantics and lexical selection. *Proceedings of ACL 1994*, 133–138.
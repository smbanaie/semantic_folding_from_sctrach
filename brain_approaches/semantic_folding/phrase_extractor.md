# Phrase Extraction Module — Technical Documentation

**Module:** `phrase_extractor.py`
**Stage:** 1 of 6 — Semantic Folding Pipeline
**Author:** [Your Name]
**Version:** 2.0 (Revised)

---

## Overview

The phrase extraction module (`phrase_extractor.py`) constitutes the **first stage** of the Semantic Folding pipeline. Its primary responsibility is to identify and extract linguistically meaningful multi-word expressions and noun phrases from a raw text corpus, producing a frequency-ranked phrase inventory that serves as the foundation for all subsequent semantic processing stages.

The module is designed around a key principle: **normalization must occur before counting**. This guarantees that phrase forms encountered during extraction are identical to those used during fingerprinting, ensuring compositional consistency across the entire pipeline.

---

## Theoretical Motivation

### Why Phrase-Level Representation?

Word-level tokenization, while computationally simple, fails to capture the compositional semantics inherent in natural language. The phrase *"machine learning"* carries a meaning that cannot be recovered by independently processing *"machine"* and *"learning"* as separate tokens. This phenomenon, known as **non-compositionality**, is pervasive in technical and scientific discourse, where domain-specific multi-word expressions (MWEs) constitute the primary carriers of conceptual meaning.

Semantic Folding theory, as formalized by Numenta's Hierarchical Temporal Memory (HTM) framework and extended by Kanerva's Sparse Distributed Representations (SDRs), operates on the assumption that semantic units must correspond to coherent conceptual entities. Phrases, rather than isolated words, more faithfully represent such entities in domain-specific corpora.

### Why Frequency-Based Filtering?

The statistical significance of a phrase is directly correlated with its recurrence in the corpus. Hapax legomena (phrases appearing only once) are statistically unreliable as semantic anchors, as they may represent transcription errors, proper nouns, or domain-irrelevant expressions. A minimum frequency threshold $f_{\min}$ ensures that only phrases with sufficient distributional evidence are retained:

$$P_{\text{valid}} = \{ p \in P \mid \text{freq}(p) \geq f_{\min} \}$$

where $P$ is the full set of extracted phrases and $\text{freq}(p)$ denotes the corpus frequency of phrase $p$.

---

## Extraction Methodology

### 1. Dual-Mode Extraction Architecture

The system implements a **primary-fallback architecture** to ensure robustness across different computational environments.

#### 1.1 Primary Method: spaCy-Based Extraction

When the spaCy library (specifically the `en_core_web_sm` model) is available, the system employs a sophisticated linguistic parsing approach.

**Extraction Targets:**

1. **Noun Chunks** — Maximal noun phrases identified by spaCy's dependency parser.
   - Example: `"the advanced machine learning algorithm"` → extracted as a complete chunk.
   - Linguistic basis: Head noun with all pre-modifiers and determiners.

2. **Named Entities** — Proper nouns and named entity spans.
   - Example: `"Stanford University"`, `"Python Programming Language"`
   - Entity types: `PERSON`, `ORG`, `GPE`, `PRODUCT`, etc.

3. **Compound Nouns** — Sequences of consecutive `NOUN` tokens.
   - Example: `"data science research methodology"`
   - Pattern: $\text{NOUN}^{n}$ where $n \geq 2$

**Algorithm:**

```bash
For each sentence s in document D:
    1. Extract noun_chunks(s) → C
    2. Extract named_entities(s) → E
    3. Extract compound_nouns(s) → N
    4. P_raw = C ∪ E ∪ N
    5. Filter: |p| > 1 character for all p ∈ P_raw
```
#### 1.2 Fallback Method: N-gram Pattern Matching

When spaCy is unavailable, the system employs NLTK-based tokenization and part-of-speech tagging with n-gram extraction.

**Algorithm:**


1. Tokenize text → tokens T
2. POS-tag tokens → tagged T'
3. Filter: keep only alphabetic tokens with |token| > 1
4. For n = 1 to max_ngram (default: 4):
       Extract all n-grams from T'
5. Return all n-grams (validation deferred to normalization stage)

**Rationale for Permissive Extraction:**

The fallback method intentionally over-generates candidate phrases, delegating structural validation to the normalization pipeline within `extract_and_normalize_phrases`. This design ensures that linguistically valid phrases are not prematurely discarded due to incomplete POS pattern coverage. Structural validation thus occurs immediately after normalization, not at n-gram generation time.

---

### 2. Normalization and Validation Pipeline

Raw extracted phrases undergo a multi-stage normalization process implemented in `lib.py`. The normalization functions are shared across all pipeline stages, ensuring that phrase forms are consistent from extraction through fingerprinting.

#### 2.1 Linguistic Normalization

The `normalize_phrase(phrase, remove_verbs=True)` function applies the following transformations in sequence:

1. **Lowercasing** — Ensures case-insensitive matching.
2. **Lemmatization** — Reduces inflected forms to base forms.
   - `"algorithms"` → `"algorithm"`
   - `"running"` → `"run"` (when verb retention is disabled)
3. **Stop Word Removal** — Eliminates high-frequency function words.
4. **Verb Filtering** — Removes verbal elements when `remove_verbs=True`, yielding a noun-phrase-centric representation.

**Mathematical Formulation:**

$$\text{normalize}(p) = \text{lemmatize}\bigl(\text{lower}\bigl(\text{filter}_{\text{stop}}\bigl(\text{filter}_{\text{verb}}(p)\bigr)\bigr)\bigr)$$

#### 2.2 Verb Handling Rationale

The `remove_verbs` parameter (default: `True`) controls whether verbal elements are stripped during normalization. When enabled, a phrase such as `"evolved language"` becomes `"language"`, and `"running algorithm"` becomes `"algorithm"`.

**Linguistic Justification:**

Semantic Folding operates on the assumption that semantic units correspond to **conceptual entities**, which in natural language are predominantly expressed as noun phrases. Verbal modifiers introduce temporal or aspectual information that is often irrelevant for entity-centric semantic similarity.

**When to Disable (`keep_verbs=True`):**

- When processing corpora where verbal phrases carry domain-specific meaning (e.g., event logs, procedural text).
- When analyzing event-centric text (news, narratives) where verb semantics are primary.
- When preserving gerunds that function as nouns (e.g., `"deep learning"`, `"natural language processing"`).

**CLI Mapping:**

The `--keep-verbs` CLI flag sets `keep_verbs=True`, which is passed to `extract_and_normalize_phrases` as `remove_verbs=not keep_verbs`, ensuring consistent behavior across the full pipeline.

#### 2.3 Structural Validation

The `is_valid_phrase_structure(tagged)` function receives a POS-tagged token list and enforces grammatical constraints.

**Validation Rules:**

1. **Minimum Length** — $|\text{phrase}| \geq 2$ characters.
2. **Alphabetic Constraint** — All tokens must be alphabetic.
3. **POS Pattern Matching** — Phrase must match one of the valid noun phrase patterns:
   - Single noun: `[NOUN]`
   - Adjective + Noun: `[ADJ, NOUN]`
   - Noun + Noun: `[NOUN, NOUN]`
   - Proper noun sequences: `[PROPN, PROPN, ...]`
   - Complex patterns: `[ADJ, NOUN, NOUN]`, `[NOUN, NOUN, NOUN]`, etc.

**Formal Definition:**

$$\text{valid}(p) = \begin{cases} \text{True} & \text{if } \text{POS}(p) \in \mathcal{P}_{\text{valid}} \\ \text{False} & \text{otherwise} \end{cases}$$

where $\mathcal{P}_{\text{valid}}$ is the set of acceptable POS tag sequences.

> **Note on Dual Invocation:** `is_valid_phrase_structure` is called at **two stages** of the pipeline:
>
> 1. During initial extraction, inside `extract_and_normalize_phrases`, to filter raw phrases before they enter the frequency counter.
> 2. After phrase expansion, inside `process_corpus_with_expansion`, to re-validate all newly generated sub-phrases.
>
> This dual validation ensures that expansion does not introduce grammatically ill-formed fragments into the final phrase inventory.

---

### 3. Context-Based Frequency Computation

Unlike naive token counting, the system employs **context-based frequency measurement** to compute phrase importance.

**Definition:**

$$\text{freq}_{\text{context}}(p) = |\{ c \in C \mid p \in c \}|$$

where $C$ is the set of all contexts (documents/sentences) and $p \in c$ denotes that phrase $p$ appears in context $c$.

**Advantages over Raw Counting:**

1. **Robustness to Repetition** — A phrase repeated 100 times in a single document receives the same weight as appearing once in 100 different documents.
2. **Distributional Significance** — Measures breadth of usage rather than raw occurrence count.
3. **Corpus Balance** — Prevents single-document dominance in phrase ranking.

**Implementation:**

python
phrase_contexts: Dict[str, Set[str]] = defaultdict(set)

for context_id, context_text in corpus:
    phrases = extract_and_normalize_phrases(context_text)
    for phrase in phrases:
        phrase_contexts[phrase].add(context_id)  # Set ensures no double-counting

phrase_counts = Counter({phrase: len(contexts)
                         for phrase, contexts in phrase_contexts.items()})

> **Deduplication Note:** The `phrase_contexts` dictionary uses `Set[str]` to store context IDs, ensuring that a phrase appearing multiple times within the same context is counted only once. This prevents single-document repetition from inflating frequency scores.

---

### 4. Phrase Expansion Strategy

After initial extraction and context-based frequency computation, the system performs **hierarchical phrase expansion** to capture sub-phrase relationships. Crucially, expansion is applied **before** the minimum frequency filter, so that sub-phrases of low-frequency parent phrases are not prematurely discarded.

#### 4.1 Expansion Algorithm

Given a phrase $p = w_1\, w_2\, \ldots\, w_n$ where $n > 1$:

1. Generate all contiguous sub-phrases via `expand_phrases` (from `lib.py`):

$$\text{expand}(p) = \{ w_i\, w_{i+1}\, \ldots\, w_j \mid 1 \leq i \leq j \leq n \}$$

2. Filter generic single words:
   - Remove if $|w| < \text{min\_word\_length}$ (default: 3).
   - Remove if $w \in \text{StopWords}$.

3. Validate each sub-phrase using `is_valid_phrase_structure`.

**Example:**


Input phrase: "machine learning algorithm"
Expansion candidates:
  - "machine learning algorithm" (original — retained)
  - "machine learning"           (valid NP — retained)
  - "learning algorithm"         (valid NP — retained)
  - "machine"                    (may be filtered if generic)
  - "learning"                   (may be filtered if generic)
  - "algorithm"                  (retained if passes length and POS check)

#### 4.2 Contiguous Subsequence Check

A critical correctness requirement is that sub-phrase containment is tested as a **contiguous word subsequence**, not a simple string membership check. The `is_subphrase` function implements this:

```python
def is_subphrase(sub_words: list, full_words: list) -> bool:
    """
    Check if sub_words is a contiguous subsequence of full_words.

    Fixes the original bug where `expanded_phrase in original_phrase.split()`
    only matched single tokens, missing multi-word sub-phrases like
    'cultural group' inside 'different cultural group'.
    """
    n, m = len(full_words), len(sub_words)
    if m >= n:
        return False
    return any(full_words[i:i + m] == sub_words for i in range(n - m + 1))
```

#### 4.3 Frequency Inheritance — Sum-Based Aggregation

Sub-phrases inherit frequencies from their parent phrases using a **sum-based aggregation rule**. Every parent phrase that contains a given sub-phrase as a contiguous subsequence contributes its own context frequency to that sub-phrase's total:

$$\text{freq}(p_{\text{sub}}) = \sum_{\substack{p \in P \\ p_{\text{sub}} \sqsubseteq p}} \text{freq}(p)$$

where $p_{\text{sub}} \sqsubseteq p$ denotes that $p_{\text{sub}}$ is a contiguous sub-sequence of $p$.

**Rationale for Sum over Max:**

Sum-based aggregation reflects the cumulative distributional evidence for a sub-phrase across all parent contexts. If `"machine learning"` appears in 50 contexts and `"learning algorithm"` appears in 30 contexts, the sub-phrase `"learning"` accumulates evidence from both parents, yielding a frequency of 80. A max-based rule would cap it at 50, discarding the additional 30 contexts and underestimating the term's distributional reach.

**Implementation:**

```python
expanded_counts: Dict[str, int] = defaultdict(int)

for original_phrase, original_freq in phrase_counts.items():
    original_words = original_phrase.split()

    # The parent phrase itself contributes its own frequency
    expanded_counts[original_phrase] += original_freq

    # Every expanded sub-phrase that is a true contiguous sub-sequence
    for sub in expanded_phrases:
        sub_words = sub.split()
        if sub_words == original_words:
            continue  # already handled above
        if is_subphrase(sub_words, original_words):
            # Sum: accumulate across all parents that contain this sub-phrase
            expanded_counts[sub] += original_freq
```
#### 4.4 Post-Expansion POS Re-Validation

After frequency aggregation, every phrase in `expanded_counts` — including newly introduced sub-phrases — is re-validated with `is_valid_phrase_structure`. This step produces a new `Counter` object (`validated`), discarding phrases that do not satisfy grammatical constraints rather than merely flagging them:

```python
# ── POS-validate all expanded sub-phrases ────────────────────────────
validated: Counter = Counter()
for phrase, freq in expanded_counts.items():
    tokens = word_tokenize(phrase)
    tagged = pos_tag(tokens)
    if is_valid_phrase_structure(tagged):
        validated[phrase] = freq
```
Invalid phrases are permanently discarded; the result is a clean, grammatically well-formed phrase inventory ready for frequency filtering.

---

### 5. Filtering and Quality Control

#### 5.1 Frequency Threshold

The minimum frequency filter is applied **after** expansion and POS re-validation, operating on the `validated` Counter:

$$P_{\text{final}} = \{ p \in P_{\text{validated}} \mid \text{freq}(p) \geq f_{\min} \}$$

**Implementation:**

```python
# ── Apply minimum frequency filter ───────────────────────────────────────
result = Counter({p: f for p, f in validated.items() if f >= min_freq})
```
Default: $f_{\min} = 2$ (phrase must appear in at least 2 distinct contexts).

Applying the filter at this late stage is intentional: it preserves sub-phrases that would be prematurely discarded if filtering were applied before expansion, including sub-phrases whose parent phrase was itself below the threshold.

#### 5.2 Generic Word Filtering

Single-word phrases are evaluated for semantic specificity via `is_generic_word` from `lib.py`.

**Generic Word Criteria:**

1. Length $< \text{min\_word\_length}$ characters (default: 3).
2. Membership in the stop word list.

**Mathematical Definition:**

$$\text{generic}(w) = \begin{cases} \text{True} & \text{if } |w| < \text{min\_word\_length} \lor w \in \text{StopWords} \\ \text{False} & \text{otherwise} \end{cases}$$

---

## Pipeline Architecture

### Complete Processing Flow

```bash
Input: Raw Corpus (context_id, context_text pairs)
    │
    ▼
[1] Text Preprocessing
    - Sentence segmentation
    - Tokenization
    │
    ▼
[2] Raw Phrase Extraction
    - Primary: spaCy (noun chunks, named entities, compound nouns)
    - Fallback: NLTK n-grams (max_ngram=4, over-generates deliberately)
    │
    ▼
[3] Normalization (lib.py: normalize_phrase)
    - Lowercasing
    - Lemmatization
    - Stop word removal
    - Verb filtering (remove_verbs = not keep_verbs)
    │
    ▼
[4] Structural Validation — Pass 1 (lib.py: is_valid_phrase_structure)
    - POS pattern matching
    - Length and alphabetic constraints
    │
    ▼
[5] Context-Based Frequency Computation
    - phrase_contexts: Dict[str, Set[str]] (deduplication within context)
    - freq(p) = number of distinct contexts containing p
    │
    ▼
[6] Phrase Expansion (before frequency filter)
    - Generate sub-phrases via expand_phrases (lib.py)
    - Contiguous subsequence containment check (is_subphrase)
    - Sum-based frequency aggregation across all parent phrases
    - Structural Validation — Pass 2 (POS re-validation of all sub-phrases)
    │
    ▼
[7] Final Frequency Filter
    - Discard phrases with freq < min_freq (default: 2)
    │
    ▼
Output: Ranked Phrase List (phrase:frequency, descending)
```
---

## Computational Complexity

### Time Complexity

Let:
- $N$ = number of contexts in corpus
- $L$ = average context length (tokens)
- $P$ = number of unique phrases extracted
- $M$ = average phrase length (words)
- $E$ = number of expanded sub-phrases

**Per-Stage Complexity:**

| Stage | Complexity | Notes |
|-------|-----------|-------|
| Extraction (spaCy) | $O(N \cdot L)$ | Dependency parsing dominates |
| Normalization | $O(P \cdot M)$ | Per-token lemmatization |
| Frequency Computation | $O(N \cdot P)$ | Set insertion per context |
| Expansion | $O(P \cdot M^2)$ | Sub-phrase generation per parent |
| Frequency Inheritance | $O(P \cdot M^2)$ in practice; $O(P \cdot E \cdot M)$ worst-case | Sparse containment structure reduces effective cost |
| POS Re-validation | $O(E \cdot M)$ | NLTK tagging per candidate |
| Sorting | $O(P \log P)$ | Final ranked output |

**Total Complexity:** $O(N \cdot L + P \cdot E \cdot M + P \log P)$

For typical corpora where $N \gg P$ and $L \gg M$, spaCy parsing dominates: $O(N \cdot L)$.

> **Note on Frequency Inheritance:** The worst-case complexity of $O(P \cdot E \cdot M)$ assumes dense containment relationships. In practice, most sub-phrases are contained by only a small fraction of parent phrases, making the effective complexity closer to $O(P \cdot M^2)$.

### Space Complexity

| Component | Space |
|-----------|-------|
| Phrase Storage | $O(P \cdot M)$ |
| Context Tracking | $O(P \cdot N)$ worst-case |
| Expansion Buffer | $O(E \cdot M)$ |
| **Total** | $O(P \cdot N + E \cdot M)$ |

---

## Statistical Properties

### Phrase Length Distribution

Empirical analysis of technical corpora shows:

$$P(|p| = k) \approx \frac{\lambda^k e^{-\lambda}}{k!}$$

where $\lambda \approx 2.3$ for scientific text (approximate Poisson distribution).

**Typical Empirical Distribution:**

| Phrase Length | Proportion |
|--------------|-----------|
| 1 word | 40–50% |
| 2 words | 30–40% |
| 3 words | 15–20% |
| 4+ words | 5–10% |

### Frequency Distribution

Phrase frequencies follow a **power-law distribution** (Zipf's law):

$$\text{freq}(p_r) \propto r^{-\alpha}$$

where $r$ is the rank of phrase $p_r$ and $\alpha \approx 1.0$ for technical corpora.

**Implications:**

- The top 10% of phrases account for approximately 50% of total occurrences.
- A long tail of low-frequency phrases requires the minimum frequency filter to control vocabulary size.
- Sub-phrase frequency accumulation via sum-based aggregation partially flattens this distribution for shorter phrases, producing a richer vocabulary of common sub-expressions.

---

## Configuration Parameters

### Parameter Reference

| Parameter | CLI Flag | Default | Description | Impact |
|-----------|----------|---------|-------------|--------|
| `min_freq` | `--min-freq` | 2 | Minimum context frequency | Higher → fewer, more reliable phrases |
| `min_word_length` | `--min-word-length` | 3 | Minimum single-word character length | Higher → fewer generic words |
| `max_ngram` | — | 4 | Maximum n-gram size (fallback only) | Higher → more complex candidate phrases |
| `keep_verbs` | `--keep-verbs` | `True` | Preserve verbal elements during normalization | `False` → strict noun-centric representation |
| `filter_generic` | `--no-filter-generic` | `True` | Remove generic single words | `False` → retain common vocabulary |
| `no_spacy` | `--no-spacy` | `False` | Force fallback extraction | `True` → use NLTK n-gram method |

> **CLI Flag Note:** The `--keep-verbs` flag sets `keep_verbs=True` (preserve verbs). The implementation uses `remove_verbs=not keep_verbs` internally. This inversion is intentional for CLI readability but should be kept in mind when reading function signatures.

### Tuning Guidelines

**For Technical / Scientific Corpora:**
- `min_freq = 2–3`: Balances coverage and reliability.
- `filter_generic = True`: Essential for domain specificity.
- `keep_verbs = True` (default): Gerunds such as `"deep learning"` are preserved.

**For General / Noisy Text:**
- `min_freq = 5–10`: Higher threshold compensates for noisy extraction.
- `filter_generic = False`: Retains common vocabulary that may carry semantic weight.
- `keep_verbs = False`: Restricts to pure noun phrases for cleaner representation.

---

## Output Format

### File Structure


phrase_1:frequency_1
phrase_2:frequency_2
...
phrase_n:frequency_n

**Properties:**

- Sorted by descending frequency.
- UTF-8 encoding.
- One phrase per line.
- Colon-delimited format (`phrase:count`).

**Example:**


machine learning:156
neural network:142
deep learning:98
artificial intelligence:87
data science:76
language evolution:12
cultural interaction:9
historical analysis:7

---

## Integration with Semantic Folding Pipeline

The extracted phrase list serves as input to subsequent pipeline stages:

1. **Stage 2 — Term Context Matrix:** Phrase-context co-occurrence statistics and IDF weight computation.
2. **Stage 3 — Grid Coordinate Assignment:** Phrase tokens are mapped to spatial positions in the semantic grid.
3. **Stage 4 — Phrase Fingerprinting:** Each phrase is encoded as a sparse distributed representation (SDR) based on its constituent term coordinates.
4. **Stage 5 — Document Fingerprinting:** Document contexts are represented as unions of phrase SDRs.
5. **Stage 6 — Query Processing:** Query phrases are matched against the vocabulary, weighted by IDF, and compared against document fingerprints.

**Critical Pipeline Requirement:**

Phrase extraction must produce **consistent, normalized forms** to ensure that:
- Identical concepts receive identical fingerprints across all stages.
- Frequency statistics accurately reflect semantic importance for IDF computation.
- Context encodings are compositionally valid.

This is achieved through:

1. **Normalization-first design** — Linguistic transformations are applied before frequency computation, guaranteeing that phrase forms in extraction match those used in fingerprinting.
2. **Expansion before filtering** — Sub-phrases of low-frequency parents are preserved rather than silently dropped.
3. **Contiguous subsequence containment** — Multi-word sub-phrases are correctly identified, fixing a prior single-token matching bug.
4. **Sum-based frequency aggregation** — Sub-phrase frequencies accumulate evidence across all containing parents, producing more informative distributional statistics.

---

## Validation and Quality Metrics

### Intrinsic Metrics

**Phrase Validity Rate** — Percentage of extracted phrases passing structural validation:

$$\text{Validity} = \frac{|P_{\text{valid}}|}{|P_{\text{raw}}|} \times 100\%$$

**Coverage** — Percentage of corpus tokens contained in extracted phrases:

$$\text{Coverage} = \frac{\displaystyle\sum_{p \in P} |p| \cdot \text{freq}(p)}{\displaystyle\sum_{t \in T} \text{freq}(t)} \times 100\%$$

**Specificity** — Mean inverse document frequency of extracted phrases:

$$\text{Specificity} = \frac{1}{|P|} \sum_{p \in P} \log \frac{N}{\text{freq}(p)}$$

### Extrinsic Validation

- **Semantic Coherence:** Manual inspection of top-$k$ phrases for linguistic well-formedness.
- **Domain Relevance:** Expert evaluation of phrase appropriateness for the target domain.
- **Downstream Performance:** Impact on semantic similarity ranking in Stage 6 query processing.

---

## Known Issues

### Token Map Misalignment Warning

During execution of Stage 4 (phrase fingerprinting) or Stage 6 (query processing), the following warning may appear:


WARNING | token_map has 831 entries but matrix has 862 rows —
          index map and matrix may be misaligned.

**Cause:** This occurs when phrases are fingerprinted and stored in the sparse matrix, but are subsequently deduplicated or filtered out of the metadata JSON (`phrase_fingerprints_meta.json`). The result is 31 orphaned rows in the matrix with no corresponding phrase label.

**Impact:** No functional impact on query processing. Orphaned rows are never matched during vocabulary lookup and do not affect similarity scores. However, they represent wasted memory and indicate a latent inconsistency between the matrix and its index.

**Recommended Remediation:** Re-run the fingerprinting stage (Stage 4) with consistent filtering parameters, or implement a post-processing step to prune unused matrix rows by aligning the sparse matrix to the metadata index after construction.

---

## Limitations and Future Work

### Current Limitations

1. **Language Dependency** — Optimized for English. Adaptation for other languages requires replacement of spaCy's `en_core_web_sm` model and revision of POS pattern sets.
2. **Domain Specificity of POS Patterns** — Validation patterns are tuned for technical and scientific discourse; may under-generate for casual or narrative text.
3. **Computational Cost** — spaCy dependency parsing can be slow for very large corpora. For corpora exceeding $10^6$ tokens, batch processing or GPU acceleration may be required.
4. **Multiword Expression Coverage** — Idiomatic expressions not matching standard noun phrase patterns (e.g., `"kick the bucket"`, `"by and large"`) are not captured.

### Potential Improvements

1. **Statistical Collocation Detection** — Integrate pointwise mutual information (PMI) or log-likelihood ratio tests for data-driven MWE identification, complementing the rule-based POS approach.
2. **Contextual Embedding Augmentation** — Use BERT or similar models to cluster semantically related phrase variants, reducing vocabulary sparsity.
3. **Active Learning Refinement** — Iterative refinement with domain-expert feedback on phrase quality scores.
4. **Cross-Lingual Extension** — Multilingual phrase extraction using language-agnostic models (e.g., spaCy `xx_ent_wiki_sm`).
5. **Matrix-Metadata Alignment** — Automated post-processing to prune orphaned matrix rows and synchronize the phrase index, eliminating the token map misalignment warning.

---

## Conclusion

The phrase extraction module implements a linguistically-informed, statistically-grounded approach to identifying semantic units in text corpora. By combining sophisticated NLP parsing (spaCy) with a robust n-gram fallback (NLTK), context-based frequency measurement, and hierarchical phrase expansion, the system produces a high-quality phrase inventory suitable for semantic fingerprinting and downstream NLP retrieval tasks.

The four key design decisions that distinguish this implementation are:

1. **Normalization-first** — Linguistic transformations are applied before frequency computation, guaranteeing that phrase forms in extraction are identical to those used in fingerprinting across all downstream pipeline stages.
2. **Expansion before filtering** — Sub-phrases of low-frequency parents are preserved rather than silently dropped, ensuring comprehensive sub-expression coverage.
3. **Contiguous subsequence containment** — Multi-word sub-phrases are correctly identified via sliding-window comparison, fixing a prior single-token matching bug that missed phrases such as `"cultural group"` inside `"different cultural group"`.
4. **Sum-based frequency aggregation** — Sub-phrase frequencies accumulate distributional evidence across all containing parent phrases, producing more informative statistics than a max-based rule and better reflecting actual corpus-wide usage breadth.

Together, these properties ensure that the phrase inventory entering the Semantic Folding pipeline is both linguistically valid and statistically well-grounded, providing a strong foundation for the sparse distributed representations constructed in subsequent stages.

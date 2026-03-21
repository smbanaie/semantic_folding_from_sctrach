# Technical Documentation

## Overview

The phrase extraction module (`phrase_extractor.py`) constitutes the first stage of the Semantic Folding pipeline. Its primary responsibility is to identify and extract linguistically meaningful multi-word expressions and noun phrases from a raw text corpus, producing a frequency-ranked phrase inventory that serves as the foundation for all subsequent semantic processing stages.

---

## Theoretical Motivation

### Why Phrase-Level Representation?

Word-level tokenization, while computationally simple, fails to capture the compositional semantics inherent in natural language. The phrase *"machine learning"* carries a meaning that cannot be recovered by independently processing *"machine"* and *"learning"* as separate tokens. This phenomenon, known as **non-compositionality**, is pervasive in technical and scientific discourse, where domain-specific multi-word expressions (MWEs) constitute the primary carriers of conceptual meaning.

Semantic Folding theory, as formalized by Numenta's Hierarchical Temporal Memory (HTM) framework and extended by Kanerva's Sparse Distributed Representations (SDRs), operates on the assumption that semantic units must correspond to coherent conceptual entities. Phrases, rather than isolated words, more faithfully represent such entities in domain-specific corpora.

### Why Frequency-Based Filtering?

The statistical significance of a phrase is directly correlated with its recurrence in the corpus. Hapax legomena (phrases appearing only once) are statistically unreliable as semantic anchors, as they may represent transcription errors, proper nouns, or domain-irrelevant expressions. A minimum frequency threshold $f_{min}$ ensures that only phrases with sufficient distributional evidence are retained:

$$P_{valid} = \{ p \in P \mid \text{freq}(p) \geq f_{min} \}$$

where $P$ is the full set of extracted phrases and $\text{freq}(p)$ denotes the corpus frequency of phrase $p$.

---

## Extraction Methodology

### 1. Dual-Mode Extraction Architecture

The system implements a **primary-fallback architecture** to ensure robustness across different computational environments:

#### 1.1 Primary Method: spaCy-Based Extraction

When the spaCy library (specifically the `en_core_web_sm` model) is available, the system employs a sophisticated linguistic parsing approach:

**Extraction Targets:**

1. **Noun Chunks**: Maximal noun phrases identified by spaCy's dependency parser
   - Example: "the advanced machine learning algorithm" → extracted as complete chunk
   - Linguistic basis: Head noun with all pre-modifiers and determiners

2. **Named Entities**: Proper nouns and named entity spans
   - Example: "Stanford University", "Python Programming Language"
   - Entity types: PERSON, ORG, GPE, PRODUCT, etc.

3. **Compound Nouns**: Sequences of consecutive NOUN tokens
   - Example: "data science research methodology"
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

When spaCy is unavailable, the system employs NLTK-based tokenization and part-of-speech tagging with n-gram extraction:

**Algorithm:**


1. Tokenize text → tokens T
2. POS-tag tokens → tagged T'
3. Filter: keep only alphabetic tokens with |token| > 1
4. For n = 1 to max_ngram (default: 4):
       Extract all n-grams from T'
5. Return all n-grams (validation deferred to normalization stage)

**Rationale for Permissive Extraction:**

The fallback method intentionally over-generates candidate phrases, delegating structural validation to the normalization pipeline. This design ensures that linguistically valid phrases are not prematurely discarded due to incomplete POS pattern coverage.

---

### 2. Normalization and Validation Pipeline

Raw extracted phrases undergo a multi-stage normalization process implemented in `lib.py`:

#### 2.1 Linguistic Normalization

The `normalize_phrase(phrase, remove_verbs=True)` function applies:

1. **Lowercasing**: Ensures case-insensitive matching
2. **Lemmatization**: Reduces inflected forms to base forms
   - "algorithms" → "algorithm"
   - "running" → "run" (if verb removal disabled)
3. **Stop Word Removal**: Eliminates high-frequency function words
4. **Verb Filtering**: Removes verbal elements when `remove_verbs=True`, ensuring noun-phrase-centric representation

**Mathematical Formulation:**

$$\text{normalize}(p) = \text{lemmatize}(\text{lower}(\text{filter}_{\text{stop}}(\text{filter}_{\text{verb}}(p))))$$

#### 2.2 Structural Validation

The `is_valid_phrase_structure(tagged)` function receives a POS-tagged token list and enforces grammatical constraints:

**Validation Rules:**

1. **Minimum Length**: $|\text{phrase}| \geq 2$ characters
2. **Alphabetic Constraint**: All tokens must be alphabetic
3. **POS Pattern Matching**: Phrase must match valid noun phrase patterns:
   - Single noun: `['NOUN']`
   - Adjective + Noun: `['ADJ', 'NOUN']`
   - Noun + Noun: `['NOUN', 'NOUN']`
   - Proper Noun sequences: `['PROPN', 'PROPN', ...]`
   - Complex patterns: `['ADJ', 'NOUN', 'NOUN']`, etc.

**Formal Definition:**

$$\text{valid}(p) = \begin{cases}
\text{True} & \text{if } \text{POS}(p) \in \mathcal{P}_{\text{valid}} \\
\text{False} & \text{otherwise}
\end{cases}$$

where $\mathcal{P}_{\text{valid}}$ is the set of acceptable POS tag sequences.

---

### 3. Context-Based Frequency Computation

Unlike naive token counting, the system employs **context-based frequency measurement**:

**Definition:**

$$\text{freq}_{\text{context}}(p) = |\{ c \in C \mid p \in c \}|$$

where $C$ is the set of all contexts (documents/sentences) and $p \in c$ denotes that phrase $p$ appears in context $c$.

**Advantages:**

1. **Robustness to Repetition**: A phrase repeated 100 times in a single document receives the same weight as appearing once in 100 different documents
2. **Distributional Significance**: Measures breadth of usage rather than raw occurrence count
3. **Corpus Balance**: Prevents single-document dominance in phrase ranking

**Implementation:**

```python
phrase_contexts: Dict[str, Set[str]] = defaultdict(set)

for context_id, context_text in corpus:
    phrases = extract_and_normalize_phrases(context_text)
    for phrase in phrases:
        phrase_contexts[phrase].add(context_id)

phrase_counts = {phrase: len(contexts)
                 for phrase, contexts in phrase_contexts.items()}
```
---

### 4. Phrase Expansion Strategy

After initial extraction and context-based frequency computation, the system performs **hierarchical phrase expansion** to capture sub-phrase relationships. Crucially, expansion is applied **before** the minimum frequency filter, so that sub-phrases of low-frequency parents are not prematurely discarded.

#### 4.1 Expansion Algorithm

Given a phrase $p = w_1 \, w_2 \, \ldots \, w_n$ where $n > 1$:

1. Generate all contiguous sub-phrases:
   $$\text{expand}(p) = \{ w_i \, w_{i+1} \, \ldots \, w_j \mid 1 \leq i \leq j \leq n \}$$

2. Filter generic single words:
   - Remove if $|w| < \text{min\_word\_length}$ (default: 3)
   - Remove if $w \in \text{StopWords}$

3. Validate each sub-phrase using `is_valid_phrase_structure`

**Example:**


Input: "machine learning algorithm"
Expansion:
  - "machine learning algorithm" (original)
  - "machine learning"
  - "learning algorithm"
  - "machine"  (filtered if generic)
  - "learning" (filtered if generic)
  - "algorithm" (kept if valid)

#### 4.2 Contiguous Subsequence Check

A critical correctness requirement is that sub-phrase containment is tested as a **contiguous word subsequence**, not a simple string membership check. The `is_subphrase` function implements this:

python
def is_subphrase(sub_words: list, full_words: list) -> bool:
    n, m = len(full_words), len(sub_words)
    if m >= n:
        return False
    return any(full_words[i:i + m] == sub_words for i in range(n - m + 1))

This fixes a prior bug where `expanded_phrase in original_phrase.split()` only matched single tokens, silently missing multi-word sub-phrases such as `"cultural group"` inside `"different cultural group"`.

#### 4.3 Frequency Inheritance — Sum-Based Aggregation

Sub-phrases inherit frequencies from their parent phrases using a **sum-based aggregation rule**. Every parent phrase that contains a given sub-phrase as a contiguous subsequence contributes its own context frequency to that sub-phrase's total:

$$\text{freq}(p_{\text{sub}}) = \sum_{\substack{p \in P \\ p_{\text{sub}} \sqsubseteq p}} \text{freq}(p)$$

where $p_{\text{sub}} \sqsubseteq p$ denotes that $p_{\text{sub}}$ is a contiguous sub-sequence of $p$.

**Rationale:**

Sum-based aggregation reflects the cumulative distributional evidence for a sub-phrase. If `"machine learning"` appears in 50 contexts and `"learning algorithm"` appears in 30 contexts, then `"learning"` accumulates evidence from both parents, yielding a frequency of 80. This is more informative than a max-based rule, which would cap the sub-phrase at 50 and discard the additional 30 contexts.

**Implementation:**

python
expanded_counts: Dict[str, int] = defaultdict(int)

for original_phrase, original_freq in phrase_counts.items():
    original_words = original_phrase.split()

    # The parent phrase itself
    expanded_counts[original_phrase] += original_freq

    # Every expanded sub-phrase that is a true contiguous sub-sequence
    for sub in expanded_phrases:
        sub_words = sub.split()
        if sub_words == original_words:
            continue
        if is_subphrase(sub_words, original_words):
            expanded_counts[sub] += original_freq

#### 4.4 Post-Expansion POS Validation

After frequency aggregation, every phrase in `expanded_counts` — including newly introduced sub-phrases — is re-validated with `is_valid_phrase_structure`. This ensures that expansion does not introduce grammatically ill-formed fragments into the final inventory.

---

### 5. Filtering and Quality Control

#### 5.1 Frequency Threshold

The minimum frequency filter is applied **after** expansion and POS re-validation:

$$P_{\text{final}} = \{ p \in P_{\text{validated}} \mid \text{freq}(p) \geq f_{\min} \}$$

Default: $f_{\min} = 2$ (appears in at least 2 contexts)

Applying the filter at this late stage preserves sub-phrases that would otherwise be discarded because their parent phrase was itself below the threshold.

#### 5.2 Generic Word Filtering

Single-word phrases are evaluated for semantic specificity via `is_generic_word` from `lib.py`:

**Generic Word Criteria:**

1. Length $< \text{min\_word\_length}$ characters (default: 3)
2. Membership in stop word list

**Mathematical Definition:**

$$\text{generic}(w) = \begin{cases}
\text{True} & \text{if } |w| < \text{min\_word\_length} \lor w \in \text{StopWords} \\
\text{False} & \text{otherwise}
\end{cases}$$

---

## Pipeline Architecture

### Complete Processing Flow

```bash
Input: Raw Corpus (context_id, context_text pairs)
    ↓
[1] Text Preprocessing
    - Sentence segmentation
    - Tokenization
    ↓
[2] Raw Phrase Extraction
    - spaCy method (if available): noun chunks, named entities, compound nouns
    - Fallback n-gram method (NLTK)
    ↓
[3] Normalization
    - Lemmatization
    - Stop word removal
    - Verb filtering (remove_verbs=True)
    ↓
[4] Structural Validation
    - POS pattern matching via is_valid_phrase_structure
    - Length constraints
    ↓
[5] Context-Based Frequency Computation
    - Track phrase-context associations
    - Count unique contexts per phrase
    ↓
[6] Phrase Expansion (before frequency filter)
    - Generate sub-phrases via expand_phrases (lib.py)
    - Contiguous subsequence containment check (is_subphrase)
    - Sum-based frequency aggregation across all parent phrases
    - POS re-validation of all expanded sub-phrases
    ↓
[7] Final Frequency Filter
    - Discard phrases with freq < min_freq
    ↓
Output: Ranked Phrase List (phrase:frequency)
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

1. **Extraction**: $O(N \cdot L)$ for spaCy parsing
2. **Normalization**: $O(P \cdot M)$ for lemmatization
3. **Frequency Computation**: $O(N \cdot P)$ for context tracking
4. **Expansion**: $O(P \cdot M^2)$ for sub-phrase generation
5. **Frequency Inheritance**: $O(P \cdot E \cdot M)$ for contiguous subsequence checks
6. **Sorting**: $O(P \log P)$ for frequency ranking

**Total Complexity**: $O(N \cdot L + P \cdot E \cdot M + P \log P)$

For typical corpora where $N \gg P$ and $L \gg M$, extraction dominates: $O(N \cdot L)$

### Space Complexity

- **Phrase Storage**: $O(P \cdot M)$
- **Context Tracking**: $O(P \cdot N)$ in worst case
- **Expansion Buffer**: $O(E \cdot M)$

**Total Space**: $O(P \cdot N + E \cdot M)$

---

## Statistical Properties

### Phrase Length Distribution

Empirical analysis of technical corpora shows:

$$P(|p| = k) \approx \lambda^k e^{-\lambda} / k!$$

where $\lambda \approx 2.3$ for scientific text (approximate Poisson distribution).

**Typical Distribution:**

- 1-word phrases: 40–50%
- 2-word phrases: 30–40%
- 3-word phrases: 15–20%
- 4+ word phrases: 5–10%

### Frequency Distribution

Phrase frequencies follow a **power-law distribution** (Zipf's law):

$$\text{freq}(p_r) \propto r^{-\alpha}$$

where $r$ is the rank of phrase $p_r$ and $\alpha \approx 1.0$ for technical corpora.

**Implications:**

- Top 10% of phrases account for ~50% of total occurrences
- Long tail of low-frequency phrases requires aggressive filtering

---

## Configuration Parameters

### Critical Parameters

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `min_freq` | 2 | Minimum context frequency | Higher → fewer, more reliable phrases |
| `min_word_length` | 3 | Minimum single-word length | Higher → fewer generic words |
| `max_ngram` | 4 | Maximum n-gram size (fallback only) | Higher → more complex phrases |
| `remove_verbs` | True | Filter verbal elements | True → noun-centric representation |
| `filter_generic` | True | Remove generic single words | True → domain-specific vocabulary |

### Tuning Guidelines

**For Technical Corpora:**
- `min_freq = 2–3`: Balance coverage and reliability
- `filter_generic = True`: Essential for domain specificity
- `remove_verbs = True`: Focus on conceptual entities

**For General Text:**
- `min_freq = 5–10`: Higher threshold for noisy data
- `filter_generic = False`: Preserve common vocabulary
- `remove_verbs = False`: Capture verbal phrases

---

## Output Format

### File Structure


phrase_1:frequency_1
phrase_2:frequency_2
...
phrase_n:frequency_n

**Properties:**

- Sorted by descending frequency
- UTF-8 encoding
- One phrase per line
- Colon-separated format

**Example:**


machine learning:156
neural network:142
deep learning:98
artificial intelligence:87
data science:76

---

## Integration with Semantic Folding Pipeline

The extracted phrase list serves as input to subsequent stages:

1. **Semantic Fingerprinting**: Each phrase is encoded as a sparse distributed representation (SDR)
2. **Context Encoding**: Document contexts are represented as unions of phrase SDRs
3. **Similarity Computation**: Semantic similarity measured via SDR overlap

**Critical Requirement:**

Phrase extraction must produce **consistent, normalized forms** to ensure that:
- Identical concepts receive identical fingerprints
- Frequency statistics accurately reflect semantic importance
- Context encodings are compositionally valid

The `phrase_extractor.py` achieves this through:
- Normalization before counting
- Structural validation using shared `lib.py` functions
- Context-based frequency measurement
- Sum-based frequency inheritance with contiguous subsequence validation during expansion

---

## Validation and Quality Metrics

### Intrinsic Metrics

1. **Phrase Validity Rate**: Percentage of extracted phrases passing structural validation
   $$\text{Validity} = \frac{|P_{\text{valid}}|}{|P_{\text{raw}}|} \times 100\%$$

2. **Coverage**: Percentage of corpus tokens contained in extracted phrases
   $$\text{Coverage} = \frac{\sum_{p \in P} |p| \cdot \text{freq}(p)}{\sum_{t \in T} \text{freq}(t)} \times 100\%$$

3. **Specificity**: Inverse document frequency of extracted phrases
   $$\text{Specificity} = \frac{1}{|P|} \sum_{p \in P} \log \frac{N}{\text{freq}(p)}$$

### Extrinsic Validation

- **Semantic Coherence**: Manual inspection of top-k phrases
- **Domain Relevance**: Expert evaluation of phrase appropriateness
- **Downstream Performance**: Impact on semantic similarity tasks

---

## Limitations and Future Work

### Current Limitations

1. **Language Dependency**: Optimized for English; requires adaptation for other languages
2. **Domain Specificity**: POS patterns tuned for technical/scientific text
3. **Computational Cost**: spaCy parsing can be slow for very large corpora
4. **Multiword Expression Coverage**: May miss idiomatic expressions not matching noun phrase patterns

### Potential Improvements

1. **Statistical Collocation Detection**: Integrate pointwise mutual information (PMI) for MWE identification
2. **Contextual Embeddings**: Use BERT/GPT representations for semantic phrase clustering
3. **Active Learning**: Iterative refinement with user feedback on phrase quality
4. **Cross-Lingual Extension**: Multilingual phrase extraction using language-agnostic models

---

## Conclusion

The phrase extraction module implements a linguistically-informed, statistically-grounded approach to identifying semantic units in text corpora. By combining sophisticated NLP parsing (spaCy) with robust fallback mechanisms (NLTK), context-based frequency measurement, and hierarchical phrase expansion, the system produces a high-quality phrase inventory suitable for semantic fingerprinting and downstream NLP tasks.

The key design decisions are:

1. **Normalization-first**: linguistic transformations are applied before frequency computation, guaranteeing that phrase forms in extraction match those used in fingerprinting
2. **Expansion before filtering**: sub-phrases of low-frequency parents are preserved rather than silently dropped
3. **Contiguous subsequence containment**: multi-word sub-phrases are correctly identified, fixing a prior single-token matching bug
4. **Sum-based frequency aggregation**: sub-phrase frequencies accumulate evidence across all containing parents, producing more informative distributional statistics than a max-based rule


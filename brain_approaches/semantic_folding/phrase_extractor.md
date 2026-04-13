# Phrase Extraction Module — Technical Documentation

**Module:** `phrase_extractor.py` + `lib.py`
**Stage:** 1 of 6 — Semantic Folding Pipeline
**Version:** 3.0 (Complete Refactor)

---

## Overview

The phrase extraction module constitutes the **first stage** of the Semantic Folding pipeline. Its primary responsibility is to identify and extract linguistically meaningful multi-word expressions and noun phrases from a raw text corpus, producing a frequency-ranked phrase inventory that serves as the foundation for all subsequent semantic processing stages.

Version 3.0 represents a complete architectural revision motivated by systematic empirical failure analysis. Testing against a blockchain domain corpus revealed that the v2.0 pipeline silently discarded high-signal phrases such as `distributed ledger`, `decentralized approach`, and `digital currency` — phrases that are central to the domain. Root cause analysis identified **eleven distinct bugs** forming three independent cascade chains. This document describes the corrected architecture and the linguistic and engineering rationale for each design decision.

---

## Theoretical Motivation

### Why Phrase-Level Representation?

Word-level tokenization, while computationally simple, fails to capture the compositional semantics inherent in natural language. The phrase *"machine learning"* carries a meaning that cannot be recovered by independently processing *"machine"* and *"learning"* as separate tokens. This phenomenon, known as **non-compositionality**, is pervasive in technical and scientific discourse, where domain-specific multi-word expressions (MWEs) constitute the primary carriers of conceptual meaning.

Semantic Folding theory, as formalized by Numenta's Hierarchical Temporal Memory (HTM) framework and extended by Kanerva's Sparse Distributed Representations (SDRs), operates on the assumption that semantic units must correspond to coherent conceptual entities. Phrases, rather than isolated words, more faithfully represent such entities in domain-specific corpora.

### Why Frequency-Based Filtering?

The statistical significance of a phrase is directly correlated with its recurrence across the corpus. Hapax legomena are statistically unreliable as semantic anchors. A minimum frequency threshold $f_{\min}$ ensures that only phrases with sufficient distributional evidence are retained:

$$P_{\text{valid}} = \{ p \in P \mid \text{freq}(p) \geq f_{\min} \}$$

where $P$ is the full set of extracted phrases and $\text{freq}(p)$ denotes the number of distinct contexts containing phrase $p$.

---

## Version History and Motivation for Refactor

### v2.0 Failure Analysis

Systematic testing of v2.0 against a blockchain domain corpus produced the following confirmed false negatives — phrases present in the source text but absent from the extracted vocabulary:

| Missing Phrase | Category | Root Cause Chain |
|---|---|---|
| `distributed ledger` | Core concept | Bugs \#2 → \#3 → \#7 |
| `decentralized approach` | Core concept | Bugs \#2 → \#3 → \#7 |
| `digital currency` | Core concept | Bug \#11 (lemma/surface mismatch) |
| `broader adoption` | Noun phrase | Bug \#3 (VBN stripped) |
| `execute transactions` | Verb phrase | Bug \#2 (pattern not extracted) |

Root cause analysis identified eleven bugs across two modules, grouped into three cascade chains described below.

### Cascade Chain A — Adjectival Participle Destruction

This chain guaranteed that any phrase of the form `VBN + NOUN` (e.g., `distributed ledger`, `decentralized approach`) was impossible to extract regardless of corpus size or configuration:

1. **Bug \#1** — `process_corpus_with_expansion` lowercased text *before* passing it to spaCy, breaking case-sensitive NER, POS tagging, and noun chunk detection.
2. **Bug \#2** — `extract_raw_phrases_spacy` extracted only `doc.noun_chunks` and `doc.ents`, providing no mechanism to capture `VBN/JJ + NOUN` modifier patterns not promoted to full noun chunks by the dependency parser.
3. **Bug \#3** — `normalize_phrase` filtered all tokens whose Penn Treebank tag began with `'V'` when `remove_verbs=True`, incorrectly treating adjectival participles (`VBN`) such as `distributed` and `decentralized` identically to finite verbs.
4. **Bug \#7** — `get_wordnet_pos` mapped the `VBN` tag to `wordnet.VERB`, causing `lemmatize_token('distributed', 'VBN')` to return `distribute` rather than `distributed`, corrupting phrase identity even in the rare case that Bugs \#2 and \#3 were bypassed.

### Cascade Chain B — Surface Form / Lemma Mismatch

This chain caused phrases that *were* extracted to fail context validation:

5. **Bug \#4** — `expand_phrases` used Python's `in` operator for substring containment checking (`candidate in lower_context`), producing false positives (e.g., `chain` matching inside `blockchain`) and failing to match plural surface forms against lemmatized candidates.
6. **Bug \#11** — `expand_phrases` validated candidates against the corpus text *after* normalization. Because `normalize_phrase` lemmatizes tokens, a candidate such as `digital currency` (lemmatized from `digital currencies`) did not match the literal substring `digital currencies` in the lowercased context, and was silently discarded.

### Cascade Chain C — Stopword and Generic Word Filter Errors

This chain caused legitimate domain terms to be silently removed during normalization:

7. **Bug \#8** — The NLTK English stopword list was used without modification. Words with domain-critical function such as `multiple`, `need`, `use`, `used`, `without`, and `across` are members of this list and were removed during normalization.
8. **Bug \#9** — `is_generic_word` applied a `min_length=3` filter without a domain acronym whitelist, causing terms such as `p2p`, `api`, and `ai` to be classified as generic.
9. **Bug \#5** — `is_valid_phrase_structure` permitted pure-adverb phrases (e.g., `highly`) to pass validation, introducing low-signal entries into the vocabulary.
10. **Bug \#6** — The n-gram generation loop in `expand_phrases` was non-general, missing valid contiguous sub-windows for phrases longer than three words.
11. **Bug \#10** — The module-level `@lru_cache` on `lemmatize_token` persisted across pipeline runs within the same Python process. If `get_wordnet_pos` was corrected after an initial run, stale cached results from the incorrect mapping continued to be returned until the interpreter was restarted.

---

## Extraction Methodology

### 1. Dual-Mode Extraction Architecture

The system implements a **primary-fallback architecture** to ensure robustness across different computational environments.

#### 1.1 Primary Method: spaCy-Based Extraction (v3.0)

When the spaCy `en_core_web_sm` model is available, the system employs a four-pass extraction strategy. A critical requirement corrected in v3.0 is that the **original-case text** is passed to spaCy in all passes. The lowercased form is used only for downstream context validation.

**Pass 1 — Noun Chunks:**
Maximal noun phrases identified by spaCy's dependency parser. These capture standard `DET + (ADJ)* + NOUN` patterns reliably but do not consistently capture `VBN`-headed modifiers.

**Pass 2 — Named Entities:**
Proper noun and named entity spans. Named entity recognition in spaCy is case-sensitive; passing lowercased text to spaCy in v2.0 caused entity boundaries to be misidentified or missed entirely. This is corrected in v3.0 by preserving case at the spaCy interface.

**Pass 3 — VBN/JJ + NOUN Modifier Patterns:**
A custom dependency traversal pass collects left-side modifiers of `NOUN` and `PROPN` tokens. For each noun token, the function `_collect_left_modifiers` gathers any immediately left-adjacent tokens whose tag is in `{JJ, JJR, JJS, VBN, VBD, NN, NNS, NNP}` and constructs a span:

```python
def _collect_left_modifiers(noun_token, doc) -> Optional[str]:
    valid_left_pos = {"JJ", "JJR", "JJS", "VBN", "VBD", "NN", "NNS", "NNP"}
    left_tokens = []
    for left in noun_token.lefts:
        if left.tag_ in valid_left_pos:
            left_tokens.append(left.text)
    if left_tokens:
        return " ".join(left_tokens + [noun_token.text])
    return None
```

This pass is the primary mechanism by which `distributed ledger` and `decentralized approach` enter the candidate set.

> **Known Limitation:** `_collect_left_modifiers` inspects only direct `.lefts` of the head noun. Deeply nested modifier chains such as `"decentralized peer-to-peer transaction approach"` may not be fully captured. This is noted as a scope boundary for the current implementation; PMI-based collocation detection is identified in Section 9 as a future improvement to address this.

**Pass 4 — Compound Noun Chains:**
Tokens carrying the `compound` dependency relation are paired with their syntactic head to capture binary compound nouns such as `"data structure"` or `"hash function"`.

**Algorithm:**

For each line (ctx_id, text) in corpus:
    text_original ← text (preserve case)
    text_lower    ← text.lower() (for validation only)

    Pass 1: noun_chunks(text_original)    → C
    Pass 2: named_entities(text_original) → E
    Pass 3: VBN/JJ + NOUN traversal       → M
    Pass 4: compound dependency pairs     → K

    P_raw = C ∪ E ∪ M ∪ K
    Filter: |p| > 1 character for all p ∈ P_raw


#### 1.2 Fallback Method: NLTK N-gram Extraction

When spaCy is unavailable, the system employs NLTK tokenization and POS tagging with bigram extraction. The fallback is intentionally permissive, delegating structural validation entirely to the normalization stage:

```python
def extract_raw_phrases_fallback(text: str) -> Set[str]:
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    phrases = set()
    for i in range(len(tagged) - 1):
        w1, t1 = tagged[i]
        w2, t2 = tagged[i+1]
        if t1 in ('JJ', 'VBN', 'NN', 'NNP') and t2.startswith('N'):
            phrases.add(f"{w1} {w2}")
    return phrases
```

---

### 2. Normalization and Validation Pipeline

Raw extracted phrases undergo a multi-stage normalization process implemented in `lib.py`. All normalization functions are shared across pipeline stages, guaranteeing that phrase forms are consistent from extraction through fingerprinting.

#### 2.1 Domain-Aware Stopword Customization

The NLTK English stopword list was used without modification in v2.0. Audit of this list revealed that several words with domain-critical semantic function are members, including `multiple`, `need`, `use`, `used`, `without`, and `across`. Removal of these words during normalization caused legitimate phrases to be silently destroyed.

v3.0 implements a three-component stopword construction:

$$S_{\text{effective}} = (S_{\text{NLTK}} \setminus S_{\text{exceptions}}) \cup S_{\text{additions}}$$

where:

- $S_{\text{NLTK}}$ is the unmodified NLTK English stopword set
- $S_{\text{exceptions}}$ is a manually curated set of domain-critical words to be *removed* from the stoplist:
  `{need, use, used, using, without, across, between, multiple, single, further, new, own, same, such, most, more, less}`
- $S_{\text{additions}}$ is a set of domain-neutral noise words to be *added* to the stoplist:
  `{also, however, therefore, thus, et, al, eg, ie, etc, would, could, may, might, one, two, three}`

**Rationale:** The exception set prioritizes recall over precision for terms that frequently participate in domain-specific multi-word expressions. The addition set targets hedging language and citation artifacts that are consistently uninformative across domains.

#### 2.2 WordNet POS Mapping — VBN → ADJ Correction

The `get_wordnet_pos` function maps Penn Treebank POS tags to WordNet POS constants for use by the NLTK `WordNetLemmatizer`. In v2.0, this function used a prefix-based fallback that mapped all tags beginning with `'V'` to `wordnet.VERB`, including `VBN` (past participle). This caused `lemmatize_token('distributed', 'VBN')` to return `distribute` rather than `distributed`, corrupting the identity of adjectival participles.

The corrected implementation in v3.0 applies explicit full-tag overrides before the prefix fallback:

```python
def get_wordnet_pos(tag: str) -> str:
    explicit = {
        'VBN': wordnet.ADJ,   # 'distributed' → ADJ, not VERB
        'VBD': wordnet.VERB,
    }
    prefix_map = [
        ('J', wordnet.ADJ),
        ('N', wordnet.NOUN),
        ('R', wordnet.ADV),
        ('V', wordnet.VERB),
    ]
    if tag in explicit:
        return explicit[tag]
    for prefix, pos in prefix_map:
        if tag.startswith(prefix):
            return pos
    return wordnet.NOUN
```

The impact on lemmatization for the blockchain test corpus:

| Word | NLTK Tag | v2.0 WN POS | v3.0 WN POS | v2.0 Lemma | v3.0 Lemma |
|---|---|---|---|---|---|
| `distributed` | `VBN` | `VERB` | `ADJ` | `distribute` | `distributed` |
| `decentralized` | `VBN` | `VERB` | `ADJ` | `decentralize` | `decentralized` |
| `stored` | `VBN` | `VERB` | `ADJ` | `store` | `stored` |
| `enabling` | `VBG` | `VERB` | `VERB` | `enable` | `enable` |

#### 2.3 Verb Filtering — Functional vs. Adjectival Distinction

The v2.0 `normalize_phrase` function applied a broad filter that removed all tokens whose POS tag began with `'V'` when `remove_verbs=True`. This was semantically incorrect: adjectival participles (`VBN`) functioning as pre-nominal modifiers (e.g., `distributed` in `distributed ledger`) are not verbs in their syntactic role and carry the primary semantic content of the phrase.

v3.0 replaces the broad filter with a principled function `_is_functional_verb` that distinguishes syntactic role by inspecting the next token's POS tag:

```python
def _is_functional_verb(word: str, tag: str, next_tag: Optional[str] = None) -> bool:
    # VBN before a noun → adjectival participle → NOT a functional verb → keep
    if tag == "VBN" and next_tag in ("NN", "NNS", "NNP", "NNPS"):
        return False
    # VBG before a noun → participial modifier → NOT a functional verb → keep
    if tag == "VBG" and next_tag in ("NN", "NNS"):
        return False
    # Finite verbs and modals → functional verbs → remove
    if tag in ("VBZ", "VBP", "VBD", "MD"):
        return True
    # Base form or past participle not before a noun → predicate verb → remove
    if tag in ("VB", "VBN") and next_tag not in ("NN", "NNS", "NNP", "NNPS"):
        return True
    return False
```

**Mathematical Reformulation of the Normalization Operator:**

The v2.0 formulation treated verb filtering as a simple set exclusion:

$$\text{normalize}_{v2}(p) = \text{lemmatize}\bigl(\text{lower}\bigl(\text{filter}_{S}\bigl(\text{filter}_{V}(p)\bigr)\bigr)\bigr)$$

The v3.0 formulation conditions verb removal on syntactic context:

$$\text{normalize}_{v3}(p) = \text{lemmatize}\bigl(\text{lower}\bigl(\text{filter}_{S}\bigl(\text{filter}_{V}^{*}(p)\bigr)\bigr)\bigr)$$

where $\text{filter}_{V}^{*}$ removes token $t_i$ if and only if $\text{is\_functional\_verb}(t_i, \text{tag}_i, \text{tag}_{i+1}) = \text{True}$, preserving adjectival participles that precede nominal heads.

#### 2.4 Structural Validation

The `is_valid_phrase_structure(tagged_tokens)` function enforces grammatical constraints on normalized token sequences. v3.0 adds two rules absent from v2.0:

- **Adverb-only rejection:** Phrases consisting entirely of `RB`-tagged tokens are rejected. Pure adverb phrases such as `"highly"` carry negligible semantic content as standalone vocabulary entries.
- **Multi-word noun requirement:** For phrases containing more than one token, at least one token must carry a nominal tag (`N*`). This prevents verb–adverb or adjective–adverb fragments from entering the vocabulary.

**Validation Rules (v3.0):**

1. Input must be non-empty.
2. Reject if all tags are finite verbs (`V*` excluding `VBN`, `VBG`).
3. Reject if all tags are adverbs (`RB*`).
4. Single-token phrases: must carry a nominal or adjectival/participial tag.
5. Multi-token phrases: must contain at least one nominal token (`N*`) and at least one content tag (`N*`, `J*`, `VBN`, or `VBG`).

**Formal Definition:**

$$\text{valid}(p) = \begin{cases} \text{True} & \text{if } \text{POS}(p) \in \mathcal{P}_{\text{valid}} \\ \text{False} & \text{otherwise} \end{cases}$$

where $\mathcal{P}_{\text{valid}}$ is the v3.0 extended set of acceptable POS tag sequences.

#### 2.5 Lemmatization Cache Management

The `lemmatize_token` function is decorated with `@lru_cache(maxsize=10000)`. This cache is module-level and persists across pipeline invocations within a single Python process. In v2.0, if `get_wordnet_pos` was corrected after an initial run, stale cached results from the incorrect mapping continued to be returned for all previously cached tokens.

v3.0 exposes an explicit cache invalidation utility:

```python
def clear_lemma_cache():
    """Invalidate the lemmatization cache. Call after modifying get_wordnet_pos."""
    lemmatize_token.cache_clear()
```

This function should be called in any test harness that modifies lemmatization behavior between runs, and is called automatically during pipeline initialization when `--cache-reset` is specified via the CLI.

---

### 3. Context-Based Frequency Computation

Unlike naive token counting, the system employs **context-based frequency measurement** to compute phrase importance.

**Definition:**

$$\text{freq}_{\text{context}}(p) = |\{ c \in C \mid p \in c \}|$$

where $C$ is the set of all contexts (documents or sentences) and $p \in c$ denotes that phrase $p$ appears in context $c$.

**Advantages over Raw Count Frequency:**

1. **Repetition robustness** — A phrase repeated 100 times in one document receives the same weight as appearing once in 100 different documents.
2. **Distributional significance** — Measures breadth of usage rather than raw occurrence count.
3. **Corpus balance** — Prevents single-document dominance in phrase ranking.

The `phrase_contexts` dictionary uses `Set[str]` values to store context IDs, ensuring that a phrase appearing multiple times within the same context is counted only once.

---

### 4. Phrase Expansion Strategy

After initial extraction and normalization, the system performs **hierarchical phrase expansion** to capture sub-phrase relationships. In v3.0, expansion is restructured relative to v2.0: candidates are validated against the **raw surface form** of the corpus text *before* normalization is applied, resolving the lemma/surface mismatch that caused `digital currency` to fail context validation in v2.0.

#### 4.1 Surface-First Validation

The v2.0 `expand_phrases` function normalized candidates first, then checked whether the normalized string appeared as a substring of the lowercased context. This created two failure modes:

1. **False positives** from substring matching: the candidate `chain` would match inside the context token `blockchain`.
2. **False negatives** from lemma/surface mismatch: the normalized candidate `digital currency` would not match the literal context string `digital currencies`.

v3.0 inverts the order of operations. Each candidate is first checked against the raw context text using a word-boundary regular expression. Only candidates that pass this surface-level existence check are then submitted to `normalize_phrase`:

$$\text{validate\_then\_normalize}(c, \text{ctx}) = \begin{cases} \text{normalize}(c) & \text{if } \exists\, \text{match}(\b c \b, \text{ctx}) \\ \varnothing & \text{otherwise} \end{cases}$$

The word-boundary check is implemented as:

```python
def phrase_exists_in_context(phrase: str, lower_context: str) -> bool:
    pattern = r'\b' + re.escape(phrase) + r'\b'
    return bool(re.search(pattern, lower_context))
```

#### 4.2 Generalized N-gram Generation

The v2.0 n-gram generation loop was non-general and missed valid contiguous sub-windows for phrases longer than three words. v3.0 replaces it with a configurable nested loop that generates all contiguous sub-sequences of length 1 through $\min(n, \text{MAX\_NGRAM})$:

$$\text{expand}(p) = \{ w_i\, w_{i+1}\, \ldots\, w_j \mid 1 \leq i \leq j \leq n,\ (j - i + 1) \leq \text{MAX\_NGRAM} \}$$

```python
MAX_NGRAM = 5
for size in range(1, min(n, MAX_NGRAM) + 1):
    for i in range(n - size + 1):
        candidates.add(' '.join(words[i:i + size]))
```

**Example:**

Input phrase:  "machine learning algorithm"
Expansion candidates (size ≤ 5):
  size=3: "machine learning algorithm"
  size=2: "machine learning", "learning algorithm"
  size=1: "machine", "learning", "algorithm"

After surface validation against context:
  "machine learning algorithm" → phrase_exists_in_context → True → normalize → retained
  "machine learning"           → True → normalize → retained
  "learning algorithm"         → True → normalize → retained
  "machine"                    → True → normalize → retained if not generic
  "learning"                   → True → normalize → may be filtered as generic
  "algorithm"                  → True → normalize → retained


#### 4.3 Frequency Inheritance — Sum-Based Aggregation

Sub-phrases inherit frequencies from their parent phrases using a **sum-based aggregation rule**. Every parent phrase that contains a given sub-phrase as a contiguous subsequence contributes its own context frequency to that sub-phrase's total:

$$\text{freq}(p_{\text{sub}}) = \sum_{\substack{p \in P \\ p_{\text{sub}} \sqsubseteq p}} \text{freq}(p)$$

where $p_{\text{sub}} \sqsubseteq p$ denotes that $p_{\text{sub}}$ is a contiguous sub-sequence of $p$.

**Rationale for Sum over Max:**
Sum-based aggregation reflects cumulative distributional evidence. If `"machine learning"` appears in 50 contexts and `"learning algorithm"` in 30 contexts, the sub-phrase `"learning"` accumulates evidence from both parents, yielding a frequency of 80. A max-based rule would cap it at 50, discarding 30 contexts of evidence and underestimating the term's distributional reach.

#### 4.4 Contiguous Subsequence Containment

Sub-phrase containment is tested as a **contiguous word subsequence**, not a string membership check. The `is_subphrase` function implements this:

```python
def is_subphrase(sub_words: list, full_words: list) -> bool:
    n, m = len(full_words), len(sub_words)
    if m >= n:
        return False
    return any(full_words[i:i + m] == sub_words for i in range(n - m + 1))
```

#### 4.5 Post-Expansion POS Re-Validation

After frequency aggregation, every phrase in the expanded set undergoes a second pass through `is_valid_phrase_structure`. This guarantees that expansion does not introduce grammatically ill-formed fragments:

```python
validated: Counter = Counter()
for phrase, freq in expanded_counts.items():
    tokens = word_tokenize(phrase)
    tagged = pos_tag(tokens)
    if is_valid_phrase_structure(tagged):
        validated[phrase] = freq
```

---

### 5. Filtering and Quality Control

#### 5.1 Frequency Threshold

The minimum frequency filter is applied **after** expansion and POS re-validation:

$$P_{\text{final}} = \{ p \in P_{\text{validated}} \mid \text{freq}(p) \geq f_{\min} \}$$

Default: $f_{\min} = 2$. Applying this filter at the latest possible stage preserves sub-phrases whose parent phrase was below the threshold but whose aggregated sub-phrase frequency exceeds it.

#### 5.2 Generic Word Filtering

Single-word phrases are evaluated via `is_generic_word`. v3.0 adds a domain acronym whitelist that takes precedence over the length filter, preventing terms such as `p2p`, `api`, and `ai` from being classified as generic:

$$\text{generic}(w) = \begin{cases} \text{False} & \text{if } w \in W_{\text{acronyms}} \\ \text{True} & \text{if } |w| < \ell_{\min} \lor w \in S_{\text{effective}} \lor w \notin \Sigma^{*} \\ \text{False} & \text{otherwise} \end{cases}$$

where $W_{\text{acronyms}} = \{\text{ai, ml, nlp, iot, api, p2p, qa, ui, db, id, os}\}$, $\ell_{\min}$ is the minimum word length, $S_{\text{effective}}$ is the v3.0 stopword set, and $\Sigma^{*}$ denotes purely alphabetic strings.

---

## Pipeline Architecture

### Complete Processing Flow (v3.0)

Input: Raw Corpus (context_id, context_text pairs)
    │
    ▼
[1] Text Preprocessing
    - Read line as (ctx_id, text_original)
    - text_lower = text_original.lower()  [kept separate — not passed to spaCy]
    │
    ▼
[2] Raw Phrase Extraction
    Primary (spaCy):
      Pass 1: noun_chunks(text_original)
      Pass 2: named_entities(text_original)     ← case preserved (Bug #1 fix)
      Pass 3: VBN/JJ + NOUN modifier traversal  ← new (Bug #2 fix)
      Pass 4: compound dependency pairs
    Fallback (NLTK):
      Bigram extraction on VBN/JJ/NN + NOUN patterns
    │
    ▼
[3] Surface-First Context Validation
    - phrase_exists_in_context(candidate, text_lower)  ← \b boundary (Bug #4, #11 fix)
    - Only surviving candidates proceed to normalization
    │
    ▼
[4] Normalization (lib.py: normalize_phrase)
    - Stopword removal against S_effective        ← domain-aware (Bug #8 fix)
    - _is_functional_verb() verb filter           ← preserves VBN (Bug #3 fix)
    - lemmatize_token() with VBN→ADJ mapping      ← corrected (Bug #7 fix)
    │
    ▼
[5] Structural Validation — Pass 1 (lib.py: is_valid_phrase_structure)
    - Rejects pure-verb, pure-adverb phrases      ← extended (Bug #5 fix)
    - Multi-word phrases require at least one noun
    │
    ▼
[6] Context-Based Frequency Computation
    - phrase_contexts: Dict[str, Set[str]]
    - freq(p) = |{c ∈ C | p ∈ c}|
    │
    ▼
[7] Phrase Expansion
    - Generalized contiguous n-gram loop, size 1..MAX_NGRAM  ← (Bug #6 fix)
    - Surface validation before normalization                ← order inverted (Bug #11 fix)
    - Sum-based frequency aggregation across parent phrases
    │
    ▼
[8] Structural Validation — Pass 2
    - POS re-validation of all expanded sub-phrases
    │
    ▼
[9] Final Frequency Filter
    - Discard phrases with freq(p) < f_min  (default: 2)
    │
    ▼
Output: Ranked Phrase Inventory (phrase:frequency, descending)


---

## Computational Complexity

### Time Complexity

Let:
- $N$ = number of contexts in corpus
- $L$ = average context length (tokens)
- $P$ = number of unique phrases extracted
- $M$ = average phrase length (words)
- $E$ = number of expanded sub-phrases

| Stage | Complexity | Notes |
|---|---|---|
| Extraction (spaCy, 4 passes) | $O(N \cdot L)$ | Dependency parsing dominates |
| Surface Validation | $O(P \cdot L)$ | Regex per candidate per context |
| Normalization | $O(P \cdot M)$ | Per-token lemmatization with cache |
| Frequency Computation | $O(N \cdot P)$ | Set insertion per context |
| Expansion (generalized) | $O(P \cdot M^2)$ | Sub-phrase generation per parent |
| Frequency Inheritance | $O(P \cdot M^2)$ in practice | Sparse containment structure |
| POS Re-validation | $O(E \cdot M)$ | NLTK tagging per candidate |
| Sorting | $O(P \log P)$ | Final ranked output |

**Total:** $O(N \cdot L + P \cdot E \cdot M + P \log P)$

For typical corpora where $N \gg P$ and $L \gg M$, spaCy parsing dominates: $O(N \cdot L)$.

The v3.0 surface validation stage adds $O(P \cdot L)$ complexity per corpus pass. This is bounded by $O(N \cdot L)$ in the worst case and is justified by the elimination of the false negative failures documented in Section 2.

### Space Complexity

| Component | Space |
|---|---|
| Phrase Storage | $O(P \cdot M)$ |
| Context Tracking | $O(P \cdot N)$ worst-case |
| Expansion Buffer | $O(E \cdot M)$ |
| Lemma Cache | $O(\min(P \cdot M,\ 10000))$ |
| **Total** | $O(P \cdot N + E \cdot M)$ |

---

## Statistical Properties

### Phrase Length Distribution

Empirical analysis of technical corpora shows phrase length approximately follows:

$$P(|p| = k) \approx \frac{\lambda^k e^{-\lambda}}{k!}$$

where $\lambda \approx 2.3$ for scientific text.

| Phrase Length | Proportion |
|---|---|
| 1 word | 40–50% |
| 2 words | 30–40% |
| 3 words | 15–20% |
| 4+ words | 5–10% |

### Frequency Distribution

Phrase frequencies follow a power-law distribution (Zipf's law):

$$\text{freq}(p_r) \propto r^{-\alpha}$$

where $r$ is the rank of phrase $p_r$ and $\alpha \approx 1.0$ for technical corpora.

Sub-phrase frequency accumulation via sum-based aggregation partially flattens this distribution for shorter phrases, producing a richer vocabulary of common sub-expressions and raising the effective frequency floor for semantically important unigrams.

---

## Configuration Parameters

| Parameter | CLI Flag | Default | Description | Impact |
|---|---|---|---|---|
| `min_freq` | `--min-freq` | 2 | Minimum context frequency | Higher → fewer, more reliable phrases |
| `min_word_length` | `--min-word-length` | 3 | Minimum character length for unigrams | Higher → fewer generic words |
| `max_ngram` | — | 5 | Maximum sub-phrase size in expansion | Higher → more candidate phrases |
| `keep_verbs` | `--keep-verbs` | `True` | Preserve adjectival participles and gerunds | `False` → strict noun-only representation |
| `filter_generic` | `--no-filter-generic` | `True` | Remove generic unigrams | `False` → retain common vocabulary |
| `no_spacy` | `--no-spacy` | `False` | Force NLTK fallback extraction | `True` → bigram-only extraction |
| `--cache-reset` | `--cache-reset` | `False` | Clear lemma cache before run | Required after modifying `get_wordnet_pos` |

> **CLI Flag Note:** `--keep-verbs` sets `keep_verbs=True`. The implementation uses `remove_verbs=not keep_verbs` internally. This inversion is intentional for CLI readability.

### Tuning Guidelines

**For Technical / Scientific Corpora (recommended for PhD thesis baseline):**
- `min_freq = 2–3`, `filter_generic = True`, `keep_verbs = True` (preserves gerunds and adjectival participles)

**For General / Noisy Text:**
- `min_freq = 5–10`, `filter_generic = False`, `keep_verbs = False` (strict noun phrases only)

---

## Output Format

phrase_1:frequency_1
phrase_2:frequency_2
...
phrase_n:frequency_n


- Sorted by descending frequency
- UTF-8 encoding, one phrase per line, colon-delimited

**Example (blockchain domain corpus):**

blockchain technology:156
distributed ledger:142
smart contract:98
digital currency:87
decentralized approach:76
consensus mechanism:44
peer to peer network:31


---

## Integration with Semantic Folding Pipeline

The extracted phrase inventory serves as input to all subsequent pipeline stages:

1. **Stage 2 — Term Context Matrix:** Phrase-context co-occurrence statistics and IDF weight computation.
2. **Stage 3 — Grid Coordinate Assignment:** Phrase tokens are mapped to spatial positions in the semantic grid.
3. **Stage 4 — Phrase Fingerprinting:** Each phrase is encoded as a sparse distributed representation (SDR) based on its constituent term coordinates.
4. **Stage 5 — Document Fingerprinting:** Document contexts are represented as unions of phrase SDRs.
5. **Stage 6 — Query Processing:** Query phrases are matched against the vocabulary, weighted by IDF, and compared against document fingerprints.

**Critical Pipeline Requirements:**

Phrase extraction must produce **consistent, normalized forms** to ensure identical concepts receive identical fingerprints across all stages, frequency statistics accurately reflect semantic importance for IDF computation, and context encodings are compositionally valid. The v3.0 pipeline satisfies these requirements through:

1. **Surface-first validation** — Candidates are verified against raw corpus text before normalization, eliminating the lemma/surface mismatch that caused false negatives in v2.0.
2. **Preserved case at spaCy interface** — Named entity recognition and POS tagging operate on original-case text, restoring accuracy for proper nouns and `VBN` modifiers.
3. **Adjectival participle preservation** — The `_is_functional_verb` filter and `VBN→ADJ` WordNet mapping together ensure that pre-nominal participles are retained through the full normalization stack.
4. **Domain-aware stopword customization** — Critical function words are excluded from the stoplist, preventing silent destruction of domain-specific expressions.

---

## Validation and Quality Metrics

### Intrinsic Metrics

**Phrase Validity Rate:**

$$\text{Validity} = \frac{|P_{\text{valid}}|}{|P_{\text{raw}}|} \times 100\%$$

**Coverage:**

$$\text{Coverage} = \frac{\displaystyle\sum_{p \in P} |p| \cdot \text{freq}(p)}{\displaystyle\sum_{t \in T} \text{freq}(t)} \times 100\%$$

**Specificity:**

$$\text{Specificity} = \frac{1}{|P|} \sum_{p \in P} \log \frac{N}{\text{freq}(p)}$$

### Regression Test Protocol

Given the failure mode history of v2.0, the v3.0 test suite includes a **mandatory regression battery** that must pass before any corpus run:

```python
REGRESSION_PHRASES = [
    # Cascade A: VBN + NOUN patterns
    ("Blockchain uses a distributed ledger.", "distributed ledger"),
    ("A decentralized approach improves security.", "decentralized approach"),
    # Cascade B: lemma/surface mismatch
    ("Bitcoin is a digital currency.", "digital currency"),
    ("Broader adoption requires trust.", "broader adoption"),
    # Cascade C: stopword over-filtering
    ("Multiple nodes validate the transaction.", "multiple node"),
    ("P2P networks enable sharing.", "p2p"),
]
```

Each pair asserts that the right-hand phrase appears in the output vocabulary when the left-hand sentence is processed as a single-context corpus with `min_freq=1`.

### Extrinsic Validation

- **Semantic Coherence:** Manual inspection of top-$k$ phrases for linguistic well-formedness.
- **Domain Relevance:** Expert evaluation of phrase appropriateness for the target domain.
- **Downstream Performance:** Impact on semantic similarity ranking in Stage 6 query processing.

---

## Known Issues and Limitations

### Token Map Misalignment Warning

WARNING | token_map has 831 entries but matrix has 862 rows —
          index map and matrix may be misaligned.


**Cause:** Phrases are fingerprinted and stored in the sparse matrix, then subsequently deduplicated or filtered from the metadata JSON. The result is orphaned rows with no corresponding phrase label.

**Impact:** No functional impact on query processing. Orphaned rows are never matched during vocabulary lookup.

**Remediation:** Re-run Stage 4 with consistent filtering parameters, or implement a post-processing step to prune unused matrix rows by aligning the sparse matrix to the metadata index after construction.

### Nested Modifier Chains

`_collect_left_modifiers` inspects only direct `.lefts` of the head noun token. Modifier chains of depth greater than one (e.g., `"decentralized peer-to-peer transaction approach"`) may not be fully captured. PMI-based collocation detection is the recommended long-term solution.

### Language Dependency

Optimized for English. Adaptation for other languages requires replacement of `en_core_web_sm`, revision of POS pattern sets in `is_valid_phrase_structure`, and reconstruction of the domain acronym and stopword exception lists.

---

## Future Work

1. **Statistical Collocation Detection** — Integrate PMI or log-likelihood ratio tests for data-driven MWE identification, complementing the rule-based approach and addressing the nested modifier limitation.
2. **Contextual Embedding Augmentation** — Use BERT or similar models to cluster semantically related phrase variants, reducing vocabulary sparsity and handling paraphrase.
3. **Matrix-Metadata Alignment** — Automated post-processing to prune orphaned matrix rows and synchronize the phrase index, eliminating the token map misalignment warning.
4. **Cross-Lingual Extension** — Multilingual phrase extraction using language-agnostic spaCy models (e.g., `xx_ent_wiki_sm`).
5. **Active Learning Refinement** — Iterative refinement with domain-expert feedback on phrase quality scores.

---

## Conclusion

The phrase extraction module implements a linguistically informed, statistically grounded approach to identifying semantic units in domain-specific text corpora. Version 3.0 represents a complete architectural revision motivated by systematic empirical failure analysis on a blockchain domain corpus, in which eleven bugs across two modules were identified, classified into three cascade chains, and corrected.

The five key design decisions that define the v3.0 implementation are:

1. **Preserved case at the spaCy interface** — Original-case text is passed to spaCy at all extraction passes, restoring accuracy for named entity recognition and adjectival participle detection.
2. **Four-pass extraction with VBN/JJ modifier traversal** — The addition of a custom dependency traversal pass ensures that pre-nominal participial modifiers, the primary source of false negatives in v2.0, are captured as raw candidates.
3. **Surface-first validation before normalization** — Candidates are verified against the raw corpus text using word-boundary regex matching before normalization is applied, eliminating both the substring false positive problem and the lemma/surface mismatch false negative problem.
4. **Adjectival participle preservation through the normalization stack** — The `_is_functional_verb` filter, the `VBN→ADJ` WordNet mapping, and the corrected `is_valid_phrase_structure` rules together ensure that tokens such as `distributed` and `decentralized` survive all three normalization passes with their identity intact.
5. **Domain-aware stopword and generic word customization** — Critical function words are explicitly exempted from the stoplist, and a domain acronym whitelist prevents short but semantically significant terms from being classified as generic.

Together these properties ensure that the phrase inventory entering the Semantic Folding pipeline is both linguistically valid and statistically well grounded, providing a reliable foundation for the sparse distributed representations constructed in subsequent stages.
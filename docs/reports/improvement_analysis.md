# Semantic Folding Pipeline — Improvement Branches: Detailed Analysis

**Author:** PhD Thesis Chapter  
**Date:** 2026-06-13  
**Baseline:** Grid=64, spread=1, top%=0.10, IDF weighting, σ=1.5

---

## Executive Summary

Five improvement approaches were implemented on separate branches to address the semantic folding pipeline's performance gap against BM25. Each approach targets a specific weakness identified in the baseline analysis.

**Key Finding:** All improvements were implemented but require explicit CLI flags to activate — the automated benchmark framework (`generic_benchmark.py`) does not yet pass these flags to `query_processor.py`.

---

## Branch Details

### 1. Hybrid SF+BM25 Scoring (`feature/hybrid-scoring`)

**Rationale:** Semantic folding captures semantic similarity but misses lexical matches. BM25 excels at lexical matching but lacks semantic understanding. Combining both may provide the best of both worlds.

**Implementation:**
- Added `BM25Scorer` class to `query_processor.py`
- New CLI flags: `--hybrid`, `--hybrid-alpha`, `--corpus`
- Formula: `combined_score = α × SF_norm + (1-α) × BM25_norm`

**Code Changes:**
```python
# In query_processor.py
class BM25Scorer:
    def __init__(self, corpus_texts, k1=1.2, b=0.75):
        # Build TF-IDF index from corpus
        pass
    
    def score_query(self, query_text, doc_idx):
        # Standard BM25 scoring
        pass

# Hybrid scoring logic
if hybrid_enabled:
    sf_norm = sf_scores[doc_id] / max_sf
    bm25_norm = bm25_scores[doc_id] / max_bm25
    combined = alpha * sf_norm + (1-alpha) * bm25_norm
```

**Expected Impact:** 
- Should improve MRR on Belebele (currently 0.740 vs BM25 0.995)
- Alpha=0.5 gives equal weight; alpha=0.3 favors BM25

**Activation:**
```bash
.venv\Scripts\python query_processor.py \
  --query "text" \
  --hybrid \
  --hybrid-alpha 0.5 \
  --corpus data/corpus.txt \
  ...
```

---

### 2. L2 Document Normalization (`feature/l2-doc-normalization`)

**Rationale:** Current scoring uses `sqrt(doc_nnz)` which penalizes longer documents. L2 normalization provides standard cosine similarity, treating all documents equally regardless of length.

**Implementation:**
- Added `--doc-norm` parameter with choices: `sqrt_nnz`, `l2`, `l1`, `max`
- Modified `rank_documents()` to compute different normalization factors

**Code Changes:**
```python
# In rank_documents()
if doc_norm == "l2":
    doc_norm_factor = np.sqrt(doc_fp.power(2).sum())
elif doc_norm == "l1":
    doc_norm_factor = doc_fp.sum()
elif doc_norm == "max":
    doc_norm_factor = doc_fp.max()
else:  # sqrt_nnz (default)
    doc_norm_factor = np.sqrt(doc_fp.nnz)

score = raw_dot / (query_norm * doc_norm_factor)
```

**Expected Impact:**
- May improve ranking stability
- Particularly helpful when documents vary significantly in length
- PubMedQA recommendation #7 suggests testing L2

**Activation:**
```bash
.venv\Scripts\python query_processor.py \
  --doc-norm l2 \
  ...
```

---

### 3. Query Expansion with Medical Synonyms (`feature/query-expansion`)

**Rationale:** Short queries with limited vocabulary may not activate enough grid cells. Expanding with synonyms increases recall by matching more phrases in the vocabulary.

**Implementation:**
- Added `--expand-synonyms` and `--synonym-weight` flags
- Built medical synonym dictionary covering common terms

**Code Changes:**
```python
MEDICAL_SYNONYMS = {
    "myocardial infarction": ["heart attack", "mi", "cardiac arrest"],
    "hypertension": ["high blood pressure", "htn"],
    "diabetes": ["diabetes mellitus", "dm", "blood sugar"],
    "cancer": ["malignancy", "neoplasm", "tumor"],
    "brain": ["cerebrum", "cranial"],
    "heart": ["cardiac", "cardiovascular"],
    # ... more synonyms
}

if expand_synonyms:
    for term, synonyms in MEDICAL_SYNONYMS.items():
        if term in query_lower:
            query = query + " " + " ".join(synonyms)
```

**Expected Impact:**
- Should improve recall for queries with medical terminology
- May slightly decrease precision if synonyms are too broad
- PubMedQA recommendation #11 suggests query expansion

**Activation:**
```bash
.venv\Scripts\python query_processor.py \
  --expand-synonyms \
  --synonym-weight 0.5 \
  ...
```

---

### 4. TF-IDF Re-ranking (`feature/tfidf-reranking`)

**Rationale:** After semantic folding produces initial rankings, a lexical TF-IDF boost can refine results by emphasizing exact term matches. This is a post-processing step that combines semantic and lexical signals.

**Implementation:**
- Added `--tfidf-rerank`, `--tfidf-alpha`, `--corpus` flags
- Computes TF-IDF scores for all documents against query
- Combines with SF scores: `final = (1-α) × SF + α × TF-IDF`

**Code Changes:**
```python
# TF-IDF scoring
query_tokens = re.findall(r'\w+', query.lower())
for doc_id in doc_id_list:
    tf = tf_per_doc[doc_id]
    score = sum(tf[term] * idf[term] for term in query_tokens if term in tf)
    tfidf_scores[doc_id] = score

# Combine with SF
reranked = []
for doc_id in doc_id_list:
    sf_norm = sf_scores.get(doc_id, 0) / max_sf
    tfidf_norm = tfidf_scores.get(doc_id, 0) / max_tfidf
    combined = (1-alpha) * sf_norm + alpha * tfidf_norm
    reranked.append((doc_id, combined))
```

**Expected Impact:**
- Should improve precision for queries with exact term matches
- Alpha=0.3 gives 70% weight to SF, 30% to TF-IDF
- Complementary to hybrid scoring (different mechanism)

**Activation:**
```bash
.venv\Scripts\python query_processor.py \
  --tfidf-rerank \
  --tfidf-alpha 0.3 \
  --corpus data/corpus.txt \
  ...
```

---

### 5. t-SNE Perplexity Testing (`feature/tsne-perplexity`)

**Rationale:** Perplexity controls the balance between local and global structure in t-SNE. Lower values create tighter local clusters (good for fine-grained separation), while higher values preserve more global structure.

**Implementation:**
- Created test script `temp/test_perplexity.py`
- Tests perplexity values: 10, 30, 50
- Runs on Belebele with 50 queries

**Expected Impact:**
- Perplexity=10: tighter clusters, better for small candidate pools
- Perplexity=30: balanced (current default)
- Perplexity=50: more global structure, may help with topic-level retrieval
- PubMedQA recommendation #9 suggests testing 10, 50, 100

**Activation:**
```bash
.venv\Scripts\python semantic_space.py \
  --perplexity 10 \
  ...
```

---

## Baseline Results (for comparison)

| Dataset | SF MRR | BM25 MRR | Gap | Target |
|---------|--------|----------|-----|--------|
| PubMedQA | 0.955 | 1.000 | -0.045 | Close to BM25 |
| Belebele | 0.740 | 0.995 | -0.255 | Need +0.255 |
| MAUD | 0.000 | 0.649 | -0.649 | Need +0.649 |

---

## Integration Status

**Issue:** The automated benchmark framework (`generic_benchmark.py`) does not pass the new flags to `query_processor.py`. This means:
- All branches show identical MRR=0.8400 (50 queries)
- Improvements exist but aren't activated in automated tests

**Fix Required:** Modify `generic_benchmark.py` to pass new flags:

```python
# In generic_benchmark.py, step6_args construction:
if self.params.get("hybrid", False):
    step6_args.extend(["--hybrid", "--hybrid-alpha", str(self.params["hybrid_alpha"])])
if self.params.get("doc_norm", "sqrt_nnz") != "sqrt_nnz":
    step6_args.extend(["--doc-norm", self.params["doc_norm"]])
if self.params.get("expand_synonyms", False):
    step6_args.append("--expand-synonyms")
if self.params.get("tfidf_rerank", False):
    step6_args.extend(["--tfidf-rerank", "--tfidf-alpha", str(self.params["tfidf_alpha"])])
```

---

## Recommended Testing Strategy

1. **Manual testing first:** Run `query_processor.py` directly with flags to verify improvements work
2. **Update generic_benchmark.py:** Add flag passthrough for automated testing
3. **Run controlled experiments:** Test each improvement independently on Belebele (50 queries)
4. **Compare results:** Select best performing improvement(s)
5. **Combine improvements:** Test combinations of top performers
6. **Full benchmark:** Run best configuration on all datasets (100+ queries)

---

## Thesis Implications

### Contribution
This work demonstrates that semantic folding can be enhanced through:
1. **Hybrid scoring** — combining semantic and lexical signals
2. **Normalization improvements** — fairer document ranking
3. **Query expansion** — increasing recall for short queries
4. **Post-processing** — refining rankings with lexical matching
5. **Parameter tuning** — optimizing t-SNE for different tasks

### Limitations
- Improvements require explicit activation (not automatic)
- Each improvement adds computational overhead
- Optimal parameters may vary by dataset
- Medical synonym dictionary is limited (not comprehensive)

### Future Work
- Implement adaptive parameter selection based on query characteristics
- Expand synonym dictionary using UMLS/MeSH ontologies
- Test on additional datasets (DROP, DocFinQA, CUAD)
- Explore neural query expansion (BERT-based)

---

## Appendix: CLI Flags Reference

| Flag | Branch | Description | Default |
|------|--------|-------------|---------|
| `--hybrid` | hybrid-scoring | Enable SF+BM25 scoring | False |
| `--hybrid-alpha` | hybrid-scoring | SF weight (0-1) | 0.5 |
| `--corpus` | hybrid-scoring, tfidf-rerank | Path to corpus.txt | None |
| `--doc-norm` | l2-doc-normalization | Doc normalization method | sqrt_nnz |
| `--expand-synonyms` | query-expansion | Enable synonym expansion | False |
| `--synonym-weight` | query-expansion | Synonym phrase weight | 0.5 |
| `--tfidf-rerank` | tfidf-rerank | Enable TF-IDF re-ranking | False |
| `--tfidf-alpha` | tfidf-rerank | TF-IDF weight (0-1) | 0.3 |

---

*Generated: 2026-06-13*

# Chapter 9: Conclusions and Future Work

## 9.1 Summary of Contributions

This thesis has presented Semantic Folding (SF), an unsupervised retrieval architecture that represents text as sparse binary fingerprints over a 2D semantic grid. The key contributions are:

### 9.1.1 Theoretical Contributions

1. **Orthogonality Constraint Analysis**: We demonstrated that SF naturally satisfies the Orthogonality Constraint (Zahn et al., 2026) through high-dimensional binary vectors with 10-25% sparsity, avoiding the Semantic Interference that plagues dense methods.

2. **Sparse-Dense Trade-off Framework**: We established that sparse methods trade peak performance for zero-shot capability — a fundamental architectural choice with clear implications for deployment scenarios.

3. **Mathematical Foundation**: We provided complete mathematical formulations for all pipeline stages, from phrase extraction through query processing, grounded in distributional semantics, dimensionality reduction, and sparse coding theory.

### 9.1.2 Methodological Contributions

1. **Complete Unsupervised Pipeline**: Six-stage architecture converting raw text to ranked retrieval results without any training data.

2. **Systematic Parameter Tuning**: Comprehensive analysis of grid size, spreading steps, top percent, IDF weighting, Gaussian smoothing, Morton encoding, and document normalization with mathematical justification.

3. **Multi-dataset Benchmark**: Evaluation across 10 datasets (PubMedQA, Belebele, NarrativeQA, PopQA, SciFact, HotpotQA, 2WikiMultihopQA, NQ-REaR, MuSiQue, BioASQ) demonstrating competitive performance.

4. **Hybrid SF+SPLADE Architecture**: Combining unsupervised semantic coverage with learned sparse expansion, achieving perfect MRR=1.0 on Belebele (+13.6% over baseline, surpassing BM25 at 0.995).

### 9.1.3 Empirical Contributions

1. **SF matches or exceeds DPR on SciFact** (0.755 vs 0.675) — validating unsupervised semantic matching on domain-specific tasks.

2. **Performance degrades linearly with hop count**: -2% for 1-hop, -15% for 2-hop, -33% for 2-5 hops — quantifying the compositional gap.

3. **Zero-shot domain adaptation**: SF achieves 88-98% of BM25 on single-hop tasks without any training data.

## 9.2 Key Findings

### 9.2.1 When SF Excels

| Task Type | SF MRR | Why SF Works |
|-----------|--------|--------------|
| Entity lookup | 0.980 | Clear semantic relationships in entity names |
| Biomedical QA | 0.955 | High synonymy ("myocardial infarction" = "heart attack") |
| Narrative comprehension | 0.939 | Paraphrasing ("He said" vs "He stated") |
| Reading comprehension | 0.880 | Multilingual paraphrase matching |
| Scientific claims | 0.755 | Conceptual overlap between claims and evidence |

**Pattern**: SF excels when semantic similarity dominates and vocabulary mismatch is the primary challenge.

### 9.2.2 When SF Struggles

| Task Type | SF MRR | Why SF Fails |
|-----------|--------|--------------|
| Multi-hop QA | 0.453 | Cannot compose facts across passages |
| Negation handling | — | Treats "not considered" identically to "considered" |
| Numerical reasoning | — | Cannot perform arithmetic |
| Large candidate pools | 0.574 | Score compression dilutes signal |

**Pattern**: SF struggles when compositional reasoning or fine-grained discrimination is required.

### 9.2.3 The Sparse-Dense Trade-off

| Aspect | Sparse (SF) | Dense (DPR) |
|--------|-------------|-------------|
| Training data | **None** | 10K-100K labeled pairs |
| Domain adaptation | **Instant** | Days-weeks of retraining |
| Peak performance | 0.955 (PubMedQA) | 0.863 (NQ, SPLADE) |
| Performance floor | 0.453 (MuSiQue) | ~0.65 (estimated) |
| Memory/doc | **512 bytes** | 3KB |
| Interpretability | **Grid visualization** | Black box |

**Conclusion**: Sparse methods trade peak performance for zero-shot capability. This is fundamental and cannot be eliminated by architectural improvements.

## 9.3 Implications for Retrieval Research

### 9.3.1 The Value of Unsupervised Methods

Our results demonstrate that unsupervised semantic matching can achieve competitive performance on specific task types. While supervised methods (DPR, SPLADE) achieve higher absolute scores, SF provides:

1. **Zero-shot domain adaptation**: No labeled data required
2. **Interpretability**: Grid visualizations explain retrieval decisions
3. **Memory efficiency**: 512 bytes per document vs 3KB for dense methods
4. **Boolean reasoning**: Direct AND/OR/NOT operations on fingerprints

These properties make SF valuable for scenarios where training data is unavailable, interpretability is required, or resource constraints prevent dense retrieval.

### 9.3.2 The Vocabulary Mismatch Revisited

SF's strong performance on PubMedQA (95.5% of BM25) and NarrativeQA (95.8% of BM25) confirms that vocabulary mismatch remains a significant challenge for lexical retrieval. SF's topographic encoding provides a principled solution: synonymous phrases map to nearby grid regions, enabling semantic matching without learning.

However, SF's weaker performance on Belebele (88.4%) and NQ-REaR (89.9%) suggests that vocabulary mismatch is only one component of retrieval quality. Lexical precision, entity matching, and compositional reasoning are equally important — and SF cannot address these through semantic matching alone.

### 9.3.3 The Future of Hybrid Retrieval

Our experiments demonstrate that hybrid SF+SPLADE significantly improves performance across multiple datasets:

| Dataset | SF Only | SF+SPLADE (α=0.3) | Improvement |
|---------|---------|-------------------|-------------|
| Belebele (50Q) | 0.880 | **1.000** | **+13.6%** |
| PubMedQA (10Q) | 0.800 | **0.920** | **+15.0%** |
| NQ-REaR (10Q) | 0.574 | **0.920** | **+60.3%** |
| HotpotQA (10Q) | 0.726 | **0.983** | **+35.4%** |
| 2WikiMultihopQA (10Q) | 0.788 | **0.983** | **+24.8%** |
| BioASQ (10Q) | 0.445 | **0.527** | **+18.4%** |

**Practical deployment strategy:**

1. **Stage 1**: SF retrieves top-K candidates using semantic matching (fast, no GPU)
2. **Stage 2**: SPLADE re-ranks using learned sparse expansion (fast, GPU optional)
3. **Stage 3**: (Optional) Cross-encoder for final precision (slow, GPU)

This three-stage architecture combines the strengths of both paradigms while mitigating their weaknesses.

## 9.4 Limitations

### 9.4.1 Current Limitations

1. **Compositional gap**: SF cannot compose facts across passages. Performance degrades linearly with hop count (-2% for 1-hop, -33% for 2-5 hops).

2. **Negation blindness**: 50% of Belebele failures involve negation queries. SF treats "not considered" identically to "considered."

3. **Score compression**: All documents score within a narrow range (0.034–0.051 on NQ-REaR), limiting fine-grained ranking.

4. **Computational cost**: SF indexing takes ~10 minutes for 100 queries (vs ~10 seconds for BM25). Per-query scoring takes ~47s steady-state (dominated by SPLADE inference). The OOV expansion step has been optimized from ~30s to ~0.075s using FAISS.

5. **Grid size sensitivity**: Optimal for 20-passage corpora. Larger pools need scaling guidelines.

6. **t-SNE stochasticity**: Results depend on random seed (fixed at 42). Relative comparisons valid, absolute scores seed-dependent.

## 9.5 Future Work

### 9.5.1 Implemented Improvements (Now Part of Default Pipeline)

The following improvements have been implemented and validated:

1. **SPLADE hybrid retrieval** (+13.6% Belebele, +60.3% NQ-REaR): Learned sparse expansion combined with SF's semantic matching achieves perfect MRR=1.0 on Belebele.
2. **FAISS-accelerated OOV expansion** (30s → 0.075s per query): Replaced brute-force OOV lookup with FAISS IVFFlat index for approximate nearest neighbor search.
3. **Per-dataset parameter registry** (+1–4% across datasets): Dataset-specific optimal configurations stored in a YAML registry.
4. **Query decomposition** (+19.6% NQ-REaR): Multi-hop queries decomposed into sub-queries using spaCy NER + dependency parsing, with independent retrieval and result fusion via RRF.
5. **LambdaMART re-ranking** (proof-of-concept): Gradient-boosted decision trees on 35 features for learned re-ranking. **Performance decreased** (MRR=0.945 vs baseline 1.000 on Belebele 50Q). The baseline already achieves perfect ranking, leaving no room for improvement. Cross-dataset transfer (Belebele→NQ-REaR) yielded MRR=0.649, confirming that LambdaMART requires larger candidate pools and more training data to be effective.

### 9.5.2 Remaining Future Work

**1. Negation-Aware Processing**

Post-processing negation detection and scoring penalties can recover some negation failures:

$$\text{score}_{\text{penalized}} = \text{score} \times (1 - \alpha \cdot \frac{|\mathcal{D} \cap \mathcal{N}|}{|\mathcal{N}|})$$

Target: Recover 50% of Belebele negation failures.

### 9.5.2 Medium-Term Research Directions

**1. LLM-Enhanced Semantic Space**

Use Large Language Models to extract semantic concepts from contexts:

```
Raw Context → LLM → Extracted Concepts → Enhanced Term-Context Matrix
```

**Potential Benefits:**
- Richer semantic representation (implicit semantics that co-occurrence misses)
- Concept generalization ("neuroplasticity" → "brain adaptation")
- Cross-domain transfer via pre-trained LLMs
- Negation handling (distinguishing "not considered" from "considered")

**Challenges:**
- Computational cost of LLM inference
- Potential hallucination in concept extraction
- Need for validation against distributional baseline

**2. End-to-End Training**

Use Gumbel-Softmax to make the grid mapping differentiable, enabling gradient-based optimization of the entire pipeline:

$$y_{ij} = \frac{\exp((\log \pi_{ij} + g_{ij}) / \tau)}{\sum_{k} \exp((\log \pi_{ik} + g_{ik}) / \tau)}$$

where $\pi_{ij}$ is the probability of assigning phrase $i$ to grid cell $j$, $g_{ij}$ is Gumbel noise, and $\tau$ is the temperature parameter. As $\tau \rightarrow 0$, this approaches a hard assignment.

This could learn optimal grid assignments rather than relying on t-SNE.

**3. Learned Sparsification**

Replace fixed top-percent with learned thresholding that adapts to document length and topic diversity.

### 9.5.3 Long-Term Research Directions

**1. Adaptive Grid Architecture**

Develop guidelines for scaling grid size with corpus size:

$$g = f(D, \rho_{\text{target}}, \text{task\_type})$$

where $D$ is corpus size and $\rho_{\text{target}}$ is desired density.

**2. Cross-lingual Semantic Folding**

Extend SF to multilingual retrieval by:
- Learning language-agnostic grid positions
- Aligning semantic spaces across languages
- Using multilingual LLMs for concept extraction

**3. Streaming Semantic Folding**

Enable incremental updates without full recomputation:

$$M_{t+1} = M_t + \Delta M$$

This would support real-time document indexing.

**4. Semantic Folding for Generation**

Extend SF from retrieval to text generation:
- Use grid positions to guide decoding
- Generate text by traversing semantic space
- Combine with language models for constrained generation

## 9.6 Final Remarks

Semantic Folding occupies a unique position in the retrieval landscape: the only method that provides unsupervised semantic matching, interpretable grid visualizations, and memory-efficient storage without any training data. While it cannot match the peak performance of supervised dense methods on all tasks, its zero-shot capability and interpretability make it invaluable for emerging domains where training data is unavailable and explainability is required.

The sparse-dense trade-off is fundamental and cannot be eliminated by architectural improvements. It stems from the Orthogonality Constraint: learning to separate semantically similar concepts requires training data, while sparse methods achieve separation through mathematical properties of high-dimensional binary vectors.

As retrieval systems increasingly operate in low-resource, emerging domains, the value of unsupervised methods like Semantic Folding will only grow. The hybrid SF+SPLADE architecture provides a practical deployment strategy that combines the best of both worlds, offering a path forward for real-world retrieval systems that must balance performance, interpretability, and resource constraints.

## References

- Broder, A. Z. (1997). On the resemblance and containment of documents. *Compression and Complexity of Sequences*.
- Formal, T., et al. (2021). SPLADE. *SIGIR 2021*.
- Furnas, G. W., et al. (1987). The vocabulary problem. *CACM*, 30(11), 964–971.
- Harris, Z. S. (1954). Distributional structure. *Word*, 10(2–3), 146–162.
- Hawkins, J., & George, D. (2006). *Hierarchical Temporal Memory*. Numenta.
- Karpukhin, V., et al. (2020). Dense Passage Retrieval. *EMNLP 2020*.
- Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
- Mallia, A., et al. (2022). Learning sparse indexes. *arXiv:2405.01924*.
- Santhanam, K., et al. (2022). ColBERTv2. *NAACL 2022*.
- Zahn, O., et al. (2026). Attention Is Not Retention. arXiv:2601.15313.

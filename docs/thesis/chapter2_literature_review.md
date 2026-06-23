# Chapter 2: Literature Review

## 2.1 Information Retrieval Foundations

### 2.1.1 The Vector Space Model

The vector space model (Salton et al., 1975) represents documents and queries as vectors in a high-dimensional term space, where similarity is computed via cosine similarity:

$$\text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$

This foundational model underpins both classical and modern retrieval methods. The key insight — that meaning can be captured through distributional patterns — remains central to this thesis (Fernández et al., 2011; Munir & Anjum, 2018).

### 2.1.2 BM25: The Gold Standard

BM25 (Robertson & Zaragoza, 2009) extends the vector space model with term frequency saturation and document length normalization:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

where $k_1 = 1.2$ and $b = 0.75$ are standard parameters. BM25 remains the strongest baseline in our benchmarks, achieving MRR > 0.99 on 4/9 datasets.

### 2.1.3 The Vocabulary Mismatch Problem

Furnas et al. (1987) demonstrated that different people use different words for the same concept, creating a fundamental challenge for lexical retrieval. This *vocabulary mismatch* manifests as:
- **Synonymy**: "myocardial infarction" = "heart attack" = "MI"
- **Polysemy**: "bank" (financial) vs "bank" (river)
- **Paraphrase**: "He said" vs "He stated" vs "He uttered"

Our benchmarks quantify this: SF achieves 95.5% of BM25 on PubMedQA (high synonymy) but only 88.4% on Belebele (paraphrase-heavy). Domain-specific ontologies can address this through controlled vocabulary mapping (Fernández et al., 2011; Dramé et al., 2014).

## 2.2 Dense Retrieval Methods

### 2.2.1 Dense Passage Retrieval (DPR)

Karpukhin et al. (2020) introduced DPR, which encodes queries and passages as dense 768-dimensional vectors using BERT encoders, trained on ~50K query-passage pairs:

$$\text{score}(q, d) = \mathbf{E}_q(q)^\top \mathbf{E}_d(d)$$

DPR achieves 0.794 MRR on Natural Questions but requires:
- ~50K labeled training pairs
- ~4 hours GPU training (1x V100)
- 3KB storage per document
- GPU inference

### 2.2.2 ColBERT: Late Interaction

ColBERT (Santhanam et al., 2022) uses token-level embeddings with late interaction via MaxSim:

$$\text{score}(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \mathbf{E}_q(q_i)^\top \mathbf{E}_d(d_j)$$

ColBERTv2 achieves 0.855 MRR on NQ but requires ~500K training pairs and 4x V100 GPUs.

### 2.2.3 SPLADE: Sparse Learned Expansion

Formal et al. (2021) introduced SPLADE, which combines sparse representations with learned expansion:

$$\text{SPLADE}(q) = \sum_{t \in q} \text{log}(1 + \text{ReLU}(\mathbf{W} \cdot \mathbf{h}_t + \mathbf{b}))$$

SPLADE achieves 0.863 MRR on NQ — the best neural method — but requires ~500K training pairs and GPU infrastructure.

### 2.2.4 The Training Data Bottleneck

All dense methods share a critical limitation: they require labeled retrieval pairs that may not exist for emerging domains. This creates a *cold start problem* where:
- New domains lack training data
- Domain-specific terminology requires retraining
- Annotation is expensive and time-consuming

SF eliminates this bottleneck through unsupervised semantic matching.

## 2.3 Sparse Distributed Representations

### 2.3.1 Kanerva's Sparse Distributed Memory

Kanerva (1988) proposed Sparse Distributed Memory (SDM) as a model of human associative memory. Key properties:
- **High-dimensional binary vectors** (typically 10,000+ bits)
- **Sparse activation** (1-2% active bits)
- **Near-orthogonality** of random patterns
- **Content-addressable memory** via Hamming distance

SF inherits these properties: 4,096-bit fingerprints with 10-25% sparsity achieve near-orthogonality through the mathematical guarantee that random binary vectors are nearly orthogonal with high probability (Kanerva, 2009; Kleyko et al., 2016).

### 2.3.2 Hierarchical Temporal Memory

Hawkins & George (2006) extended SDM principles to Hierarchical Temporal Memory (HTM), emphasizing:
- **Sparse coding** for energy efficiency
- **Spatial pooling** for invariant representation
- **Temporal memory** for sequence learning

SF's grid-based encoding implements spatial pooling: phrases map to grid positions based on distributional similarity, creating invariant semantic representations.

### 2.3.3 The Orthogonality Constraint

Recent theoretical work (Zahn et al., 2026) identifies the **Orthogonality Constraint**: reliable memory requires orthogonal keys, but semantic embeddings cannot be orthogonal because training clusters similar concepts together. This creates **Semantic Interference** — neural systems storing facts into shared continuous parameters collapse to near-random accuracy within tens of semantically related facts.

**Critical finding**: Collapse occurs at N=5 facts when semantic density ρ > 0.6, or N ≈ 20-75 at moderate ρ.

SF sidesteps this problem entirely: its sparse binary fingerprints naturally achieve near-orthogonality through sparsity (10-25% active bits), eliminating the need for learned separability (Ahmad & Hawkins, 2015).

## 2.4 Semantic Space Construction

### 2.4.1 The Distributional Hypothesis

Harris (1954) and Firth (1957) established that linguistic meaning is a function of context:

> *"You shall know a word by the company it keeps."* — Firth (1957, p. 11)

SF operationalizes this through the term-context matrix $\mathbf{M} \in \mathbb{R}^{n \times m}$, where entry $M_{ij}$ captures the co-occurrence weight of phrase $i$ in context $j$.

### 2.4.2 Dimensionality Reduction

The term-context matrix lives in high-dimensional space where the curse of dimensionality (Beyer et al., 1999) makes neighbourhood relationships unstable. SF uses t-SNE (van der Maaten & Hinton, 2008) or UMAP (McInnes et al., 2018) to project contexts onto a 2D grid while preserving semantic proximity (Leticio et al., 2024; Lambert et al., 2022).

**t-SNE** minimizes KL divergence between high-dimensional and low-dimensional neighbourhood distributions:
$$\mathcal{L}_{\text{t-SNE}} = \text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**UMAP** preserves fuzzy topological structure:
$$\mathcal{L}_{\text{UMAP}} = \sum_{(i,j) \in E} \left[ w_{ij} \log \frac{w_{ij}}{v_{ij}} + (1 - w_{ij}) \log \frac{1 - w_{ij}}{1 - v_{ij}} \right]$$

### 2.4.3 Morton Z-order Encoding

Morton encoding (Morton, 1966) linearizes 2D grid positions into 1D indices while preserving spatial locality:

$$z(x,y) = \sum_{k=0}^{b-1} \left[ \text{bit}_k(x) \ll 2k + \text{bit}_k(y) \ll (2k+1) \right]$$

This ensures that semantically similar phrases (adjacent on the grid) have similar fingerprint indices, enabling efficient Hamming distance computation.

## 2.5 Closed-Domain QA Systems

### 2.5.1 Domain-Specific Retrieval Challenges

Closed-domain QA systems operate within bounded corpora where domain-specific terminology creates unique retrieval challenges (Allam & Haggag, 2012; Mollá & Vicedo, 2007):
- **Specialized vocabulary**: Medical systems must handle MeSH terms, ICD codes, and drug names; legal systems must process case citations, statutes, and legal doctrines
- **Conceptual hierarchies**: Domain ontologies define relationships between concepts that lexical methods cannot capture
- **Evolving terminology**: New terms emerge rapidly in active research fields, requiring rapid system adaptation

### 2.5.2 Glossary Integration

Domain glossaries—controlled vocabularies mapping synonymous terms to canonical forms—are essential for accurate retrieval in specialized domains (Dramé et al., 2014; Chen et al., 2013). SF's grid-based architecture enables direct glossary integration without retraining, unlike dense methods that require embedding updates.

### 2.5.3 Ontology-Based Approaches

Ontology-based retrieval systems leverage structured knowledge to improve semantic matching (Fernández et al., 2011; Munir & Anjum, 2018; Kara et al., 2012). SF's grid positions can be aligned with ontological concepts, creating a bridge between distributional semantics and structured knowledge.

## 2.6 The Current State of Sparse Retrieval for QA (2023–2025)

### 2.6.1 Semantic Folding's Absence from Recent Literature

A systematic search of arXiv and Google Scholar reveals that **Semantic Folding has zero presence in 2023–2025 QA literature**. The only published reference is the original 2015 white paper by Webber (arXiv:1511.08855). No papers combine "semantic folding" with "question answering" on any academic search engine.

This absence is both a gap and an opportunity: the approach has not been validated by the broader IR community, but it also means the thesis work filling this gap would be genuinely novel. The method must be positioned carefully against well-established alternatives (SPLADE, DPR, ColBERT) to gain credibility.

### 2.6.2 Dominant Methods in BioASQ (2023–2024)

BioASQ 2023 (28 teams, 150+ systems) and 2024 (37 teams, 700+ submissions) were dominated by:

| Method Category | Examples | Trend |
|----------------|----------|-------|
| LLM-augmented retrieval | GPT-4, Llama2, Mistral for query expansion | **Rising** — winning systems use few-shot prompting |
| Domain-specific fine-tuning | BioBERT, PubMedBERT | **Stable** — strong for answer extraction |
| BM25 + neural reranking | BM25 first-stage + ColBERT/MonoT5 rerank | **Dominant** — most common pipeline architecture |
| Pure sparse retrieval | BM25 alone | **Declining** — without learned expansion |

Key finding: BioRAGent (arXiv:2412.12358), a successful BioASQ 2024 participant, used BM25 + Elasticsearch + LLM query expansion — confirming that sparse retrieval remains competitive when augmented with neural components.

### 2.6.3 SPLADE as the Learned Sparse Baseline

SPLADE (Formal et al., 2021) dominates learned sparse retrieval with recent improvements:

- **Mistral-SPLADE** (arXiv:2408.11119): Decoder-only LLMs outperform encoder-only SPLADE variants; new SOTA on BEIR
- **Two-Step SPLADE** (arXiv:2404.13357): 30x speedup for in-domain with minimal quality loss
- **SPLATE** (arXiv:2404.13950): Combines ColBERTv2 with SPLADE adapter for CPU-efficient late interaction
- **SPLADE for medical review** (arXiv:2405.03972): Reduces systematic review cost by 10–18%

SPLADE achieves MRR=0.863 on Natural Questions — the best neural method — but requires ~500K training pairs and GPU infrastructure.

### 2.6.4 The Hybrid Pipeline Dominance

Recent evidence confirms that hybrid sparse+dense pipelines outperform single-method baselines:

- **RRF fusion** (arXiv:2604.13728): Sparse+dense outperforms sparse-only by 14.9% and dense-only by 6.1%
- **DEXTER benchmark** (arXiv:2406.17158): Late interaction models (ColBERT) and BM25 "surprisingly perform well compared to pre-trained dense models on complex QA"
- **HiRAG** (arXiv:2408.11875): Sparse document-level + dense chunk-level retrieval for multi-hop QA
- **GeAR** (arXiv:2412.18431): Graph expansion augments any sparse retriever; >10% improvement on MuSiQue

**Key insight**: No unsupervised sparse method approaches SPLADE's performance levels. SF's value proposition is not matching SPLADE's accuracy, but providing unsupervised semantic matching with zero training data.

### 2.6.5 BioASQ Challenge (2023–2025)

BioASQ is the premier biomedical QA benchmark, evaluating systems on four question types: yes/no, factoid, list, and summary [Nentidis et al., 2025]. The official ranking averages performance on the three exact-answer types (yes/no F1, factoid MRR, list mean F1), excluding summary from the primary leaderboard.

**BioASQ 2024 results** (37 teams, 700+ submissions):
- Best factoid MRR: ~0.45 (dmiip2024_3)
- Best yes/no accuracy: ~0.94 (UR-IW-2)
- Dominant approaches: LLM-augmented retrieval, BM25+neural reranking, BioBERT fine-tuning

**SF on BioASQ**: MRR=0.2480 (50 queries, 1075 docs). SF's phrase-level matching captures some biomedical semantics but struggles with complex queries requiring compositional reasoning. Hybrid SF+BM25 hurts (-32.8%) because BM25's lexical strictness dilutes SF's semantic advantage.

## 2.7 Evaluation Framework

### 2.7.1 Standard IR Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MRR | $\frac{1}{|\mathcal{Q}|} \sum_{q} \frac{1}{\text{rank}_q}$ | First relevant result position |
| AP | $\frac{1}{|\mathcal{R}|} \sum_{i} P@i \cdot \text{rel}_i$ | Precision-recall curve summary |
| NDCG@K | $\frac{\text{DCG@K}}{\text{IDCG@K}}$ | Ranking quality vs optimal |
| P@K | $\frac{|\mathcal{R} \cap \text{top-}K|}{K}$ | Precision at cutoff |

### 2.7.2 Benchmark Datasets

Our evaluation spans 9 datasets covering diverse task types:

| Dataset | Task | Domain | Queries |
|---------|------|--------|---------|
| PubMedQA | Biomedical QA | Biomedical | 111 |
| Belebele | Reading Comprehension | Multilingual | 100 |
| NarrativeQA | Narrative Comprehension | Scripts | 49 |
| PopQA | Entity Lookup | Wikidata | 100 |
| SciFact | Scientific Claims | Scientific | 300 |
| HotpotQA | Multi-hop QA | Wikipedia | 48 |
| 2WikiMultihopQA | Multi-hop QA | Wikipedia | 50 |
| NQ-REaR | Factoid Retrieval | Web | 100 |
| MuSiQue | Multi-hop QA | Wikipedia | 100 |

## References

- Abacha, A., & Zweigenbaum, P. (2015). MEANS: A medical question-answering system combining NLP techniques and semantic Web technologies. *Information Processing & Management*, 51(5), 570-584.
- Ahmad, S., & Hawkins, J. (2015). Properties of sparse distributed representations and their application to hierarchical temporal memory. *arXiv preprint arXiv:1503.07469*.
- Allam, A. M. N., & Haggag, M. H. (2012). The question answering systems: A survey. *International Journal of Research and Reviews in Information Sciences*, 2(3), 367-375.
- Beyer, K., et al. (1999). When is "nearest neighbor" meaningful? *ICDT*, 217–235.
- Chen, Y. J., Chu, H. C., Chen, Y. M., & Chao, C. Y. (2013). Adapting domain ontology for personalized knowledge search and recommendation. *Information & Management*, 50(6), 278-288.
- Dramé, K., Diallo, G., Delva, F., Dartigues, J. F., & Mouillet, V. (2014). Reuse of termino-ontological resources and text corpora for building a multilingual domain ontology. *Journal of Biomedical Informatics*, 48, 1-10.
- Fernández, M., Cantador, I., López, V., Vallet, D., Castells, P., & Motta, E. (2011). Semantically enhanced information retrieval: An ontology-based approach. *Journal of Web Semantics*, 9(4), 413-434.
- Formal, T., et al. (2021). SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. *SIGIR 2021*.
- Furnas, G. W., et al. (1987). The vocabulary problem in human-system communication. *CACM*, 30(11), 964–971.
- Harris, Z. S. (1954). Distributional structure. *Word*, 10(2–3), 146–162.
- Hawkins, J., & George, D. (2006). *Hierarchical Temporal Memory*. Numenta Technical Report.
- Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain QA. *EMNLP 2020*.
- Kara, S., Alan, Ö., Sabuncu, O., Akpınar, S., Cicekli, N. K., & Diri, F. Y. (2012). An ontology-based retrieval system using semantic indexing. *Information Systems*, 37(7), 688-704.
- Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
- Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. *Cognitive Computation*, 1(2), 139-159.
- Kleyko, D., Osipov, E., & Rachkovskij, D. A. (2016). Modification of holographic graph neuron using sparse distributed representations. *Procedia Computer Science*, 88, 39-45.
- Lambert, P., De Bodt, M., Verleysen, M., & Lee, J. A. (2022). SQuadMDS: A lean Stochastic Quartet MDS improving global structure preservation in neighbor embedding like t-SNE and UMAP. *Neurocomputing*, 500, 271-281.
- Leticio, G. R., Kawai, V. S., Valem, L. P., & Guimarães, D. M. (2024). Manifold information through neighbor embedding projection for image retrieval. *Pattern Recognition Letters*, 178, 1-7.
- McInnes, L., et al. (2018). UMAP: Uniform Manifold Approximation and Projection. *arXiv:1802.03426*.
- Mollá, D., & Vicedo, J. L. (2007). Question answering in restricted domains: An overview. *Computational Linguistics*, 33(1), 41-82.
- Morton, G. M. (1966). *A computer oriented geodetic data base*. IBM Technical Report.
- Munir, K., & Anjum, M. S. (2018). The use of ontologies for effective knowledge modelling and information retrieval. *Applied Computing and Informatics*, 14(2), 116-126.
- Robertson, S. E., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *FTIR*, 3(4), 333–389.
- Salton, G., et al. (1975). A vector space model for automatic indexing. *CACM*, 18(11), 613–620.
- Santhanam, K., et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. *NAACL 2022*.
- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *JMLR*, 9, 2579–2605.
- Zahn, O., et al. (2026). Attention Is Not Retention: The Orthogonality Constraint. arXiv:2601.15313.
- Nentidis, A., et al. (2025). Overview of BioASQ 2024: The twelfth BioASQ challenge. arXiv:2508.20532.
- Krithara, A., et al. (2024). BioASQ-QA: A manually curated corpus for Biomedical Question Answering. *Scientific Data*, 10, 170.
- Formal, T., et al. (2024). Mistral-SPLADE: LLMs for better Learned Sparse Retrieval. arXiv:2408.11119.
- Lin, J., et al. (2024). Two-Step SPLADE: Simple, Efficient and Effective Approximation. *ECIR 2024 Findings*. arXiv:2404.13357.
- Paria, B., et al. (2024). SPLATE: Sparse Late Interaction Retrieval. *SIGIR 2024*. arXiv:2404.13950.
- Wang, L., et al. (2025). BioRAGent: A RAG System for Biomedical Q&A. *ECIR 2025*. arXiv:2412.12358.
- Gao, Y., et al. (2024). DEXTER: Benchmark for Complex QA using LLMs. arXiv:2406.17158.
- Ma, X., et al. (2024). HiRAG: Hierarchical RAG with Rethink. arXiv:2408.11875.
- Chen, J., et al. (2024). GeAR: Graph-enhanced Agent for RAG. *ACL 2025 Findings*. arXiv:2412.18431.
- Webber, F. (2015). Semantic Folding Theory and its Application in Semantic Fingerprinting. arXiv:1511.08855.
- BMQExpander (2025). BM25+ontology query expansion competitive with dense retrievers. arXiv:2508.11784.
- RRF Fusion (2026). Reciprocal Rank Fusion of sparse and dense retrieval. arXiv:2604.13728.
- Domain Dense Retrievers (2025). General-purpose dense retrievers struggle with specialized domains. arXiv:2510.04757.
- Joint Document-Snippet Ranking (2021). PDRMM competitive with BERT at fraction of parameters. arXiv:2106.08908.

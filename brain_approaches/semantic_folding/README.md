# Semantic Folding: Knowledge Graph to Semantic Space Embeddings

## Overview

**Status**: 🔄 Pipeline Implementation (Phase 1-5 Complete) | Ready for Evaluation Phases

Semantic Folding is a novel approach for constructing semantic space embeddings (fingerprints) from knowledge graphs and textual corpora. Unlike traditional dimensionality reduction techniques (UMAP, t-SNE, PCA), semantic folding creates interpretable, grid-based semantic spaces where:

- **Phrases** are represented as spatial fingerprints showing their semantic distribution
- **Documents** are represented as composite fingerprints combining their constituent phrases
- **Semantic relationships** are preserved through graph-based positioning and spatial proximity

## Modern Implementation

### Production-Ready Pipeline
The semantic folding approach has been modernized into a production-ready pipeline with comprehensive engineering:

#### Core Components (✅ Completed)
- **`scratchpad.py`**: Main orchestration script with comprehensive logging
- **`phrase_extractor.py`**: Modern phrase extraction with 1-4 word limits
- **`term_context.py`**: Sparse matrix term-context construction (95% memory savings)
- **`semantic_space.py`**: Force-directed graph layout with configurable grids
- **`phrase_fingerprints.py`**: Efficient fingerprint generation from semantic coordinates
- **`doc_fingerprints.py`**: Document fingerprint aggregation with metadata

#### Key Improvements
- **Scalability**: Handles corpora from 1K to 50K+ documents
- **Memory Efficiency**: Sparse matrices reduce memory usage by 95%
- **Robustness**: Fallback mechanisms for missing dependencies
- **Optimized Quality**: Tuned parameters for optimal semantic space distribution (32×32 grid, controlled connectivity)
- **TF-IDF Matrix Normalization**: Reduces high-frequency word dominance in semantic relationships
- **Document Fingerprint Sparsification**: Keeps only top 5% of activated cells for improved semantic focus
- **Advanced Phrase Processing**: 1-4 word phrase limits with quality filtering
- **Interactive TUI**: Command-line interface with progress tracking and error checking
- **Resume Capability**: Automatic recovery from interruptions
- **Comprehensive Logging**: Dual console/file output with color coding and debug modes

## Advanced Features

### TF-IDF Matrix Normalization

Reduces the dominance of high-frequency words in semantic relationships:

- **Problem**: Common words like "it", "that", "the" can overwhelm meaningful semantic connections
- **Solution**: Applies TF-IDF weighting to term-context matrix entries
- **Formula**: `TF-IDF = TF × log(N/DF)` where TF is term frequency, DF is document frequency
- **Benefit**: Balances semantic relationships by down-weighting ubiquitous terms
- **Configuration**: `normalize_matrix: true` (enabled by default)

### Document Fingerprint Thresholding

Creates more focused and interpretable document representations:

- **Problem**: Document fingerprints may contain too much noise from weakly relevant terms
- **Solution**: Retains only the top N% most activated cells in document fingerprints
- **Method**: After aggregating phrase fingerprints, keeps only cells above a threshold
- **Benefit**: Improves semantic specificity and reduces storage requirements
- **Configuration**: `doc_top_percent: 0.05` (keeps top 5%, configurable)

### Optimized Configuration

The pipeline uses carefully tuned parameters for optimal quality:

```yaml
# Core semantic parameters
grid_size: 32                    # Larger grid for better distribution
max_edges: 200                  # Controlled connectivity
edge_threshold: 0.05           # Balanced similarity threshold

# Quality enhancements
normalize_matrix: true         # TF-IDF normalization
doc_top_percent: 0.05         # Document fingerprint thresholding

# Performance settings
batch_size: 1000               # Efficient processing
use_spacy: true               # Advanced phrase extraction
```

### Quick Start

#### Interactive TUI (Recommended)
```bash
# Install dependencies
uv sync
uv add pyyaml questionary

# Launch interactive interface
uv run python brain_approaches/semantic_folding/semantic_folder.py
```

#### Command Line (Advanced)
```bash
# Run on MuSiQue corpus (default)
uv run python brain_approaches/semantic_folding/scratchpad.py \
  --corpus_path data/HippoRAG2/dataset/musique_corpus.json

# Run on custom corpus
uv run python brain_approaches/semantic_folding/scratchpad.py \
  --corpus_path /path/to/your/corpus.json \
  --grid_size 8

# Check results
ls outputs/$(date +%Y%m%d)_*/fingerprints/ | wc -l  # 25K+ fingerprint files
```

## Configuration

The semantic folding pipeline is highly configurable through `config/semantic_folding.yml`:

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_size` | 32 | Semantic space grid size (8, 16, 32) |
| `max_edges` | 200 | Maximum edges in semantic graph |
| `edge_threshold` | 0.05 | Minimum similarity for graph connections |
| `normalize_matrix` | true | Apply TF-IDF normalization |
| `doc_top_percent` | 0.05 | Document fingerprint threshold percentage |

### Quality vs Speed Trade-offs

**Maximum Quality** (Recommended):
```yaml
grid_size: 32
max_edges: 500
edge_threshold: 0.01
normalize_matrix: true
doc_top_percent: 0.03
```

**Balanced Performance**:
```yaml
grid_size: 32
max_edges: 200
edge_threshold: 0.05
normalize_matrix: true
doc_top_percent: 0.05
```

**Fast Processing**:
```yaml
grid_size: 16
max_edges: 100
edge_threshold: 0.1
normalize_matrix: false
doc_top_percent: 0.1
```

### Output Structure
```
outputs/YYYYMMDD_HHMMSS/
├── corpus.txt                 # Processed corpus
├── phrases.txt               # 25K+ filtered phrases
├── term_context_matrix.npz   # Sparse matrix (827K entries, 0.28% density)
├── context_coordinates.csv   # 16×16 grid coordinates
├── fingerprints/             # 25K+ phrase fingerprint files
├── doc_fingerprints/         # 11K+ document fingerprints + metadata
├── logs/pipeline.log         # Comprehensive execution log
└── visualizations/           # Heatmaps and graphs (if matplotlib available)
```

## Experimental Framework

The implementation includes comprehensive notebooks demonstrating the algorithm and comparative studies:

### Core Algorithm Notebooks
- **`Semantic_Space_Construction.ipynb`**: Complete end-to-end pipeline with Google Colab integration
- **`ground-up.ipynb`**: Foundational phrase extraction and processing
- **`pre_processing.ipynb`**: Text preprocessing with spaCy lemmatization and cleaning

### Comparative Methods
- **`Document_representation.ipynb`**: Traditional TF-IDF + SVD approach for baseline comparison
- **`Document_clustering.ipynb`**: Self-Organizing Maps (SOM) clustering on TF-IDF vectors
- **`Umap_fingerprint.ipynb`**: UMAP-based dimensionality reduction with grid fingerprinting
- **`tsne_fingerprint.ipynb`**: t-SNE visualization and fingerprint generation
- **`ISOMAP.ipynb`**: ISOMAP manifold learning comparison
- **`SOM_fingerprint.ipynb`**: SOM-based semantic fingerprinting
- **`UMAP_2.ipynb`, `UMAP_3.ipynb`**: Extended UMAP experiments

## Core Algorithm

### Pipeline Overview

The modern semantic folding pipeline consists of 8 phases with comprehensive evaluation:

#### Core Pipeline (✅ Implemented)
1. **Corpus Loading** (`scratchpad.py`): Load and preprocess JSON/CSV corpora
2. **Phrase Extraction** (`phrase_extractor.py`): Extract 1-4 word phrases with quality filtering
3. **Term-Context Matrix** (`term_context.py`): Sparse matrix construction (95% memory savings)
4. **Semantic Space Construction** (`semantic_space.py`): 16×16 grid positioning via force-directed layout
5. **Fingerprint Generation** (`phrase_fingerprints.py`, `doc_fingerprints.py`): Create phrase and document fingerprints

#### Evaluation Pipeline (🔄 Next)
6. **LanceDB Integration**: Fast vector similarity search for semantic retrieval
7. **Multi-Method Evaluation**: Compare against TF-IDF, BM25, Dense, Graph-based baselines
8. **Comprehensive Benchmarking**: Recall@K, MRR, MAP, EM, F1 metrics with statistical testing

#### Legacy Components
- **Experimental Notebooks**: Original research implementations (see Experimental Framework below)

### Step-by-Step Technical Details

#### 1. Phrase Extraction
**Input**: Text corpus (`corpus.txt`) - format: `context_id,text_content`
**Output**: Frequency-sorted list of phrases (`phrases.txt`)

**Algorithm**:
- Uses spaCy NLP pipeline for linguistic analysis
- Extracts noun phrases (`doc.noun_chunks`)
- Extracts verb phrases (tokens with `dep_ == "VP"`)
- Filters single-word phrases to remove stop words
- Ranks phrases by frequency of occurrence

**Key Parameters**:
- Minimum phrase length: > 1 character
- Stop word filtering for single tokens
- Frequency-based ranking

#### 2. Term-Context Matrix Construction
**Input**: Phrases list, corpus text
**Output**: CSV matrix (`term_context_matrix.csv`)

**Algorithm**:
- Creates sparse matrix: contexts × phrases
- Cell values: count of phrase occurrences in each context
- Uses exact string matching for phrase detection

**Matrix Structure**:
```
Context ID | phrase_1 | phrase_2 | ... | phrase_n
-----------|----------|----------|-----|----------
1          | 2        | 0        | ... | 1
2          | 0        | 3        | ... | 0
...
```

#### 3. Semantic Space Construction
**Input**: Term-context matrix
**Output**: Semantic coordinates (`context_coordinates.csv`), visualizations

**Algorithm**:
1. **Graph Construction**:
   - Nodes: contexts (documents)
   - Edges: weighted by phrase overlap (dot product of context vectors)
   - Weight normalization: `weight / 20` (configurable threshold)

2. **Force-Directed Layout**:
   - Uses NetworkX `spring_layout` with configurable parameters
   - `k = NUM_DIMENSIONS / 2` for optimal spacing
   - `seed=200` for reproducible positioning

3. **Grid Mapping**:
   - Maps continuous 2D coordinates to discrete grid positions
   - Grid size: `NUM_DIMENSIONS × NUM_DIMENSIONS` (default: 10×10)
   - Coordinate transformation: `(row, col) = (y_pos × scale, x_pos × scale)`

4. **Visualization**:
   - Network graph with edge weights
   - Context-context similarity heatmap
   - Interactive Plotly heatmaps with hover information

**Key Parameters**:
- `NUM_DIMENSIONS`: Grid resolution (affects granularity)
- `NUM_CONTEXT`: Number of contexts to process
- Edge weight threshold: 0.1 (normalized)

#### 4. Phrase Fingerprint Generation
**Input**: Context coordinates, term-context matrix
**Output**: Individual phrase fingerprint matrices (`fingerprints/*.txt`)

**Algorithm**:
For each phrase:
1. Find contexts where phrase appears (term_context_matrix[context][phrase] > 0)
2. Map context IDs to semantic coordinates
3. Create `NUM_DIMENSIONS × NUM_DIMENSIONS` matrix
4. Increment cells at semantic positions where phrase occurs

**Matrix Structure**:
```
Phrase: "machine learning"
Fingerprint Matrix (8×8 example):
0 0 0 1 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0
```

#### 5. Document Fingerprint Generation
**Input**: Corpus text, phrase fingerprints
**Output**: Document fingerprint matrices (`doc_fingerprints/*.txt`)

**Algorithm**:
For each document:
1. Extract noun phrases using spaCy
2. Sum fingerprint matrices of all phrases in the document
3. Result: composite semantic fingerprint

**Mathematical Foundation**:
```
doc_fingerprint[i][j] = Σ phrase_fingerprint[k][i][j] for all k in document_phrases
```

#### 6. Visualization Tools
**Single/Multi-Phrase Comparison**: Heatmap visualization of phrase distributions
**Document Comparison**: Side-by-side fingerprint comparison with text overlays

## Key Innovations

### 1. Interpretable Semantic Spaces
- Unlike black-box embeddings (word2vec, BERT), semantic folding produces spatial representations
- Each position in the grid has semantic meaning based on context proximity
- Phrases cluster spatially based on co-occurrence patterns

### 2. Knowledge Graph Integration
- Contexts can represent knowledge graph entities/nodes
- Phrase fingerprints capture relational semantics
- Document fingerprints represent composite entity representations

### 3. Preservation of Semantic Relationships
- Graph-based construction maintains relational structure
- Force-directed layout respects semantic distances
- Grid mapping enables efficient similarity computations

## Applications for Knowledge Graphs

### Entity Embedding Construction
1. **Input**: Knowledge graph triples as textual contexts
2. **Process**: Generate phrase fingerprints for entity mentions
3. **Output**: Spatial embeddings capturing semantic neighborhoods

### Semantic Similarity Search
- Compare entity fingerprints using matrix similarity metrics
- Spatial proximity indicates semantic relatedness
- Supports both exact and fuzzy matching

### Knowledge Graph Completion
- Predict missing relationships based on fingerprint patterns
- Identify semantically similar entities for link prediction
- Support ontology alignment and schema matching

## Experimental Methodology

### Data Preprocessing Pipeline
**Input**: Raw text corpus
**Processing** (see `pre_processing.ipynb`):
1. **Lemmatization**: Convert words to base forms using spaCy
2. **Stop Word Removal**: Filter out common function words
3. **Punctuation Removal**: Clean non-alphabetic tokens
4. **Token Filtering**: Preserve only meaningful content words

**Output**: Cleaned, normalized text corpus ready for semantic analysis

### Comparative Baselines

#### TF-IDF + SVD (`Document_representation.ipynb`)
```python
# Traditional approach for comparison
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
svd = TruncatedSVD(n_components=400)
document_vectors = svd.fit_transform(tfidf_matrix)
```

#### Self-Organizing Maps (`Document_clustering.ipynb`)
```python
# SOM clustering on TF-IDF vectors
som = MiniSom(som_x, som_y, n_components, sigma=0.5, learning_rate=0.5)
som.pca_weights_init(document_vectors)
som.train_random(document_vectors, 1000)
```

#### UMAP Fingerprints (`Umap_fingerprint.ipynb`)
```python
# UMAP-based grid fingerprinting
umap_model = umap.UMAP(n_components=2, random_state=123)
document_vectors_umap = umap_model.fit_transform(document_vectors_reduced)
# Bin into 16x16 grid for fingerprint generation
```

### Evaluation Metrics

#### Semantic Coherence
- **Phrase Clustering Quality**: Related phrases should occupy proximate grid regions
- **Context Preservation**: Semantically similar contexts should have similar coordinate positions
- **Fingerprint Distinctiveness**: Different concepts should produce distinguishable spatial patterns

#### Comparative Performance
- **vs. TF-IDF+SVD**: Semantic folding provides spatial interpretability vs. continuous vectors
- **vs. UMAP/SOM**: Graph-based positioning respects relational structure vs. manifold learning
- **vs. t-SNE**: Preserves local neighborhoods while maintaining global grid structure

## Comparative Analysis with Traditional Methods

### vs. UMAP/t-SNE (See `notebooks/Umap_fingerprint.ipynb`)
**Traditional Approach**:
- Dimensionality reduction on vector embeddings (GloVe, BERT)
- Preserves local/global structure through manifold learning
- Output: Continuous 2D coordinates

**Semantic Folding Advantages**:
- **Interpretability**: Grid positions have semantic meaning
- **Composability**: Document fingerprints from phrase combinations
- **Knowledge Integration**: Direct incorporation of relational information

### vs. Self-Organizing Maps (SOM)
**Similarities**:
- Both create discrete grid representations
- Preserve topological relationships

**Differences**:
- SOM: Unsupervised clustering of vector data
- Semantic Folding: Graph-based positioning with explicit semantic constraints

## Technical Requirements

### Dependencies
```
spacy>=3.0.0
networkx>=2.5
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
plotly>=4.14.0
scikit-learn>=0.24.0 (for comparison notebooks)
umap-learn>=0.5.0 (for comparison notebooks)
```

### Installation
```bash
pip install spacy networkx numpy matplotlib seaborn plotly
python -m spacy download en_core_web_sm
```

### Cloud Deployment (Google Colab)
All notebooks are designed for Google Colab execution:

```python
# Mount Google Drive for data persistence
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install spacy networkx numpy matplotlib seaborn plotly
!python -m spacy download en_core_web_sm
```

### Data Formats
**Input Corpus** (`corpus.txt`):
```
1,The machine learning algorithm processes data efficiently.
2,Neural networks are powerful computational models.
3,Data science combines statistics and programming.
```

**Alternative Format** (`Bopenbook.txt`):
```
1,Context text here
2,Another context document
...
```

### Input Format Requirements
**Corpus File** (`corpus.txt`):
```
1,The machine learning algorithm processes data efficiently.
2,Neural networks are powerful computational models.
3,Data science combines statistics and programming.
...
```

**Expected Output Structure**:
```
semantic_folding/
├── phrases.txt                    # Extracted phrases with frequencies
├── term_context_matrix.csv       # Phrase occurrence matrix
├── context_coordinates.csv       # Semantic space positions
├── fingerprints/                 # Individual phrase fingerprints
├── doc_fingerprints/            # Document-level fingerprints
└── images/                      # Visualization outputs
```

## Performance Characteristics

### Computational Complexity
- **Phrase Extraction**: O(N) where N = corpus size
- **Matrix Construction**: O(C × P) where C = contexts, P = phrases
- **Graph Construction**: O(C² × P) - quadratic in contexts
- **Fingerprint Generation**: O(P × C × D²) where D = grid dimensions

### Scalability Considerations
- Memory bottleneck: term-context matrix for large corpora
- Recommended: Pre-filter phrases by frequency/domain relevance
- Grid resolution trade-off: higher D = more precision but larger matrices

### Experimental Configurations

#### Small-Scale Experiments (`Document_clustering.ipynb`)
- **Corpus Size**: ~20 documents
- **TF-IDF Features**: 10
- **SVD Components**: 10
- **SOM Grid**: 30×30
- **Use Case**: Proof-of-concept and visualization

#### Medium-Scale Experiments (`Semantic_Space_Construction.ipynb`)
- **Contexts**: 20 documents
- **Grid Dimensions**: 10×10 (configurable)
- **Phrase Threshold**: Frequency-based filtering
- **Use Case**: Full semantic space construction

#### Large-Scale Experiments (`Umap_fingerprint.ipynb`)
- **Documents**: Hundreds to thousands
- **UMAP Components**: 2D projection
- **Grid Binning**: 16×16 fingerprint matrix
- **Use Case**: Scalability testing and comparison

### Error Handling and Debugging

#### Common Issues (`ground-up.ipynb`)
```python
# Input validation - ensure string input for spaCy
try:
    doc = nlp(text_string)  # Not list of strings
except ValueError as e:
    print(f"Input error: {e}")
    # Process line by line instead
```

#### File Path Handling
```python
# Colab vs local path compatibility
import os
base_path = "/content/drive/MyDrive/semantic_folding/"  # Colab
# base_path = "./"  # Local

phrases_file = os.path.join(base_path, "phrases.txt")
```

#### Memory Optimization
```python
# For large corpora, process in batches
batch_size = 1000
for i in range(0, len(corpus), batch_size):
    batch = corpus[i:i+batch_size]
    # Process batch
```

### Quality Metrics
- **Semantic Coherence**: Phrases in same grid region should be related
- **Context Preservation**: Similar contexts should have proximate coordinates
- **Fingerprint Distinctiveness**: Different phrases should have different spatial patterns

## Future Research Directions

### Extensions
1. **Multi-Modal Integration**: Combine text with structured data
2. **Temporal Dynamics**: Track semantic space evolution over time
3. **Hierarchical Spaces**: Multi-resolution grid representations
4. **Cross-Lingual Alignment**: Project multiple languages into shared space

### Optimization Opportunities
1. **Approximate Methods**: Faster graph layout algorithms
2. **Distributed Processing**: Parallel fingerprint generation
3. **Compression**: Sparse matrix representations for large vocabularies

## Usage Example

## Advanced Usage Patterns

### Research Applications

#### Comparative Analysis Pipeline
```bash
# Run semantic folding
python 1-phrase_extractor.py
python 2-term_context.py
python 3-semantic_space.py
python 4-fingerprints_generator.py

# Run comparative baselines (from notebooks)
# TF-IDF + SVD: Document_representation.ipynb
# SOM Clustering: Document_clustering.ipynb
# UMAP Fingerprints: Umap_fingerprint.ipynb
```

#### Parameter Sensitivity Analysis
```python
# Experiment with different grid resolutions
NUM_DIMENSIONS = [8, 10, 12, 16]  # Test different scales

for dims in NUM_DIMENSIONS:
    # Modify 3-semantic_space.py parameters
    # Compare resulting semantic spaces
    pass
```

#### Domain Adaptation
```python
# Fine-tune for specific domains
# Adjust phrase filtering thresholds
# Modify context similarity weights
# Customize grid positioning algorithms
```

### Integration with Knowledge Graphs

#### Triple-to-Context Conversion
```python
# Convert KG triples to contexts
def triple_to_context(subject, predicate, object):
    return f"{subject} {predicate} {object}"

# Generate semantic fingerprints for entities
contexts = [triple_to_context(s, p, o) for s, p, o in triples]
```

#### Entity Similarity Computation
```python
# Compare entity fingerprints
def compute_entity_similarity(entity1_fingerprint, entity2_fingerprint):
    # Cosine similarity, Jaccard similarity, or spatial proximity
    return spatial_similarity(entity1_fingerprint, entity2_fingerprint)
```

## Pipeline Execution

### Basic Workflow
```bash
# Sequential execution
python 1-phrase_extractor.py     # Extract phrases
python 2-term_context.py         # Build term-context matrix
python 3-semantic_space.py       # Construct semantic space
python 4-fingerprints_generator.py  # Generate phrase fingerprints
python 6-generate_document_fingerprints.py  # Create document fingerprints
python 7-visualize-docs.py       # Visualize results
```

### Notebook-Based Execution
For interactive experimentation and visualization:
1. **Start with `pre_processing.ipynb`** for data cleaning
2. **Run `Semantic_Space_Construction.ipynb`** for complete pipeline
3. **Use domain-specific notebooks** for comparative analysis
4. **Execute `ground-up.ipynb`** for debugging and development

## References and Related Work

### Foundational Work
- **Distributional Semantics**: Harris (1954), Firth (1957)
- **Vector Space Models**: Salton et al. (1975)
- **Graph-based Methods**: Sahlgren (2006) random indexing
- **Manifold Learning**: Roweis & Saul (2000) locally linear embedding

### Implementation References
- **spaCy Documentation**: Linguistic processing and NLP pipelines
- **NetworkX**: Graph algorithms and force-directed layouts
- **UMAP**: McInnes et al. (2018) - Uniform Manifold Approximation
- **MiniSom**: Self-Organizing Map implementation for Python

### Experimental Notebooks
- **`Semantic_Space_Construction.ipynb`**: Complete algorithm implementation with Colab integration
- **`Umap_fingerprint.ipynb`**: Comparative UMAP-based fingerprinting methodology
- **`Document_clustering.ipynb`**: SOM clustering baseline implementation
- **`Document_representation.ipynb`**: TF-IDF + SVD traditional approach
- **`pre_processing.ipynb`**: Text preprocessing and cleaning pipeline
- **`ground-up.ipynb`**: Foundational phrase extraction experiments

### Key Technical Insights from Notebooks
1. **Preprocessing Importance**: Lemmatization and stop-word removal significantly impact phrase quality
2. **Parameter Sensitivity**: Grid dimensions (8×8 vs 16×16) affect fingerprint resolution vs. sparsity
3. **Scalability Trade-offs**: UMAP scales better than force-directed layouts for large corpora
4. **Evaluation Challenges**: Semantic coherence metrics need domain-specific validation
5. **Integration Patterns**: Google Colab provides accessible experimentation environment

---

**Note**: This implementation provides a foundation for semantic folding research. Parameters should be tuned for specific domains and corpus characteristics. The notebooks demonstrate both the algorithm's capabilities and its comparison with traditional methods. The approach is particularly valuable for applications requiring interpretable, spatially-organized semantic representations.
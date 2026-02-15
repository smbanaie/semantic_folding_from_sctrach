# Semantic Folding Pipeline - How to Run

This guide provides step-by-step instructions for running the Semantic Folding evaluation pipeline on different corpora.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Interactive TUI Interface](#interactive-tui-interface)
4. [Quick Start](#quick-start)
5. [Running the Pipeline](#running-the-pipeline)
6. [Testing with Different Corpora](#testing-with-different-corpora)
7. [Output Structure](#output-structure)
8. [Configuration Options](#configuration-options)
9. [Troubleshooting](#troubleshooting)
10. [Performance Tuning](#performance-tuning)

## Prerequisites

### System Requirements
- **Python**: 3.11+
- **RAM**: Minimum 8GB, recommended 16GB+ for large corpora
- **Storage**: 10GB+ free space for intermediate files
- **OS**: Linux, macOS, or Windows (with WSL recommended)

### Dependencies
The pipeline uses `uv` for dependency management. Key dependencies include:
- `loguru` - Advanced logging
- `scipy` - Sparse matrix operations (optional, falls back to dense)
- `tqdm` - Progress bars
- `lancedb` - Vector storage (for Phase 4+)
- `rank-bm25` - BM25 baseline
- `sentence-transformers` - Dense retrieval baseline
- `plotly` - Visualization (optional)

## Installation

### 1. Install UV (if not already installed)
```bash
# On Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.sh | iex"
```

### 2. Clone and setup the project
```bash
cd knowledge-graph-builder
uv sync
```

### 3. Optional: Install additional dependencies
```bash
# For full functionality (recommended)
uv add scipy lancedb rank-bm25 plotly

# For NLP processing (if available)
uv sync --extra nlp

# For the interactive TUI interface
uv add pyyaml questionary
```

## Interactive TUI Interface

The easiest way to run and manage the semantic folding pipeline is through the interactive Text User Interface (TUI):

```bash
# Launch the TUI
uv run python brain_approaches/semantic_folding/semantic_folder.py
```

### TUI Features

- **Pipeline Status Overview**: Shows completion status of each phase and automatically detects errors from previous runs
- **Interactive Phase Selection**: Run individual phases or the complete pipeline
- **Configuration Management**: Easily modify corpus path, grid size, and logging options
- **Output File Browser**: View and inspect generated files and directory structure
- **Error Detection**: Automatically checks log files for errors and warnings
- **Output Management**: Clean up old output directories with confirmation
- **Progress Reporting**: Real-time progress indicators with elapsed time and completion statistics
- **Resume Capability**: Automatically resume interrupted pipelines from the last completed phase

### TUI Menu Options

1. **🔄 Run All Phases**: Execute the complete pipeline from start to finish
2. **🎯 Run Specific Phase**: Select and run individual pipeline phases (useful for debugging or resuming)
3. **🗂️ View Output Files**: Browse output directories, view file counts, and key statistics
4. **📋 Configure Pipeline**: Interactive setup of corpus path, grid size, and logging levels
5. **🧹 Clean Old Outputs**: Remove previous run directories with size information and confirmation
6. **❌ Exit**: Close the interface

### Configuration File

Pipeline settings are managed through `config/semantic_folding.yml`:

```yaml
# Semantic Folding Pipeline Configuration - OPTIMIZED SETTINGS
# This file contains production-ready optimized settings for quality results
# Based on extensive testing and quality analysis

# Input Data
corpus_path: "data/HippoRAG2/dataset/musique_corpus.json"
queries_path: "data/HippoRAG2/dataset/musique.json"

# Output Settings
output_base: "outputs"

# Pipeline Parameters - OPTIMIZED FOR QUALITY
grid_size: 32                    # Semantic space grid size (optimized for distribution)
max_phrases: null               # Limit phrases for testing (null = no limit)
max_docs: null                  # Limit documents for testing (null = no limit)

# Logging Configuration
log_level: "INFO"               # Console log level (DEBUG, INFO, WARNING, ERROR)
debug: false                    # Enable debug mode with stack traces

# Performance Settings - TUNED FOR OPTIMAL RESULTS
batch_size: 1000               # Processing batch size for large datasets
max_edges: 200                 # Maximum edges in semantic graph (tuned for clean layouts)
edge_threshold: 0.05          # Minimum similarity for graph edges (optimized for connectivity)

# Matrix Normalization Settings
normalize_matrix: true         # Apply TF-IDF normalization to reduce high-frequency word dominance

# Document Fingerprint Settings
doc_top_percent: 0.05         # Keep only top N% of cells in document fingerprints (default: 5%)
doc_no_threshold: false       # Set to true to disable document fingerprint thresholding

# Module Settings
use_spacy: true               # Use spaCy for phrase extraction (fallback if unavailable)
no_visualization: false       # Generate matplotlib visualizations

# Quality Optimization Notes:
# - grid_size: 32 provides better semantic space utilization than 16 or 8
# - max_edges: 200 creates cleaner, more meaningful layouts than 50000
# - edge_threshold: 0.05 allows better connectivity than 0.1
# - normalize_matrix: TF-IDF weighting reduces high-frequency word dominance
# - doc_top_percent: 5% threshold improves document fingerprint sparsity
# - These settings were determined through quality analysis and testing
```

### Resume Mode

Resume a pipeline that was interrupted:

```bash
# Resume from last saved state
uv run python brain_approaches/semantic_folding/semantic_folder.py --resume
```

The resume state is automatically saved after each completed phase and stored in `~/.semantic_folding_resume.json`.

### Progress Reporting

The TUI provides real-time progress feedback during execution:

- **Progress Indicators**: ASCII spinning indicators showing current phase and elapsed time
- **Phase Statistics**: Detailed output showing what was created (documents processed, files generated, etc.)
- **Completion Summary**: Final statistics for the entire pipeline including file counts and sizes
- **Error Details**: Clear error reporting with relevant output when phases fail

Example progress output:
```
>>> Phase 2: Phrase Extraction
============================================================
Command: uv run python brain_approaches/semantic_folding/phrase_extractor.py --corpus_path outputs/musique_20260215_123456/corpus.txt --output_dir outputs/musique_20260215_123456
Starting execution...
| Phrase Extraction... (0.0s elapsed)
/ Phrase Extraction... (0.5s elapsed)
- Phrase Extraction... (1.0s elapsed)
   [SUCCESS] Phase 2 completed successfully!
   Phrases extracted: 25,572
   Matrix created: 11656 × 25572
   Sparsity: 0.0028 (827,185 entries)
```

### Matrix Normalization & Document Fingerprint Thresholding

The pipeline includes advanced normalization and thresholding features:

#### TF-IDF Matrix Normalization
- **Purpose**: Reduces dominance of high-frequency/common words in semantic relationships
- **Method**: Applies TF-IDF weighting to term-context matrix entries
- **Benefit**: Prevents stopwords and frequent phrases from overwhelming semantic connections
- **Control**: Set `normalize_matrix: false` to disable

#### Document Fingerprint Sparsification
- **Purpose**: Creates more focused and interpretable document representations
- **Method**: Retains only top 5% of activated cells in document fingerprints
- **Benefit**: Improves semantic specificity and reduces noise
- **Control**: Adjust `doc_top_percent` (default: 0.05) or set `doc_no_threshold: true`

#### Expected Improvements
- **Better semantic separation** between related vs unrelated documents
- **Reduced centralization** in semantic space visualizations
- **More meaningful fingerprints** with higher information density
- **Improved retrieval quality** through focused semantic representations

### Complete Configuration Reference

The semantic folding pipeline supports extensive configuration for optimization:

#### Input/Output Settings
```yaml
corpus_path: "data/HippoRAG2/dataset/musique_corpus.json"  # Path to input corpus
queries_path: "data/HippoRAG2/dataset/musique.json"       # Path to query file (optional)
output_base: "outputs"                                    # Base directory for outputs
```

#### Core Pipeline Parameters
```yaml
grid_size: 32                    # Semantic space grid size (8, 16, 32 recommended)
max_phrases: null               # Limit phrases for testing (null = no limit)
max_docs: null                  # Limit documents for testing (null = no limit)
```

#### Semantic Graph Construction
```yaml
max_edges: 200                 # Maximum edges in semantic graph (tuned for clean layouts)
edge_threshold: 0.05          # Minimum similarity for graph edges (optimized for connectivity)
```

#### Matrix Processing
```yaml
normalize_matrix: true         # Apply TF-IDF normalization to reduce high-frequency word dominance
batch_size: 1000               # Processing chunk size for large datasets
```

#### Document Fingerprint Generation
```yaml
doc_top_percent: 0.05         # Keep only top N% of cells in document fingerprints
doc_no_threshold: false       # Disable thresholding to keep all cells
```

#### Logging and Visualization
```yaml
log_level: "INFO"             # Console log level (DEBUG, INFO, WARNING, ERROR)
debug: false                  # Enable debug mode with detailed tracebacks
use_spacy: true               # Use spaCy for phrase extraction (fallback if unavailable)
no_visualization: false       # Generate matplotlib visualizations
```

### Command-Line Arguments

Each module supports additional command-line arguments:

#### Phrase Extraction (`phrase_extractor.py`)
```bash
--corpus_path PATH          # Input corpus file
--output_dir DIR           # Output directory
--batch_size INT           # Batch size for spaCy processing (default: 100)
--min_freq INT             # Minimum phrase frequency (default: 2)
--max_length INT           # Maximum phrase length (default: 50)
--use_spacy               # Force spaCy usage
--max_words INT           # Maximum words per phrase (default: 4)
```

#### Term-Context Matrix (`term_context.py`)
```bash
--phrases_path PATH        # Phrases file path
--corpus_path PATH         # Corpus file path
--output_dir DIR          # Output directory
--chunk_size INT          # Processing chunk size (default: 1000)
--no_normalization        # Disable TF-IDF normalization
```

#### Semantic Space (`semantic_space.py`)
```bash
--matrix_path PATH        # Term-context matrix file
--corpus_path PATH        # Corpus file path
--output_dir DIR          # Output directory
--grid_size INT           # Grid size (default: 16)
--max_edges INT           # Maximum edges in graph (default: 50000)
--edge_threshold FLOAT    # Edge weight threshold (default: 0.1)
```

#### Phrase Fingerprints (`phrase_fingerprints.py`)
```bash
--matrix_path PATH        # Term-context matrix file
--coordinates_path PATH   # Context coordinates file
--phrases_path PATH       # Phrases file path
--output_dir DIR          # Output directory
--grid_size INT           # Grid size (default: 16)
--max_phrases INT         # Limit phrases to process
```

#### Document Fingerprints (`doc_fingerprints.py`)
```bash
--corpus_path PATH        # Corpus file path
--phrases_path PATH       # Phrases file path
--fingerprints_dir DIR    # Phrase fingerprints directory
--output_dir DIR          # Output directory
--grid_size INT           # Grid size (default: 16)
--max_docs INT           # Limit documents to process
--top_percent FLOAT      # Top percentage threshold (default: 0.05)
--no_threshold           # Disable thresholding
--use_spacy              # Force spaCy usage
```

### Performance Tuning Guidelines

#### For Speed
```yaml
grid_size: 16
max_edges: 100
edge_threshold: 0.1
batch_size: 500
```

#### For Quality (Recommended)
```yaml
grid_size: 32
max_edges: 200
edge_threshold: 0.05
normalize_matrix: true
doc_top_percent: 0.05
```

#### For Memory Efficiency
```yaml
batch_size: 100
max_edges: 100
grid_size: 16
```

#### For Maximum Quality
```yaml
grid_size: 32
max_edges: 500
edge_threshold: 0.01
normalize_matrix: true
doc_top_percent: 0.03
batch_size: 2000
```

### Non-Interactive Mode

For automation, CI/CD, or scripting:

```bash
# Show status only (no interactive prompts)
uv run python brain_approaches/semantic_folding/semantic_folder.py --non-interactive

# Run specific phase in specific output directory
uv run python brain_approaches/semantic_folding/semantic_folder.py --run-phase 3 --output-dir outputs/musique_20260215_123456
```

## Quick Start

### Run on MuSiQue dataset (default)
```bash
cd brain_approaches/semantic_folding
uv run python scratchpad.py --corpus_path ../../data/HippoRAG2/dataset/musique_corpus.json
```

### Run on custom corpus
```bash
# Assuming you have a corpus.json file
uv run python scratchpad.py --corpus_path /path/to/your/corpus.json
```

## Running the Pipeline

### Command Line Options

```bash
uv run python scratchpad.py [OPTIONS]

Options:
  --corpus_path PATH        Path to corpus JSON file (required)
  --queries_path PATH       Path to queries JSON file (default: musique.json)
  --grid_size INT           Semantic space grid size (default: 16)
  --top_k INT [INT ...]     Top-K values for evaluation (default: [1,5,10,20])
  --output_base PATH        Base output directory (default: "outputs")
  --log_level {DEBUG,INFO,WARNING,ERROR}
                            Console logging level (default: INFO)
  --debug                   Enable debug mode with detailed tracebacks
  --help                    Show help message
```

### Pipeline Phases

The pipeline runs through 8 phases with current completion status:

#### ✅ **Completed Phases (1-5)**
1. **Phase 1**: Corpus Loading & Preprocessing ✅
   - Loads JSON corpus and converts to text format
   - Creates timestamped output directory
   - Logs corpus statistics (11,656 passages, 930K tokens)

2. **Phase 2**: Phrase Extraction ✅
   - Extracts 1-4 word phrases with quality filtering
   - Generates 25,572 filtered phrases from raw 134K extractions
   - Creates phrase frequency visualizations

3. **Phase 3**: Term-Context Matrix ✅
   - Builds sparse matrix: 11,656 × 25,572 (0.28% density, 827K entries)
   - 95% memory reduction compared to dense matrices
   - Generates sparsity heatmaps and statistics

4. **Phase 4**: Semantic Space Construction ✅
   - Creates 16×16 semantic grid via force-directed graph layout
   - Maps 11,656 contexts to grid coordinates
   - Generates network visualizations and grid heatmaps

5. **Phase 5**: Fingerprint Generation ✅
   - Generates 25K+ phrase fingerprints (16×16 binary matrices)
   - Creates 11K+ document fingerprints with comprehensive metadata
   - Efficient phrase-to-fingerprint aggregation

#### 🔄 **Next Phases (6-8)**
6. **Phase 6**: LanceDB Integration (Next Priority)
   - Fast vector similarity search for semantic retrieval
   - Bulk storage of fingerprints with metadata
   - Query fingerprint matching and ranking

7. **Phase 7**: Multi-Method Evaluation
   - Compare Semantic Folding vs TF-IDF/BM25/Dense/Graph baselines
   - Recall@K, MRR, MAP, EM, F1 metrics with statistical testing
   - Performance analysis and ablation studies

8. **Phase 8**: Comprehensive Benchmarking
   - Multi-corpus evaluation (MuSiQue, HotpotQA, 2WikiMultiHopQA)
   - Scalability testing and error analysis
   - Production deployment validation

## Testing with Different Corpora

### Corpus Format Requirements

Your corpus must be a JSON file with the following structure:

```json
[
  {
    "title": "Document Title",
    "text": "Full document text content..."
  },
  {
    "title": "Another Document",
    "text": "More text content..."
  }
]
```

### Example: Testing with Custom Corpus

1. **Prepare your corpus**:
```bash
# Create a corpus.json file
[
  {"title": "Doc 1", "text": "This is a sample document about artificial intelligence and machine learning."},
  {"title": "Doc 2", "text": "Natural language processing is a subfield of AI that focuses on language understanding."}
]
```

2. **Run the pipeline**:
```bash
uv run python scratchpad.py --corpus_path /path/to/your/corpus.json
```

3. **Check results**:
```bash
ls -la outputs/$(date +%Y%m%d)_*/
# Look for:
# - corpus.txt (processed corpus)
# - phrases.txt (extracted phrases)
# - term_context_matrix.npz (sparse matrix)
# - logs/pipeline.log (execution log)
```

### Supported Corpus Types

- **MuSiQue**: Question-answering dataset (19K passages)
- **HotpotQA**: Multi-hop QA dataset
- **2WikiMultiHopQA**: Wikipedia-based QA
- **Custom JSON**: Any corpus in the required JSON format

### Corpus Size Guidelines

| Corpus Size | Memory | Time | Recommended Config |
|-------------|--------|------|-------------------|
| < 1K docs   | 2GB    | 5min | Default settings  |
| 1K-10K docs | 4GB    | 15min| Default settings  |
| 10K-50K docs| 8GB    | 1hr  | `--grid_size 16`  |
| > 50K docs  | 16GB+  | 4hr+ | `--grid_size 8`   |

## Output Structure

```
outputs/YYYYMMDD_HHMMSS/
├── corpus.txt                    # Processed corpus (idx,title: text)
├── phrases.txt                   # 25K+ filtered phrases with frequencies
├── term_context_matrix.npz      # Sparse matrix (827K entries, 0.28% density)
├── term_context_matrix.json     # Matrix metadata and statistics
├── context_coordinates.csv      # 16×16 grid coordinates for all contexts
├── fingerprints/                # 25K+ phrase fingerprint files (16×16 matrices)
│   ├── phrase_1_fingerprint.txt
│   └── ...
├── doc_fingerprints/            # 11K+ document fingerprints + metadata
│   ├── doc_0_fingerprint.txt
│   ├── doc_0_metadata.json
│   └── ...
├── logs/
│   └── pipeline.log             # Comprehensive execution log
└── visualizations/              # Charts and plots (if matplotlib available)
    ├── phrase_frequencies.png   # Top phrases bar chart
    ├── matrix_sparsity.png      # Matrix sparsity visualization
    ├── semantic_network.png     # Force-directed graph layout
    └── semantic_grid_heatmap.png # Grid distribution heatmap
```

## Configuration Options

### Logging Options
Control logging verbosity and output:
- `--log_level INFO`: Default console logging (INFO, WARNING, ERROR)
- `--log_level DEBUG`: Detailed console logging with all messages
- `--debug`: Enable debug mode with stack traces and diagnostics

### Grid Size
Controls semantic space resolution:
- `--grid_size 8`: Faster, lower resolution (64 dimensions)
- `--grid_size 16`: Default, balanced (256 dimensions)
- `--grid_size 32`: Higher resolution, slower (1024 dimensions)

### Top-K Values
Evaluation metrics computed for these ranks:
- `--top_k 1 5 10 20`: Default (standard IR metrics)
- `--top_k 1 3 5 10 20 50`: More detailed analysis

### Memory Optimization
For large corpora, the pipeline automatically:
- Uses sparse matrices when scipy is available
- Falls back to memory-efficient dense operations
- Processes documents in batches
- Limits visualization for very large datasets

## Troubleshooting

### Common Issues

#### 1. "scipy not available" Warning
**Symptom**: Pipeline falls back to dense matrix operations
**Solution**:
```bash
uv add scipy
# Restart pipeline
```

#### 2. Memory Errors
**Symptom**: "MemoryError" or "killed" process
**Solutions**:
- Reduce grid size: `--grid_size 8`
- Process smaller batches (modify code)
- Add more RAM or use smaller corpus

#### 3. Slow Performance
**Symptom**: Pipeline takes very long
**Solutions**:
- Use smaller grid size
- Process subset of corpus for testing
- Ensure scipy is installed for sparse operations

#### 4. No Output Files Created
**Symptom**: Pipeline runs but no files in output directory
**Check**:
```bash
# Check logs
tail -f outputs/*/logs/pipeline.log

# Verify corpus format
python -c "import json; print(len(json.load(open('corpus.json'))))"
```

#### 5. Import Errors
**Symptom**: "ModuleNotFoundError"
**Solution**:
```bash
# Ensure dependencies are installed
uv sync
uv add loguru tqdm

# If using NLP features
uv sync --extra nlp
```

### Debug Mode

Run with detailed logging:
```bash
# Set log level to DEBUG
export LOGURU_LEVEL=DEBUG
uv run python scratchpad.py --corpus_path corpus.json
```

### Recovery from Failures

If pipeline fails mid-execution:

1. **Check logs** for exact failure point
2. **Resume from checkpoint** (if implemented) or
3. **Restart** - pipeline will recreate output directory

## Performance Tuning

### For Speed
```bash
# Smaller grid, fewer evaluation points
uv run python scratchpad.py \
  --corpus_path corpus.json \
  --grid_size 8 \
  --top_k 1 5 10
```

### For Accuracy
```bash
# Larger grid, more evaluation points
uv run python scratchpad.py \
  --corpus_path corpus.json \
  --grid_size 32 \
  --top_k 1 3 5 10 20 50
```

### Memory Usage Optimization
- Use scipy for sparse matrices (reduces memory by ~95%)
- Process smaller corpora first
- Monitor memory with `htop` or `top`

### Scaling to Large Corpora
1. **Test on small subset first**
2. **Use appropriate grid size** (8-16 for large corpora)
3. **Ensure scipy is available**
4. **Monitor disk space** (matrices can be large)

## Validation

### Check Pipeline Completion
```bash
# Verify all phases completed
grep "Phase.*completed successfully" outputs/*/logs/pipeline.log

# Check file counts
find outputs/*/ -name "*.txt" -o -name "*.npz" -o -name "*.json" | wc -l
```

### Validate Results
```bash
# Check phrase quality
head -20 outputs/*/phrases.txt

# Verify matrix dimensions
python -c "
import numpy as np
import json
meta = json.load(open('outputs/*/term_context_matrix.json'))
print(f'Matrix: {meta[\"num_contexts\"]} x {meta[\"num_phrases\"]}')
print(f'Density: {meta[\"density\"]:.4f}')
"
```

## Contributing

### Adding New Corpora
1. Convert to required JSON format
2. Test with small subset first
3. Update this guide with corpus-specific notes
4. Document any format variations

### Extending the Pipeline
1. Add new phases in `scratchpad.py`
2. Update task list in `SCRATCHPAD.md`
3. Add configuration options as needed
4. Update this documentation

## Support

For issues or questions:
1. Check the logs in `outputs/*/logs/pipeline.log`
2. Verify corpus format matches requirements
3. Test with smaller corpus first
4. Check dependency installation

---

*Last updated: February 15, 2026*
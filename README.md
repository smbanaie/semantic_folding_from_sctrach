# Knowledge Graph Builder

A LangGraph-based multi-agent system that transforms unstructured text corpora into structured knowledge graphs stored in Memgraph.

## Features

- **OpenIE Integration**: Implements Open Information Extraction following HippoRAG2 methodology for schemaless knowledge triple extraction
- **Multi-Agent Pipeline**: Specialized agents for analysis, splitting, chunking, extraction, and validation
- **Parallel Processing**: Efficient extraction using concurrent chunk processing
- **Enhanced Entity Normalization**: Reduces entity length by 42.2% while preserving semantic meaning
- **Quick Boundary-Based Splitting**: Fast splitting for large textbooks without resource-intensive embeddings
- **Cost-Optimized**: Uses free OpenRouter API models with intelligent model selection
- **Production-Ready**: Error handling, retry logic, and validation at every stage

## Architecture

The system uses a 5-agent pipeline:
1. **Analyzer**: Understands corpus and generates extraction strategy
2. **Splitter**: Divides corpus into logical sections
3. **Chunker**: Creates optimal context windows with overlap
4. **Extractor**: Extracts knowledge triples in parallel using OpenIE methodology
5. **Reviewer**: Validates, normalizes, and deduplicates triples

See [Architecture.md](./Architecture.md) for detailed documentation.

## Enhanced Entity Normalization

The Reviewer agent includes advanced entity normalization that significantly improves graph readability:

- **42.2% average reduction** in entity length
- **Preserves semantic meaning** while shortening lengthy phrases
- **Intelligent acronym handling** (LLMs, RAG, NER, etc.)
- **Examples of improvements**:
  - "Rich Relational Structure" → "Relational Structure"
  - "Large Language Models" → "LLMs"
  - "Questions That Require Connecting Information From Multiple Sources" → "Complex Questions"
  - "Traditional RAG Approaches" → "Traditional RAG"

## OpenIE Integration

The system implements **Open Information Extraction (OpenIE)** following the HippoRAG2 methodology:

- **Schemaless Extraction**: Extracts any relationships from text without predefined ontologies
- **Two-Phase Process**:
  1. **Entity Extraction**: Identifies named entities and key concepts
  2. **Relation Extraction**: Uses entities as context for accurate triple extraction
- **Domain-Independent**: Works across any subject matter without domain-specific training
- **Discrete Triples**: Focuses on noun phrases for fine-grained pattern separation
- **Fallback Mechanism**: Automatically falls back to standard extraction if OpenIE fails

This approach enables flexible knowledge graph construction that can handle diverse and complex relationships in text corpora.

## Prerequisites

- Python 3.11+
- [UV package manager](https://github.com/astral-sh/uv) - Install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- OpenRouter API key (get free at [openrouter.ai](https://openrouter.ai))
- Memgraph (Docker recommended):
  ```bash
  docker run -p 7687:7687 memgraph/memgraph
  ```

## Setup

1. **Clone and navigate to the project**:
   ```bash
   cd knowledge-graph-builder
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```
   This creates a virtual environment (`.venv/`) and installs all dependencies.

3. **Configure environment**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your OpenRouter API key.

4. **Add corpus files**:
   Place your text files in `data/corpus/` directory.

## Usage

### Basic Usage

```bash
# Run with default corpus from data/corpus/
uv run main.py

# Use specific corpus directory
uv run main.py --corpus-dir ./data/custom_corpus

# Process specific files
uv run main.py --corpus-files "doc1.txt,doc2.txt"
uv run python main.py --corpus-files data/corpus/graphrag_rag_corpus.txt

# Clear graph and reprocess
uv run main.py --clear-graph
```

### UV Commands

```bash
# Install dependencies
uv sync

# Add a new package
uv add <package-name>

# Add dev dependency
uv add --dev <package-name>

# Run Python script
uv run main.py

uv run python main.py --corpus-files data/corpus/sample.txt --output-directory data/output --chunk-size 800 --max-parallel-extractions 5
uv run python main.py --corpus-files data/corpus/sample.txt --visualize
uv run python main.py --corpus-files data/corpus/sample.txt
# Run tests
uv run pytest

# Enter virtual environment (optional)
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

## Project Structure

```
knowledge-graph-builder/
├── src/
│   ├── agents/          # Agent nodes (analyzer, splitter, chunker, extractor, reviewer)
│   ├── graph/           # LangGraph workflow definition
│   ├── storage/         # Memgraph client
│   ├── utils/           # Helper utilities (OpenRouter client)
│   ├── models/          # Pydantic data models
│   └── config.py        # Configuration management
├── tests/               # Test files
├── data/
│   ├── corpus/          # Input corpus files
│   ├── input/           # Additional input files
│   └── output/          # Generated results
├── main.py              # Entry point
└── pyproject.toml       # UV project configuration
```

## Development

See [Cursor.guide.md](./Cursor.guide.md) for step-by-step implementation guide.

### Running Tests

```bash
uv run pytest
```

### Code Style

This project follows Python best practices with:
- Type hints
- Pydantic for data validation
- Async/await for I/O operations
- Comprehensive error handling

## Configuration

All configuration is managed through environment variables (see `.env.example`):

- **OpenRouter**: API key and base URL
- **Memgraph**: Connection URI, user, password
- **Corpus**: Directories and file paths
- **Models**: Model selection for each agent
- **Processing**: Chunk size, overlap, parallelism settings

## Output

The system generates:
- **Knowledge Graph**: Stored in Memgraph (queryable via Cypher)
- **Triples JSON**: Validated triples saved to `data/output/triples.json`
- **Statistics**: Console output with processing stats

### Loading Triples into Memgraph

Use the provided graph importer script for easy import:

```bash
# Import triples from JSON file
uv run python graph_importer.py data/output/triples.json

# Or with a specific file
uv run python graph_importer.py data/output/triples_20251228_234452.json

# Skip the visualization guide
uv run python graph_importer.py data/output/triples.json --no-guide
```

The script will:
- Load triples from your JSON file
- Clear any existing graph in Memgraph
- Bulk import all triples
- Show import statistics
- Display visualization and querying instructions

**JSON Format Expected:**
```json
[
  {
    "subject": "RAG",
    "predicate": "COMBINE",
    "object": "LLMs",
    "confidence": 0.95,
    "source_chunk_id": "section_0_chunk_0",
    "metadata": {}
  }
]
```

### Visualization with Memgraph Lab

After importing with `graph_importer.py`, you'll get a complete guide. For quick reference:

- **Memgraph Lab Web Interface**: Open `http://localhost:3000` in your browser
- **Interactive Graph Exploration**: Visual graph representations with Cypher queries
- **Query Editor**: Write and execute Cypher queries with syntax highlighting

### Essential Cypher Queries

**View the entire graph:**
```cypher
MATCH (s)-[r]->(o) RETURN s, r, o LIMIT 50;
```

**Find most connected entities:**
```cypher
MATCH (n)-[r]-()
RETURN n.name, count(r) AS connections
ORDER BY connections DESC LIMIT 10;
```

**Explore relationships for specific entities:**
```cypher
MATCH (s)-[r]->(o)
WHERE s.name CONTAINS "RAG"
RETURN s, r, o;
```

**Get graph statistics:**
```cypher
MATCH (n) RETURN count(n) AS nodes;
MATCH ()-[r]->() RETURN count(r) AS relationships;
```

**Find relationships by type:**
```cypher
MATCH (s)-[r]->(o)
WHERE type(r) = "COMBINE"
RETURN s, r, o LIMIT 20;
```

**Analyze confidence scores:**
```cypher
MATCH (s)-[r]->(o)
RETURN s.name, type(r), o.name, r.confidence
ORDER BY r.confidence DESC LIMIT 25;
```

**Find paths between entities:**
```cypher
MATCH path = (a)-[*2]-(b)
WHERE a.name = "RAG" AND b.name = "LLMs"
RETURN path LIMIT 5;
```

**Advanced: Subgraph extraction:**
```cypher
MATCH (n)-[r]-()
WITH n, count(r) AS degree
WHERE degree > 5
MATCH (n)-[r]-(m)
RETURN n, r, m;
```

## Performance

Expected throughput:
- 10,000 words: ~3-5 minutes
- 100,000 words: ~15-25 minutes
- 1,000,000 words: ~2-3 hours

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]


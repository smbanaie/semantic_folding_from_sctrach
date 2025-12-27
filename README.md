# Knowledge Graph Builder

A LangGraph-based multi-agent system that transforms unstructured text corpora into structured knowledge graphs stored in Memgraph.

## Features

- **Multi-Agent Pipeline**: Specialized agents for analysis, splitting, chunking, extraction, and validation
- **Parallel Processing**: Efficient extraction using concurrent chunk processing
- **Cost-Optimized**: Uses free OpenRouter API models with intelligent model selection
- **Production-Ready**: Error handling, retry logic, and validation at every stage

## Architecture

The system uses a 5-agent pipeline:
1. **Analyzer**: Understands corpus and generates extraction strategy
2. **Splitter**: Divides corpus into logical sections
3. **Chunker**: Creates optimal context windows with overlap
4. **Extractor**: Extracts knowledge triples in parallel
5. **Reviewer**: Validates, normalizes, and deduplicates triples

See [Architecture.md](./Architecture.md) for detailed documentation.

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

Visualize the graph using [Memgraph Lab](http://localhost:3000).

## Performance

Expected throughput:
- 10,000 words: ~3-5 minutes
- 100,000 words: ~15-25 minutes
- 1,000,000 words: ~2-3 hours

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]


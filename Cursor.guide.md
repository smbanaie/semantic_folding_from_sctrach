# Cursor AI Agent-Oriented Coding Guide

## Building Knowledge Graph System Step-by-Step

------

## 🚀 Quick Start Strategy

### Phase 1: Project Setup (10 min)

### Phase 2: Core Infrastructure (30 min)

### Phase 3: Agent Implementation (60 min)

### Phase 4: Integration & Testing (30 min)

------

## 📋 Pre-Setup Checklist

Before starting in Cursor:

- [ ] Install Cursor IDE
- [ ] Install `uv` package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] Have Python 3.9+ installed (uv will manage it)
- [ ] Have OpenRouter API key ready (get free at openrouter.ai)
- [ ] Have Memgraph installed (Docker recommended: `docker run -p 7687:7687 memgraph/memgraph`)
- [ ] Create empty project folder: `knowledge-graph-builder`

------

## PHASE 1: PROJECT SETUP

### Step 1.1: Initial Project Structure

**Cursor Prompt:**

```
Create a Python project using UV package manager for a LangGraph-based multi-agent knowledge graph builder with these requirements:

1. Project name: knowledge-graph-builder
2. Use UV for dependency management (modern, fast Python package manager)
3. Python 3.11+
4. Structure:
   - src/agents/ (agent nodes)
   - src/graph/ (LangGraph workflow)
   - src/storage/ (Memgraph client)
   - src/utils/ (helpers)
   - src/models/ (Pydantic models)
   - tests/
   - data/input/ (for corpus files)
   - data/output/ (for results)
   - data/corpus/ (default corpus storage)

5. Create:
   - pyproject.toml with UV configuration and dependencies:
     * langgraph>=0.2.0
     * langchain>=0.3.0
     * langchain-openai>=0.2.0
     * pydantic>=2.0.0
     * pydantic-settings>=2.0.0
     * neo4j>=5.0.0
     * python-dotenv>=1.0.0
     * aiohttp>=3.9.0
     * tiktoken>=0.7.0
     * nltk>=3.8.0
     * rich>=13.0.0
     * tqdm>=4.66.0
   - .env.example with placeholders
   - .gitignore for Python and .venv/
   - README.md with UV setup instructions
   - main.py as entry point

6. Initialize with: `uv init` structure

Generate all files. Include UV commands in README:
- `uv sync` to install deps
- `uv run main.py` to execute
- `uv add <package>` to add deps
```

**After generation, run in terminal:**

```bash
cd knowledge-graph-builder
uv sync  # Creates .venv and installs all dependencies
```

**What to expect:** Cursor will create the full folder structure with pyproject.toml instead of requirements.txt.

------

### Step 1.2: Configuration Setup

**Cursor Prompt:**

```
Create src/config.py with a Pydantic Settings class that:

1. Loads from environment variables and .env file
2. Includes:
   - OPENROUTER_API_KEY
   - OPENROUTER_BASE_URL (default: https://openrouter.ai/api/v1)
   - MEMGRAPH_URI (default: bolt://localhost:7687)
   - MEMGRAPH_USER (default: "")
   - MEMGRAPH_PASSWORD (default: "")
   
   # Corpus Configuration
   - CORPUS_FILES: List[str] (default: ["data/corpus/sample.txt"])
   - CORPUS_DIRECTORY: str (default: "data/corpus")
   - INPUT_DIRECTORY: str (default: "data/input")
   - OUTPUT_DIRECTORY: str (default: "data/output")
   
   # Model configurations for each agent:
   - ANALYZER_MODEL = "meta-llama/llama-3.1-70b-instruct:free"
   - SPLITTER_MODEL = "google/gemini-flash-1.5"
   - CHUNKER_MODEL = "google/gemini-flash-1.5"
   - EXTRACTOR_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
   - REVIEWER_MODEL = "meta-llama/llama-3.1-70b-instruct:free"
   
   # Processing settings
   - CHUNK_SIZE: int (default: 1000)
   - CHUNK_OVERLAP: float (default: 0.15)
   - MAX_PARALLEL_EXTRACTIONS: int (default: 10)
   - BATCH_SIZE: int (default: 5)

3. Add method: get_corpus_files() -> List[Path] that:
   - Returns list of all .txt files in CORPUS_DIRECTORY
   - Falls back to CORPUS_FILES if directory is empty
   - Validates files exist

4. Add method: load_corpus() -> str that:
   - Reads and concatenates all corpus files
   - Returns combined text
   - Handles encoding errors

Use pydantic-settings v2 with model_config for .env loading.
Include proper validation and helpful error messages.
```

------

## PHASE 2: CORE INFRASTRUCTURE

### Step 2.1: Data Models

**Cursor Prompt:**

```
Create src/models/data_models.py with Pydantic models for:

1. Section: id, content, metadata (dict), start_pos, end_pos
2. Chunk: id, content, section_id, chunk_index, overlap_with_next
3. Triple: subject, predicate, object, confidence (float), source_chunk_id, metadata
4. GraphState (TypedDict) with fields:
   - corpus: str
   - corpus_metadata: dict
   - extraction_strategy: str
   - extraction_prompts: dict
   - domain_context: str
   - sections: List[Section]
   - chunks: List[Chunk]
   - raw_triples: List[Triple]
   - validated_triples: List[Triple]
   - graph_stats: dict

All models should have proper type hints, validation, and example values in docstrings.
```

------

### Step 2.2: OpenRouter Client

**Cursor Prompt:**

```
Create src/utils/openrouter_client.py with:

1. Class OpenRouterClient that:
   - Initializes with API key and model name
   - Has async method `generate(prompt: str, system_prompt: str = None) -> str`
   - Uses aiohttp for requests
   - Handles errors and retries (max 3 attempts)
   - Logs token usage
   - Supports streaming (optional)

2. Follow OpenAI-compatible API format
3. Add proper error handling for rate limits, timeouts
4. Include docstrings and type hints

Make it production-ready with logging.
```

------

### Step 2.3: Memgraph Client

**Cursor Prompt:**

```
Create src/storage/memgraph_client.py with:

1. Class MemgraphClient that:
   - Connects using neo4j driver (Memgraph uses Neo4j protocol)
   - Method: create_node(entity: str, entity_type: str, metadata: dict)
   - Method: create_relationship(subject: str, predicate: str, object: str, metadata: dict)
   - Method: bulk_insert_triples(triples: List[Triple])
   - Method: get_stats() -> dict (node count, edge count)
   - Method: clear_graph() (for testing)

2. Use context manager (with statement)
3. Handle connection errors gracefully
4. Use parameterized queries for safety
5. Add logging

Include example usage in docstring.
```

------

## PHASE 3: AGENT IMPLEMENTATION

### Step 3.1: Analyzer Agent

**Cursor Prompt:**

```
Create src/agents/analyzer_node.py with function analyzer_agent(state: GraphState) -> GraphState:

This agent:
1. Takes corpus from state
2. Analyzes the text to determine:
   - Domain/topic (scientific, news, literature, etc.)
   - Key entity types likely present (people, organizations, locations, etc.)
   - Common relationship types
3. Generates:
   - extraction_strategy: A clear strategy paragraph
   - extraction_prompts: Dict with keys "entity_extraction", "relationship_extraction" containing specific prompts
   - domain_context: Brief domain description
4. Uses OpenRouterClient with ANALYZER_MODEL
5. Returns updated state

Prompt engineering tips:
- Ask model to think step-by-step
- Request structured output (use JSON format)
- Include examples of good extraction prompts

Add comprehensive logging and error handling.
```

**Follow-up Cursor Prompt:**

```
Add to analyzer_agent:
1. Sample analysis for a 5000-word corpus about "Climate Change"
2. Example extraction prompts it might generate
3. Unit test in tests/test_analyzer.py with mock OpenRouter response
```

------

### Step 3.2: Splitter Agent

**Cursor Prompt:**

```
Create src/agents/splitter_node.py with function splitter_agent(state: GraphState) -> GraphState:

This agent:
1. Takes corpus and extraction_strategy from state
2. Uses LLM to identify logical section boundaries (chapters, topics, major theme changes)
3. Splits corpus into sections (target: 3-10 sections depending on length)
4. Creates Section objects with:
   - Unique IDs (section_0, section_1, etc.)
   - Content
   - Metadata (estimated_entities, topic, position)
5. Uses OpenRouterClient with SPLITTER_MODEL
6. Returns state with sections populated

Include:
- Fallback: If LLM fails, split by character count (every 5000 chars)
- Validation: Ensure sections don't lose content
- Logging of split statistics

Add docstring with example input/output.
```

------

### Step 3.3: Chunker Agent

**Cursor Prompt:**

```
Create src/agents/chunker_node.py with function chunker_agent(state: GraphState) -> GraphState:

This agent:
1. Takes sections from state
2. For each section:
   - Split into chunks of ~1000 tokens (use tiktoken or estimate 1 token ≈ 4 chars)
   - Chunk at sentence boundaries (don't break mid-sentence)
   - Apply 15% overlap between consecutive chunks
   - Create Chunk objects with proper IDs (section_0_chunk_0, etc.)
3. Uses simple text processing (no LLM needed, but can use for semantic splitting)
4. Returns state with chunks populated

Include:
- Sentence tokenization (use nltk or regex)
- Overlap calculation logic
- Chunk validation (min 200 tokens, max 2000 tokens)
- Statistics logging

Add unit tests for edge cases.
```

------

### Step 3.4: Extractor Agent (Parallel)

**Cursor Prompt:**

```
Create src/agents/extractor_node.py with:

1. Function extract_from_chunk(chunk: Chunk, extraction_prompts: dict, client: OpenRouterClient) -> List[Triple]
   - Takes single chunk and extraction prompts
   - Calls LLM to extract triples
   - Parses JSON response into Triple objects
   - Handles errors gracefully

2. Function extractor_agent(state: GraphState) -> GraphState (main node function)
   - Takes chunks and extraction_prompts from state
   - Uses asyncio.gather to process ALL chunks in parallel
   - Aggregates results into raw_triples
   - Adds extraction_stats (total_triples, chunks_processed, errors)
   - Returns updated state

Key requirements:
- Use EXTRACTOR_MODEL (lighter, faster model)
- Prompt should request JSON format: [{"subject": "...", "predicate": "...", "object": "..."}]
- Add retry logic for failed chunks
- Log progress (e.g., "Processed 45/100 chunks")

Include example of a good extraction prompt and expected output.
```

**Follow-up Cursor Prompt:**

```
Optimize extractor_agent for:
1. Batch API calls (5-10 chunks per request if API supports)
2. Rate limiting (max 10 concurrent requests)
3. Progress bar using tqdm
4. Save intermediate results every 50 chunks
```

------

### Step 3.5: Reviewer Agent

**Cursor Prompt:**

```
Create src/agents/reviewer_node.py with function reviewer_agent(state: GraphState) -> GraphState:

This agent:
1. Takes raw_triples from state
2. Performs validation and correction:
   - Entity normalization (e.g., "NYC" → "New York City", "Biden" → "Joe Biden")
   - Duplicate detection (same triple in different chunks)
   - Consistency checks (contradicting triples)
   - Error correction (malformed triples)
3. Uses LLM (REVIEWER_MODEL) with chain-of-thought prompting
4. Process in batches of 50-100 triples
5. Returns state with validated_triples and corrections_made

Include:
- Fuzzy matching for entity similarity (use difflib)
- Confidence scoring for each triple
- Statistics on corrections made
- Option to flag suspicious triples for human review

Add examples of common errors and how to fix them.
```

------

## PHASE 4: LANGGRAPH WORKFLOW

### Step 4.1: Build the Graph

**Cursor Prompt:**

```
Create src/graph/workflow.py that:

1. Imports all agent functions
2. Defines GraphState using the models
3. Creates StateGraph:
   - Add nodes: analyzer, splitter, chunker, extractor, reviewer, storage
   - Set entry_point to analyzer
   - Add sequential edges: analyzer → splitter → chunker → extractor → reviewer → storage
   - Add conditional edge: if extractor has errors > 30%, route back to chunker with error flag
4. Implements storage_node function that calls MemgraphClient
5. Compiles the graph
6. Exports: build_graph() function

Include:
- Checkpointing configuration
- Error handling at graph level
- Logging of state transitions
- Option to visualize graph structure

Add docstring explaining the flow.
```

------

### Step 4.2: Main Entry Point

**Cursor Prompt:**

```
Create main.py that:

1. Uses the Config class to load corpus files automatically
2. Initializes the LangGraph workflow using build_graph()
3. Creates initial state with corpus from config.load_corpus()
4. Invokes the graph: result = app.invoke(initial_state)
5. Prints statistics using rich console:
   - Corpus files loaded (from config.get_corpus_files())
   - Total sections, chunks
   - Triples extracted, validated
   - Graph nodes and edges created
   - Time taken
6. Saves validated_triples to data/output/triples.json
7. Optionally saves to timestamped file: triples_{timestamp}.json

Add:
- argparse for CLI options:
  * --corpus-dir: Override CORPUS_DIRECTORY from config
  * --corpus-files: Specific files to process (comma-separated)
  * --output: Custom output path
  * --visualize: Show graph visualization
  * --clear-graph: Clear Memgraph before running
- Rich progress output with panels and progress bars
- Exception handling for entire pipeline
- Option to resume from checkpoint (LangGraph feature)
- Summary report at the end

Example usage patterns:
```bash
# Use default corpus from config
uv run main.py

# Use specific corpus directory
uv run main.py --corpus-dir ./data/custom_corpus

# Process specific files
uv run main.py --corpus-files "doc1.txt,doc2.txt"

# Clear graph and reprocess
uv run main.py --clear-graph
```

Make it user-friendly with clear messages and colored output using rich.

```
---

## PHASE 5: TESTING & VALIDATION

### Step 5.1: Integration Test

**Cursor Prompt:**
```

Create tests/test_integration.py that:

1. Uses a small test corpus (500 words about "Tesla company")
2. Runs the full pipeline end-to-end
3. Asserts:
   - Sections created > 0
   - Chunks created > 0
   - Triples extracted > 10
   - Validated triples <= raw triples
   - Memgraph has nodes and edges
4. Cleans up (clears test graph) after test

Use pytest fixtures for setup/teardown. Mock OpenRouter API calls with realistic responses.

```
---

### Step 5.2: Sample Corpus

**Cursor Prompt:**
```

Create data/corpus/sample.txt with:

A 2000-word article about "Artificial Intelligence in Healthcare" that includes:

- Multiple named entities (organizations, people, technologies)
- Clear relationships (X developed Y, Z is used for W)
- Different topics (diagnostics, drug discovery, patient care)
- Some ambiguous references to test coreference

Make it realistic and information-dense for testing.

```
**Additional Cursor Prompt:**
```

Also create data/corpus/example_biology.txt with:

A 1500-word text about "CRISPR Gene Editing" including:

- Scientists and their institutions
- Discoveries and applications
- Ethical considerations
- Technical processes

This will test multi-file corpus loading from config.

```
---

## 🎯 HANDS-ON WORKFLOW IN CURSOR

### Session 1: Infrastructure (45 min)
1. Start with Phase 1 prompts → Get project structure
2. **Run in terminal:** `cd knowledge-graph-builder && uv sync`
3. Run Phase 2 prompts → Build core utilities
4. **Test**: Try `uv run python -c "from src.config import Config; print(Config())"` - check no errors
5. **Checkpoint**: `git init && git add . && git commit -m "Initial setup with uv"`

### Session 2: Agents (60 min)
1. Build Analyzer → Test with small text
2. Build Splitter → Test with sample corpus
3. Build Chunker → Test with one section
4. **Test**: `uv run python -m pytest tests/test_analyzer.py`
5. **Checkpoint**: `git commit -am "Add analyzer, splitter, chunker"`

### Session 3: Extraction (60 min)
1. Build Extractor → Test with one chunk
2. Build Reviewer → Test with sample triples
3. Test parallel extraction with 10 chunks
4. **Checkpoint**: `git commit -am "Add extractor and reviewer"`

### Session 4: Integration (45 min)
1. Build LangGraph workflow
2. Create main.py with corpus loading from config
3. Add sample corpus files to data/corpus/
4. Run end-to-end: `uv run main.py`
5. Fix bugs, tune prompts
6. **Done**: Full pipeline working!

---

## 💡 CURSOR POWER TIPS

### Use Cursor's Composer Mode
- Select multiple files, ask: "Make these work together"
- Example: "Connect analyzer_node.py to openrouter_client.py"

### Iterate with Chat
```

After generating code: "Add error handling for network timeouts" "Make this function async" "Add type hints and docstrings" "Write a unit test for this"

```
### Use @-mentions
```

"@analyzer_node.py @data_models.py ensure the analyzer returns the correct GraphState format"

```
### Debug with Cursor
```

"This function fails with KeyError. Fix it and explain why." "Optimize this for better performance." "Make this code more Pythonic."

```
---

## 🐛 COMMON ISSUES & FIXES

### Issue 1: LangGraph State Not Updating
**Cursor Prompt:**
```

My analyzer_agent is not updating the state correctly. Debug src/agents/analyzer_node.py and ensure it returns a properly updated GraphState dict.

```
### Issue 2: Parallel Extraction Too Slow
**Cursor Prompt:**
```

Optimize src/agents/extractor_node.py to:

1. Limit concurrent requests to 10
2. Use asyncio.Semaphore
3. Add batch processing

```
### Issue 3: Memgraph Connection Fails
**Cursor Prompt:**
```

Fix src/storage/memgraph_client.py connection handling:

1. Add retry logic with exponential backoff
2. Test connection on initialization
3. Provide clear error messages

```
---

## 📊 VALIDATION CHECKLIST

After building, verify:

- [ ] UV is installed: `uv --version`
- [ ] Dependencies installed: `uv sync` completes successfully
- [ ] All imports work: `uv run python -c "from src import config, models"`
- [ ] Config loads from .env: Check OPENROUTER_API_KEY is set
- [ ] Corpus files load: `config.get_corpus_files()` returns files from data/corpus/
- [ ] OpenRouter client returns responses
- [ ] Memgraph client connects: `docker ps` shows memgraph running
- [ ] Analyzer produces extraction strategy
- [ ] Splitter creates logical sections
- [ ] Chunker overlaps chunks correctly
- [ ] Extractor runs in parallel and extracts triples
- [ ] Reviewer reduces duplicates
- [ ] Storage writes to Memgraph
- [ ] Full pipeline completes: `uv run main.py` succeeds
- [ ] Can query graph in Memgraph Lab (http://localhost:3000)
- [ ] Multiple corpus files processed correctly

---

## 🚀 NEXT STEPS AFTER BASIC BUILD

**Cursor Prompts for Enhancements:**
```

"Add a web UI using Streamlit to upload corpus files to data/corpus/ and visualize the knowledge graph extraction progress in real-time"

```

```

"Implement incremental updates: add new corpus files to data/corpus/ and process only new documents without re-processing existing graph"

```

```

"Add entity linking: connect extracted entities to Wikidata IDs"

```

```

"Create a REST API using FastAPI to expose the pipeline as a service. Accept corpus files via POST endpoint and return graph statistics"

```

```

"Add corpus management CLI commands:

- uv run python -m src.cli add-corpus <file>
- uv run python -m src.cli list-corpus
- uv run python -m src.cli clear-corpus"

```
---

## 🛠️ UV-SPECIFIC TIPS

### Adding New Dependencies
```bash
# Add a new package
uv add numpy

# Add dev dependency
uv add --dev pytest-cov

# Update all packages
uv sync --upgrade
```

### Running Commands

```bash
# Run main script
uv run main.py

# Run tests
uv run pytest

# Run specific Python command
uv run python -c "from src.config import Config; print(Config())"

# Enter virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Troubleshooting UV

```bash
# Clear cache and reinstall
uv cache clean
uv sync --reinstall

# Check what's installed
uv pip list

# Lock file out of sync
uv lock
uv sync
```

------

## 📚 LEARNING RESOURCES

While coding in Cursor, keep these open:

- **LangGraph Docs**: langchain-ai.github.io/langgraph
- **OpenRouter Models**: openrouter.ai/models
- **Memgraph Docs**: memgraph.com/docs
- **Pydantic**: docs.pydantic.dev

------

*Use Cursor's CMD+K (Mac) or CTRL+K (Windows) to chat about any code you generate. It will understand your entire project context!*

------

## 📁 QUICK REFERENCE: .env.example

```bash
# OpenRouter Configuration
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Memgraph Configuration
MEMGRAPH_URI=bolt://localhost:7687
MEMGRAPH_USER=
MEMGRAPH_PASSWORD=

# Corpus Configuration
CORPUS_DIRECTORY=data/corpus
CORPUS_FILES=["data/corpus/sample.txt"]
INPUT_DIRECTORY=data/input
OUTPUT_DIRECTORY=data/output

# Model Configuration
ANALYZER_MODEL=meta-llama/llama-3.1-70b-instruct:free
SPLITTER_MODEL=google/gemini-flash-1.5
CHUNKER_MODEL=google/gemini-flash-1.5
EXTRACTOR_MODEL=meta-llama/llama-3.1-8b-instruct:free
REVIEWER_MODEL=meta-llama/llama-3.1-70b-instruct:free

# Processing Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=0.15
MAX_PARALLEL_EXTRACTIONS=10
BATCH_SIZE=5
```

Copy this to `.env` and fill in your values!
# Knowledge Graph Builder Architecture

## Multi-Agent System with LangGraph

------

## System Overview

A LangGraph-based multi-agent system that transforms unstructured text corpora into structured knowledge graphs stored in Memgraph. The system uses specialized agents working in a coordinated pipeline to analyze, chunk, extract, and validate knowledge triples.

------

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT CORPUS                             │
│                      (Text Documents)                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  ENTRY POINT   │
                    │  (LangGraph)   │
                    └────────┬───────┘
                             │
                             ▼
        ╔════════════════════════════════════════════════╗
        ║           ANALYZER NODE (Agent 1)              ║
        ║  • Reads entire corpus                         ║
        ║  • Identifies domain & key concepts            ║
        ║  • Generates extraction strategy               ║
        ║  • Creates optimized prompts                   ║
        ║  Model: meta-llama/llama-3.1-70b-instruct     ║
        ╚════════════════════════════════════════════════╝
                             │
                             ▼
        ╔════════════════════════════════════════════════╗
        ║           SPLITTER NODE (Agent 2)              ║
        ║  • Divides corpus into sections                ║
        ║  • Identifies logical boundaries               ║
        ║  • Creates section metadata                    ║
        ║  • Enables parallel processing                 ║
        ║  Model: google/gemini-flash-1.5               ║
        ╚════════════════════════════════════════════════╝
                             │
                             ▼
        ╔════════════════════════════════════════════════╗
        ║           CHUNKER NODE (Agent 3)               ║
        ║  • Breaks sections into chunks                 ║
        ║  • Maintains semantic coherence                ║
        ║  • Applies overlapping strategy                ║
        ║  • Optimizes for context windows               ║
        ║  Model: google/gemini-flash-1.5               ║
        ╚════════════════════════════════════════════════╝
                             │
                             ▼
        ╔════════════════════════════════════════════════╗
        ║         EXTRACTOR NODE (Agent 4)               ║
        ║  • Processes chunks in PARALLEL                ║
        ║  • Extracts (subject, predicate, object)       ║
        ║  • Uses analyzer-generated prompts             ║
        ║  • Outputs structured triples                  ║
        ║  Model: meta-llama/llama-3.1-8b-instruct      ║
        ║  Instances: Multiple (parallel execution)      ║
        ╚════════════════════════════════════════════════╝
                             │
                             ▼
        ╔════════════════════════════════════════════════╗
        ║          REVIEWER NODE (Agent 5)               ║
        ║  • Validates extracted triples                 ║
        ║  • Detects & merges duplicates                 ║
        ║  • Corrects inconsistencies                    ║
        ║  • Normalizes entity names                     ║
        ║  Model: meta-llama/llama-3.1-70b-instruct     ║
        ╚════════════════════════════════════════════════╝
                             │
                             ▼
        ╔════════════════════════════════════════════════╗
        ║         STORAGE NODE (Final Step)              ║
        ║  • Connects to Memgraph                        ║
        ║  • Creates nodes & relationships               ║
        ║  • Builds knowledge graph                      ║
        ║  • Indexes for efficient querying              ║
        ╚════════════════════════════════════════════════╝
                             │
                             ▼
                    ┌────────────────┐
                    │      END       │
                    └────────────────┘
```

------

## State Management

### GraphState Schema

```python
class GraphState(TypedDict):
    # Input
    corpus: str
    corpus_metadata: Dict
    
    # Analyzer outputs
    extraction_strategy: str
    extraction_prompts: Dict[str, str]
    domain_context: str
    
    # Splitter outputs
    sections: List[Section]
    section_metadata: List[Dict]
    
    # Chunker outputs
    chunks: List[Chunk]
    chunk_mapping: Dict[str, str]  # chunk_id -> section_id
    
    # Extractor outputs
    raw_triples: List[Triple]
    extraction_stats: Dict
    
    # Reviewer outputs
    validated_triples: List[Triple]
    corrections_made: List[Dict]
    
    # Storage outputs
    graph_stats: Dict
    storage_status: str
```

------

## Agent Specifications

### 1. Analyzer Agent

**Purpose**: Understand corpus and create extraction strategy
 **Input**: Raw text corpus
 **Output**: Extraction strategy, domain context, specialized prompts
 **Model**: Llama 3.1 70B (complex reasoning)
 **Key Tasks**:

- Identify corpus domain (scientific, news, literature, etc.)
- Detect key entity types and relationships
- Generate context-aware extraction prompts
- Define entity normalization rules

### 2. Splitter Agent

**Purpose**: Divide corpus for parallel processing
 **Input**: Corpus + extraction strategy
 **Output**: Independent sections with metadata
 **Model**: Gemini Flash 1.5 (fast, efficient)
 **Key Tasks**:

- Identify logical boundaries (chapters, topics, documents)
- Maintain structural integrity
- Create section metadata (topic, entity count estimates)
- Balance section sizes for parallelism

### 3. Chunker Agent

**Purpose**: Create optimal context windows
 **Input**: Sections from splitter
 **Output**: Semantic chunks with overlap
 **Model**: Gemini Flash 1.5 (fast, efficient)
 **Key Tasks**:

- Chunk at sentence/paragraph boundaries
- Apply 10-20% overlap between chunks
- Maintain semantic coherence
- Tag chunks with section references

### 4. Extractor Agent

**Purpose**: Extract knowledge triples
 **Input**: Chunks + extraction prompts
 **Output**: Raw (subject, predicate, object) triples
 **Model**: Llama 3.1 8B (balanced speed/quality)
 **Key Tasks**:

- Extract entities and relationships
- Apply domain-specific prompts
- Process multiple chunks in parallel
- Output structured JSON triples

### 5. Reviewer Agent

**Purpose**: Validate and refine triples
 **Input**: Raw triples from all chunks
 **Output**: Validated, deduplicated triples
 **Model**: Llama 3.1 70B (high-quality reasoning)
 **Key Tasks**:

- Merge duplicate entities (e.g., "NYC" → "New York City")
- Validate triple consistency
- Correct extraction errors
- Normalize entity representations

------

## Data Flow & Parallelization

### Sequential Stages

1. Analyzer → Single pass over corpus
2. Splitter → Divides into N sections
3. Chunker → Creates M chunks per section

### Parallel Stage

1. Extractor → **Processes all M×N chunks simultaneously**
   - Uses LangGraph's parallel execution
   - Each chunk processed independently
   - Results aggregated automatically

### Final Sequential Stages

1. Reviewer → Validates all triples
2. Storage → Writes to Memgraph

------

## Technology Stack

| Component        | Technology         | Purpose                        |
| ---------------- | ------------------ | ------------------------------ |
| Framework        | LangGraph          | Agent orchestration & workflow |
| LLM Provider     | OpenRouter         | Free API access to models      |
| State Management | LangGraph State    | Pass data between agents       |
| Parallelization  | LangGraph Parallel | Concurrent extraction          |
| Graph Database   | Memgraph           | Knowledge graph storage        |
| Protocol         | Neo4j Bolt         | Memgraph connectivity          |
| Language         | Python 3.9+        | Implementation                 |

------

## Model Selection Strategy

| Agent     | Model            | Reasoning                              |
| --------- | ---------------- | -------------------------------------- |
| Analyzer  | Llama 3.1 70B    | Needs deep understanding & strategy    |
| Splitter  | Gemini Flash 1.5 | Fast structural analysis               |
| Chunker   | Gemini Flash 1.5 | Speed matters, simple task             |
| Extractor | Llama 3.1 8B     | Balanced quality/speed, many instances |
| Reviewer  | Llama 3.1 70B    | Quality critical for validation        |

**Cost Optimization**: Heavy models (70B) for strategy & validation, lighter models (8B, Flash) for repetitive tasks.

------

## Error Handling & Resilience

### LangGraph Features

- **Checkpointing**: Save state after each node
- **Retry Logic**: Automatic retry on API failures
- **Conditional Edges**: Route based on agent outputs
- **Human-in-the-loop**: Optional manual review steps

### Fallback Strategies

- If Extractor fails on chunk → Skip and log
- If Reviewer detects >30% errors → Re-extract problematic chunks
- If Storage fails → Save triples to JSON, retry later

------

## Performance Characteristics

### Expected Throughput

- **10,000 word corpus**: ~3-5 minutes
- **100,000 word corpus**: ~15-25 minutes
- **1,000,000 word corpus**: ~2-3 hours

### Bottlenecks

1. **Analyzer**: Single-threaded, complex (1-2 min)
2. **Extractor**: Parallel, but many API calls (bulk of time)
3. **Reviewer**: Single-threaded, complex (2-3 min)

### Optimization Opportunities

- Cache analyzer prompts for similar corpora
- Batch extractor API calls (10-20 chunks per call)
- Pre-filter chunks unlikely to contain triples

------

## Output: Knowledge Graph in Memgraph

### Node Types

- **Entity**: Represents subjects/objects from triples
- **Properties**: name, type, source_chunk_ids, confidence

### Relationship Types

- Dynamic based on predicates (e.g., "WORKS_AT", "LOCATED_IN")
- **Properties**: predicate, source_chunk_id, confidence

### Visualization

- Memgraph Lab for interactive exploration
- Cypher queries for analysis
- Export to formats: JSON, GraphML, CSV

------

## Extension Points

### Future Enhancements

1. **Coreference Resolution Agent**: Link pronouns to entities
2. **Entity Linking Agent**: Connect to knowledge bases (Wikidata)
3. **Temporal Agent**: Extract time-based relationships
4. **Confidence Scoring**: Add ML-based quality scores
5. **Incremental Updates**: Add new documents to existing graph

------

*This architecture balances speed, cost, and quality using free OpenRouter models in a robust LangGraph pipeline.*
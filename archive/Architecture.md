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
        ║  • NEW: Quick boundary-based splitting         ║
        ║    - Detects chapter/section headers           ║
        ║    - Finds paragraph boundaries                ║
        ║    - No resource-intensive embeddings          ║
        ║  • Fallback: Semantic splitting with embeddings║
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
        ║  • Uses OpenIE methodology with 2-phase        ║
        ║    extraction: entities → relations            ║
        ║  • Schemaless extraction (no ontologies)       ║
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
- NEW: Sample picker for large corpuses
  - Extracts representative samples from different sections
  - Analyzes samples to understand document structure
  - Creates optimized prompts based on sample analysis
  - Especially effective for textbooks with clear chapter boundaries
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
- NEW: Quick boundary-based splitting for large documents
  - Detects chapter/section headers (capitalized, short lines)
  - Finds paragraph boundaries (double newlines)
  - No resource-intensive embeddings required
  - Much faster for textbook-sized documents
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

**Purpose**: Extract knowledge triples using OpenIE methodology
 **Input**: Chunks + extraction prompts
 **Output**: Raw (subject, predicate, object) triples
 **Model**: Llama 3.1 8B (balanced speed/quality)
 **Key Tasks**:

- **OpenIE Two-Phase Extraction**:
  1. **Entity Extraction**: Identify named entities and key concepts
  2. **Relation Extraction**: Extract relationships using entities as context
- **Schemaless Approach**: No predefined ontologies or relation schemas required
- **Domain-Independent**: Works across any subject matter without domain-specific training
- Apply analyzer-generated prompts for context-specific extraction
- Process multiple chunks in parallel with rate limiting
- Output structured JSON triples with fallback mechanisms

### 5. Reviewer Agent

**Purpose**: Validate and refine triples
 **Input**: Raw triples from all chunks
 **Output**: Validated, deduplicated triples
 **Model**: Llama 3.1 70B (high-quality reasoning)
 **Key Tasks**:

- Merge duplicate entities (e.g., "NYC" → "New York City")
- Validate triple consistency
- Correct extraction errors
- **Enhanced Entity Normalization**: LLM-based shortening to ensure entities are ≤ 3 words
  - Two-stage approach: Rule-based normalization first, then LLM-based shortening for long entities
  - Batch processing for efficient API usage
  - Semantic preservation: Maintains meaning while shortening
  - Preserve acronyms (LLMs, RAG, NER, etc.)
  - Improve graph readability and visualization
  - Configurable via `max_entity_words` and `use_llm_entity_shortening` settings
- Normalize entity representations

### 6. Enhanced Entity Normalization

**Purpose**: Improve graph readability and usability through intelligent entity shortening to maximum 3 words

**Key Features**:
- **LLM-Based Shortening**: Uses language models to semantically shorten entities exceeding word limit
- **Two-Stage Process**: 
  1. Rule-based normalization (fast, handles common phrases)
  2. LLM-based shortening for remaining long entities (semantic-aware)
- **Batch Processing**: Processes multiple entities in single API call for efficiency
- **Caching**: Avoids re-processing duplicate entities
- **Semantic preservation**: All meaning maintained during shortening
- **Smart acronym handling**: Correctly preserves LLMs, RAG, NER, etc.
- **Configurable**: `max_entity_words` (default: 3) and `use_llm_entity_shortening` (default: true)
- **Improved visualization**: More compact and readable knowledge graphs

**Examples**:
- "Questions That Require Connecting Information From Multiple Sources" → "Multi-Source Queries" or "Complex Queries"
- "Computational Cost Of Building And Maintaining Knowledge Graphs" → "Knowledge Graph Costs"
- "Underlying Knowledge Representation Structure" → "Knowledge Structure"
- "Several Challenges In Implementation And Deployment" → "Implementation Challenges"
- "Large Language Models" → "LLMs"
- "Rich Relational Structure" → "Relational Structure"
- "Traditional RAG Approaches" → "Traditional RAG"

**Implementation Details**:
- Processes entities in batches of 20 for optimal API usage
- Falls back to rule-based normalization if LLM fails
- Logs entity length statistics for monitoring
- Ensures all output entities are ≤ `max_entity_words` (default: 3)

**Benefits**:
- Better graph visualization and navigation (shorter labels)
- Enhanced user experience when exploring knowledge graphs
- Improved semantic space construction (consistent entity lengths)
- Reduced storage requirements for entity labels
- Maintained semantic accuracy and relationships

### 7. Splitting Strategy

**Quick Boundary-Based Splitting (Default)**:
- **Purpose**: Fast splitting for large textbooks and documents
- **Method**: Analyzes document structure to find natural boundaries
- **Benefits**: No embeddings required, much faster processing
- **Triggers**: When `use_quick_splitting=true` (default)

**Fallback Chain**:
1. **Primary**: Quick boundary-based splitting
2. **Fallback 1**: Semantic splitting with embeddings and clustering
3. **Fallback 2**: LLM-based splitting with specialized prompts
4. **Final fallback**: Character-based splitting

**Boundary Detection**:
- Chapter/section headers (capitalized, short lines)
- Paragraph boundaries (double newlines)
- Smart sentence boundary preservation
- Content preservation verification

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
   - Normalizes entities (rule-based + LLM-based shortening)
   - Deduplicates triples
   - Validates consistency with LLM
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
| Splitter  | Gemini Flash 1.5 | Fast structural analysis (NEW: boundary detection) |
| Chunker   | Gemini Flash 1.5 | Speed matters, simple task             |
| Extractor | Llama 3.1 8B     | Balanced quality/speed, many instances |
| Reviewer  | Llama 3.1 70B    | Quality critical for validation        |

**Cost Optimization**: Heavy models (70B) for strategy & validation, lighter models (8B, Flash) for repetitive tasks.

**NEW: Splitting Strategy**:
- **Primary**: Quick boundary-based splitting (no LLM required)
- **Fallback**: Gemini Flash 1.5 for LLM-based splitting when boundaries unclear
- **Cost**: Zero LLM cost for most textbook splitting scenarios

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

### Enhanced Entity Normalization Benefits

- **Improved Readability**: All entities shortened to ≤ 3 words (configurable via `max_entity_words`)
- **LLM-Based Semantic Shortening**: Intelligent shortening that preserves meaning
- **Better Visualization**: More compact graph display with consistent entity lengths
- **Enhanced Usability**: Easier to understand relationships and navigate graphs
- **Preserved Semantics**: All meaning maintained during shortening through semantic-aware LLM processing
- **Smart Acronym Handling**: Correctly preserves LLMs, RAG, NER, etc.
- **Efficient Processing**: Batch processing reduces API calls and costs

### Quick Splitting Benefits

- **Speed**: No embeddings = 10x faster for large documents
- **Memory**: No embedding models loaded = 50% lower memory usage
- **Scalability**: Can handle textbook-sized documents efficiently
- **Reliability**: Multiple fallback options ensure robustness

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

**NEW: Enhanced for Large Textbooks**
- Quick boundary-based splitting eliminates embedding overhead
- Sample picker enables efficient analysis of large corpuses
- Multiple fallback strategies ensure reliability
- Optimized for textbook-sized documents with clear chapter structures

**NEW: LLM-Based Entity Shortening**
- Intelligent semantic shortening ensures all entities are ≤ 3 words
- Two-stage normalization: rule-based first, then LLM for long entities
- Batch processing for efficient API usage
- Configurable via `max_entity_words` and `use_llm_entity_shortening` settings
- Improves graph visualization and semantic space construction
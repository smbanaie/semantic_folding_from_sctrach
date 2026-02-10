# Cypher Query Guide for Knowledge Graph Visualization

## Cypher Quick Tutorial

**Cypher** is Neo4j's query language for graphs, used by Memgraph. Here are the basics:

### Core Concepts
- **Nodes**: `(n:Entity)` - entities like "RAG", "LLMs"
- **Relationships**: `[r:PREDICATE]` - connections like "COMBINE", "REPRESENTS"
- **Patterns**: `(subject)-[relationship]->(object)`

### Basic Syntax
- `MATCH` - find patterns in the graph
- `WHERE` - filter results
- `RETURN` - specify what to return
- `LIMIT` - restrict number of results
- `ORDER BY` - sort results

### Common Patterns
```cypher
-- Find all nodes
MATCH (n) RETURN n;

-- Find relationships between nodes
MATCH (s)-[r]->(o) RETURN s, r, o;

-- Filter by property
MATCH (n) WHERE n.name CONTAINS "RAG" RETURN n;

-- Count things
MATCH (n) RETURN count(n) AS total_nodes;
```

---

## Memgraph Quick Queries

Use these example Cypher queries in **Memgraph Lab** (`http://localhost:3000`) after importing triples with:

```bash
uv run python graph_importer.py data/output/triples.json
```

## Essential Queries

### Quick Graph Statistics
```cypher
-- Count all nodes and relationships
MATCH (n) RETURN count(n) AS total_nodes;
MATCH ()-[r]->() RETURN count(r) AS total_relationships;

-- Get basic graph info
CALL graph_info.info() YIELD nodes, edges RETURN nodes, edges;
```

### Explore Triples and Relationships
```cypher
-- View entire graph (sample)
MATCH (s)-[r]->(o) RETURN s, r, o LIMIT 50;

-- Show triples as table with properties
MATCH (s)-[r]->(o)
RETURN s.name AS subject, type(r) AS predicate, o.name AS object, r.confidence AS confidence
ORDER BY r.confidence DESC
LIMIT 25;

-- Find most common relationship types
MATCH ()-[r]->()
RETURN type(r) AS relationship_type, count(*) AS frequency
ORDER BY frequency DESC
LIMIT 15;
```

### Entity Analysis
```cypher
-- Find most connected entities (highest degree)
MATCH (n)-[r]-()
RETURN n.name, count(r) AS connections
ORDER BY connections DESC LIMIT 10;

-- Explore relationships for specific entity
MATCH (s)-[r]->(o)
WHERE s.name CONTAINS "RAG"
RETURN s, r, o LIMIT 20;

-- Find entities with highest confidence relationships
MATCH (s)-[r]->(o)
RETURN DISTINCT s.name, max(r.confidence) AS max_confidence
ORDER BY max_confidence DESC LIMIT 10;
```

### Advanced Path Finding
```cypher
-- Shortest path between two entities
MATCH p=shortestPath(
  (a {name:'RAG'})-[*]-(b {name:'LLMs'})
)
RETURN p;

-- Find all paths up to length 3
MATCH p=(a {name:'RAG'})-[*1..3]-(b)
RETURN p LIMIT 10;

-- Find subgraph around an entity (2-hop neighborhood)
MATCH (center {name:'RAG'})-[*1..2]-(neighbor)
RETURN center, neighbor LIMIT 50;
```

### Performance Optimization
```cypher
-- Create index on node names for faster lookups
CREATE INDEX ON :Entity(name);

-- Show existing indexes
SHOW INDEX INFO;

-- Profile query performance
PROFILE MATCH (n)-[r]->() RETURN count(r);
```

---

## Visualization Guide

### Getting Started with Memgraph Lab

1. **Import your triples** using the graph importer:
   ```bash
   uv run python graph_importer.py data/output/triples.json
   ```

2. **Open Memgraph Lab**: Visit `http://localhost:3000` in your browser

3. **Run queries** in the query editor and click **Graph** tab for visualization

### Graph vs Table View

The **Graph** tab in Memgraph Lab only works when your query returns actual `Node` and `Relationship` objects.

**❌ Won't show graph (table only):**
```cypher
MATCH (s)-[r]->(o)
RETURN s.name, type(r), o.name  -- Returns strings only
```

**✅ Will show graph:**
```cypher
MATCH (s)-[r]->(o)
RETURN s, r, o  -- Returns graph objects
```

### Tips for Better Visualization

- **Limit results** for performance: `LIMIT 50` for exploration
- **Use focused queries** instead of full graph queries
- **Create indexes** for faster neighborhood exploration
- **Increase limits** in Lab settings if needed (gear icon → visualization settings)

### Memgraph Lab Features

- **Query Editor**: Write and execute Cypher queries
- **Graph Canvas**: Interactive visualization of nodes and relationships
- **Table View**: Tabular results for data inspection
- **Query History**: Access previous queries
- **Settings**: Customize visualization limits and appearance

### Starting Memgraph (if not running)

If you're using Docker Compose (recommended):
```bash
docker-compose up -d
```

Or standalone Memgraph + Lab:
```bash
docker run -d -p 7687:7687 -p 3000:3000 memgraph/memgraph-mage:latest
```

### Troubleshooting

- **Can't connect?** Ensure Memgraph is running on `bolt://localhost:7687`
- **Graph tab disabled?** Check that query returns `Node`/`Relationship` objects
- **Slow performance?** Create indexes and use smaller result limits
- **Large graphs?** Use focused neighborhood queries instead of full graph

This guide covers the most common operations for exploring your knowledge graph. Start with the statistics queries, then explore specific entities and relationships that interest you!

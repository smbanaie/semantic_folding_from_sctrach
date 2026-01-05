**Memgraph Quick Queries**

Use these example Cypher queries in Memgraph Lab (http://localhost:3000/lab/query) after importing `data/output/triples_YYYYMMDD_HHMMSS.json`.

- **Quick Checks:**

```cypher
MATCH (n:Entity) RETURN count(n) AS entities;
MATCH ()-[r:REL]->() RETURN count(r) AS relationships;
```

- **Show a sample of triples:**

```cypher
MATCH (s:Entity)-[r:REL]->(o:Entity)
RETURN s.name AS subject, r.predicate AS predicate, o.name AS object, r.confidence
LIMIT 50;
```

- **Top predicates:**

```cypher
MATCH ()-[r:REL]->()
RETURN r.predicate AS predicate, count(*) AS cnt
ORDER BY cnt DESC
LIMIT 20;
```

- **Inspect neighborhood of an entity (example: `GraphRAG`):**

```cypher
MATCH (g:Entity {name: 'GraphRAG'})-[r]->(o)
RETURN g, r, o
LIMIT 200;
```

- **Shortest path between two entities:**

```cypher
MATCH p=shortestPath(
  (a:Entity {name:'GraphRAG'})-[*]-(b:Entity {name:'Traditional RAG Approaches'})
)
RETURN p;
```

- **Create an index for faster lookups:**

```cypher
CREATE INDEX ON :Entity(name);
```

**Notes**

- The loader script (`scripts/load_triples_to_memgraph.py`) by default creates nodes with label `:Entity` and relationships as `:REL` with a `predicate` property. If you ran the loader with `--use-relationship-type` it may create relationship types derived from the predicate.
- If Memgraph Lab is not reachable at `http://localhost:3000`, ensure the service is running and port `3000` is exposed (Docker example below).

Docker run example to start Memgraph with Lab exposed:

```bash
docker run -p 7687:7687 -p 3000:3000 --name memgraph -it memgraph/memgraph:latest
```

Import example (after starting Memgraph):

```bash
python scripts/load_triples_to_memgraph.py --file data/output/triples_20251228_024650.json --bolt bolt://localhost:7687
```

If you want, I can also add a short README section with screenshots or pre-made queries tailored to your dataset—tell me which queries you want saved as favorites in Lab.

---

## Visualization Guide

If the **Graph** tab in Memgraph Lab is disabled after running a query, it's usually because the result set contains only scalar values (strings, numbers) instead of node/relationship objects. The Lab graph visualizer requires actual graph objects to render nodes and edges.

- Why your CSV-like result appears but Graph tab is disabled

  Example scalar-returning query you ran:

  ```cypher
  MATCH (s:Entity)-[r:REL]->(o:Entity)
  RETURN s.name AS subject, r.predicate AS predicate, o.name AS object, r.confidence
  LIMIT 50;
  ```

  This returns plain columns (`subject`, `predicate`, `object`, `confidence`) — Lab shows them as a table (CSV-like) but cannot enable the Graph view because there are no `Node` / `Relationship` objects in the result.

- How to enable the Graph view

  Return the actual node and relationship objects instead of their properties:

  ```cypher
  MATCH (s:Entity)-[r:REL]->(o:Entity)
  RETURN s, r, o
  LIMIT 50;
  ```

  After running that, click the **Graph** tab in the results pane — Lab will render the returned nodes and edges.

- Helpful variants

  - If you want to label the returned columns while keeping objects:

    ```cypher
    MATCH (s:Entity)-[r:REL]->(o:Entity)
    RETURN s AS subject_node, r AS rel, o AS object_node
    LIMIT 50;
    ```

  - If you need both the table view and the graph view, run two queries: one returning scalars for CSV export, and another returning objects for visualization.

- If the Graph tab is still disabled

  - Reduce the `LIMIT` (Lab may refuse large result sets for visualization).
  - Open Lab settings (gear icon) and increase the `Max nodes` / `Max relationships` visualization limits.
  - Ensure returned values are real `Node`/`Relationship` objects (those come directly from `MATCH` results; if you use custom projections, return the objects explicitly).

- Performance and indexing

  - Create an index on `:Entity(name)` to speed lookups and interactive exploration:

    ```cypher
    CREATE INDEX ON :Entity(name);
    ```

  - Use small, focused queries when exploring the graph (e.g., neighborhood of a single entity) to keep the visualization responsive.

- Example: show neighborhood and then visualize

  ```cypher
  MATCH (g:Entity {name: 'GraphRAG'})-[r]->(o)
  RETURN g, r, o
  LIMIT 200;
  ```

  Click **Graph** to inspect the rendered subgraph.

If you'd like, I will append a short note to `docs/memgraph-queries.md` with a screenshot example or add pre-saved Lab queries — tell me which you'd prefer.

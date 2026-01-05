#!/usr/bin/env python3
"""Load triples JSON into Memgraph (Neo4j Bolt) with a safe loader.

Usage:
  python scripts/load_triples_to_memgraph.py --file path/to/triples.json [--bolt bolt://localhost:7687] [--user USER --password PASS]

Options:
  --test-only    : parse file and print stats, do not connect to DB
  --batch-size N : number of triples per transaction (default 200)

The loader creates `:Entity` nodes and relationships with predicate stored
in a `predicate` property on a generic `:REL` relationship to avoid unsafe
relationship types. Set `--use-relationship-type` to attempt using predicate
as relationship type (sanitizes names but may still be unsafe for arbitrary text).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "triples" in data:
        return data["triples"]
    if isinstance(data, list):
        return data
    raise ValueError("Unexpected JSON format: expected list or { 'triples': [...] }")


def safe_predicate_type(predicate: str) -> str:
    """Create a safe relationship type name from predicate.

    Converts to uppercase letters, digits and underscores only.
    If result starts with a digit, prefix with 'R_'.
    """
    if not predicate:
        return "REL"
    s = predicate.upper()
    # Replace non-alnum with underscore
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    s = s.strip("_")
    if not s:
        return "REL"
    if s[0].isdigit():
        s = "R_" + s
    # Limit length
    return s[:50]


def chunked(seq: Iterable, n: int):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    parser = argparse.ArgumentParser(description="Load triples JSON into Memgraph")
    parser.add_argument("--file", "-f", required=True, help="Path to triples.json")
    parser.add_argument("--bolt", default="bolt://localhost:7687", help="Bolt URI")
    parser.add_argument("--user", default="", help="DB user")
    parser.add_argument("--password", default="", help="DB password")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--use-relationship-type", action="store_true", help="Use sanitized predicate as relationship type instead of property")

    args = parser.parse_args()
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    triples = load_json(path)
    print(f"Loaded {len(triples)} triples from {path}")

    # Quick validation
    bad = 0
    for i, t in enumerate(triples):
        if not all(k in t for k in ("subject", "predicate", "object")):
            bad += 1
    if bad:
        print(f"Warning: {bad} entries missing required keys (subject,predicate,object)")

    if args.test_only:
        # Show a small sample
        print("Sample triples:")
        for t in triples[:5]:
            print(t)
        return

    # Connect and insert
    try:
        from neo4j import GraphDatabase
    except Exception as e:
        raise SystemExit("Please install neo4j driver: pip install neo4j") from e

    auth = (args.user, args.password) if args.user or args.password else None
    driver = GraphDatabase.driver(args.bolt, auth=auth)

    def _write_batch(tx, batch: List[Dict[str, Any]]):
        # Use UNWIND to merge nodes and create relationships
        if args.use_relationship_type:
            # Build relationship type dynamically using parameterized cypher is not supported,
            # so we create them via APOC or dynamic cypher; for safety we fall back to property method.
            cypher = """
UNWIND $batch AS t
MERGE (s:Entity {name: t.subject})
MERGE (o:Entity {name: t.object})
WITH s,o,t
CALL apoc.do.when(
  t.predicate IS NOT NULL,
  'WITH s,o,t CALL apoc.merge.relationship(s, $rtype, {}, {}, o) YIELD rel RETURN rel',
  'RETURN null as rel',
  {s:s,o:o,t:t,rtype: t._reltype}
)
YIELD value
RETURN count(*)
"""
            # Prepare batch with sanitized rel type
            for item in batch:
                item["_reltype"] = safe_predicate_type(item.get("predicate", ""))
            tx.run(cypher, batch=batch)
        else:
            cypher = """
UNWIND $batch AS t
MERGE (s:Entity {name: t.subject})
MERGE (o:Entity {name: t.object})
MERGE (s)-[r:REL {predicate: t.predicate}]->(o)
SET r.confidence = coalesce(t.confidence, 1.0), r.source_chunk_id = t.source_chunk_id
RETURN count(r) AS created
"""
            tx.run(cypher, batch=batch)

    # Insert in batches
    total = 0
    with driver.session() as session:
        for batch in chunked(triples, args.batch_size):
            # neo4j Python driver v5+: use execute_write / execute_read
            try:
                session.execute_write(_write_batch, batch)
            except AttributeError:
                # Fallback for older driver versions that use write_transaction
                session.write_transaction(_write_batch, batch)
            total += len(batch)
            print(f"Inserted {total}/{len(triples)}")

    driver.close()
    print("Done")


if __name__ == "__main__":
    main()

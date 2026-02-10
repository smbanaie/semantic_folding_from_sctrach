#!/usr/bin/env python3
"""
Knowledge Graph Importer

Load triples from JSON files into Memgraph for visualization.

Usage:
    python graph_importer.py data/output/triples.json
    python graph_importer.py --help

Features:
- Load triples from JSON array format
- Clear existing graph before import
- Show import statistics
- Provide visualization instructions
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from src.models.data_models import Triple
from src.storage.memgraph_client import MemgraphClient


def load_triples_from_json(json_file_path: str) -> List[Triple]:
    """Load triples from JSON file."""
    json_path = Path(json_file_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Triples file not found: {json_file_path}")

    print(f"Loading triples from {json_file_path}...")

    with json_path.open('r', encoding='utf-8') as f:
        raw_triples = json.load(f)

    if not isinstance(raw_triples, list):
        raise ValueError("JSON file must contain an array of triples")

    triples = []
    for i, item in enumerate(raw_triples):
        try:
            triple = Triple(
                subject=item['subject'],
                predicate=item['predicate'],
                object=item['object'],
                confidence=item.get('confidence', 1.0),
                source_chunk_id=item.get('source_chunk_id', ''),
                metadata=item.get('metadata', {})
            )
            triples.append(triple)
        except KeyError as e:
            raise ValueError(f"Triple {i} missing required field: {e}")

    print(f"[OK] Loaded {len(triples)} triples from JSON")
    return triples


def import_to_memgraph(triples: List[Triple]) -> None:
    """Import triples into Memgraph."""
    print("\nConnecting to Memgraph...")

    try:
        with MemgraphClient() as client:
            print("[OK] Connected to Memgraph")

            print("Clearing existing graph...")
            client.clear_graph()
            print("[OK] Graph cleared")

            print(f"Importing {len(triples)} triples...")
            client.bulk_insert_triples(triples)
            print("[OK] Triples imported")

            # Get statistics
            stats = client.get_stats()
            print("\n[SUCCESS] Import Complete!")
            print(f"   - Nodes: {stats['node_count']}")
            print(f"   - Relationships: {stats['edge_count']}")

    except Exception as e:
        print(f"[ERROR] Error importing to Memgraph: {e}")
        print("Make sure Memgraph is running: docker-compose up")
        sys.exit(1)


def show_visualization_guide():
    """Show visualization and querying instructions."""
    print("\n" + "=" * 70)
    print("VISUALIZATION & QUERYING GUIDE")
    print("=" * 70)

    print("\nWEB INTERFACE:")
    print("   Open: http://localhost:3000")
    print("   Use Memgraph Lab for interactive graph exploration")

    print("\nCOMMON CYPHER QUERIES:")
    print("\n   1. VIEW ENTIRE GRAPH:")
    print("      MATCH (s)-[r]->(o) RETURN s, r, o LIMIT 50;")

    print("\n   2. FIND MOST CONNECTED ENTITIES:")
    print("      MATCH (n)-[r]-()")
    print("      RETURN n.name, count(r) AS connections")
    print("      ORDER BY connections DESC LIMIT 10;")

    print("\n   3. EXPLORE RELATIONSHIPS FOR SPECIFIC ENTITY:")
    print("      MATCH (s)-[r]->(o)")
    print("      WHERE s.name CONTAINS 'RAG'")
    print("      RETURN s, r, o;")

    print("\n   4. GET GRAPH STATISTICS:")
    print("      MATCH (n) RETURN count(n) AS nodes;")
    print("      MATCH ()-[r]->() RETURN count(r) AS relationships;")

    print("\n   5. FIND RELATIONSHIPS BY TYPE:")
    print("      MATCH (s)-[r]->(o)")
    print("      WHERE type(r) = 'COMBINE'")
    print("      RETURN s, r, o LIMIT 20;")

    print("\n   6. ANALYZE CONFIDENCE SCORES:")
    print("      MATCH (s)-[r]->(o)")
    print("      RETURN s.name, type(r), o.name, r.confidence")
    print("      ORDER BY r.confidence DESC LIMIT 25;")

    print("\n   7. FIND PATHS BETWEEN ENTITIES:")
    print("      MATCH path = (a)-[*2]-(b)")
    print("      WHERE a.name = 'RAG' AND b.name = 'LLMs'")
    print("      RETURN path LIMIT 5;")

    print("\nTIPS:")
    print("   - Use LIMIT to avoid large result sets")
    print("   - Use CONTAINS for partial string matching")
    print("   - Use ORDER BY and LIMIT for top-N queries")
    print("   - Use [*n] for path finding (variable length relationships)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import triples from JSON into Memgraph for visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python graph_importer.py data/output/triples.json
  python graph_importer.py --help

The triples JSON should be an array of objects with:
  - subject: string
  - predicate: string
  - object: string
  - confidence: number (optional, defaults to 1.0)
  - source_chunk_id: string (optional)
  - metadata: object (optional)
        """
    )

    parser.add_argument(
        'triples_file',
        help='Path to JSON file containing triples array'
    )

    parser.add_argument(
        '--no-guide',
        action='store_true',
        help='Skip showing visualization guide after import'
    )

    args = parser.parse_args()

    try:
        # Load triples
        triples = load_triples_from_json(args.triples_file)

        # Import to Memgraph
        import_to_memgraph(triples)

        # Show guide
        if not args.no_guide:
            show_visualization_guide()

        print("\n[SUCCESS] Graph import complete! Ready for exploration.")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
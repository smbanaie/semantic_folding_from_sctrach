#!/usr/bin/env python3
"""
Create a query-text JSONL from raw MuSiQue data, mapped to existing run indices.

The raw MuSiQue JSONL contains full entries with paragraphs. This script
extracts only the question text and maps it to the query indices used in
an existing benchmark run.

Usage:
    .venv\Scripts\python -m semantic_folding.tools.create_query_jsonl \
        --raw-jsonl data/HippoRAG2/dataset/musique/musique_ans_v1.0_dev.jsonl \
        --run-dir outputs/musique_benchmark/runs/run_20260617_022257 \
        --output temp/musique_queries.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import get_logger

logger = get_logger("create_query_jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-jsonl", type=Path, required=True,
                        help="Raw dataset JSONL with 'question' field")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Existing benchmark run directory with query_doc_map.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL with query_idx and question text")
    args = parser.parse_args()

    # Load run's query_doc_map to get candidate doc IDs per query
    with open(args.run_dir / "query_doc_map.json", encoding="utf-8") as f:
        query_doc_map = json.load(f)

    # Load gold labels
    with open(args.run_dir / "query_gold.json", encoding="utf-8") as f:
        gold_labels = json.load(f)

    logger.info(f"Run has {len(query_doc_map)} queries")

    # Load raw JSONL entries
    raw_entries = []
    with open(args.raw_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_entries.append(json.loads(line))
    logger.info(f"Raw data has {len(raw_entries)} entries")

    # For each query in the run, find the matching raw entry
    # The run's query_doc_map[query_idx] = list of candidate doc IDs
    # We need to match by finding the raw entry whose paragraphs match those doc IDs
    output_entries = []

    for q_idx_str, candidate_docs in query_doc_map.items():
        q_idx = int(q_idx_str)
        gold_docs = gold_labels.get(q_idx_str, [])

        # Find the raw entry that contains the gold paragraph(s)
        matched_entry = None
        for entry in raw_entries:
            entry_doc_ids = [p.get("id", "") for p in entry.get("paragraphs", [])]
            if any(gd in entry_doc_ids for gd in gold_docs):
                matched_entry = entry
                break

        if matched_entry is None:
            logger.warning(f"  [{q_idx}] No matching raw entry found for gold docs {gold_docs[:3]}")
            continue

        question = matched_entry.get("question", "")
        output_entries.append({
            "query_idx": q_idx,
            "question": question,
            "answer": matched_entry.get("answer", ""),
            "candidate_docs": candidate_docs,
            "gold_docs": gold_docs,
        })

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in output_entries:
            f.write(json.dumps(entry, default=lambda o: float(o) if hasattr(o, 'item') else o) + "\n")

    logger.info(f"\nCreated {len(output_entries)} query entries")
    logger.info(f"Output: {args.output}")

    # Show first 3
    for e in output_entries[:3]:
        logger.info(f"  [{e['query_idx']}] {e['question'][:70]}...")


if __name__ == "__main__":
    main()

"""
Convert BioASQ13 official format to MuSiQue-like JSONL.

BioASQ13 format:
  - documents: list of PubMed URLs
  - snippets: separate list with {text, document} pairs
  - ideal_answer: list of answer strings

Output: MuSiQue-compatible JSONL with gold/distractor paragraphs.
"""

import json
import random
import sys
from pathlib import Path

random.seed(42)

RAW_DIR = Path("data/bioasq/raw")
OUTPUT_DIR = Path("data/bioasq/converted")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_pmid(url: str) -> str:
    """Extract PMID from PubMed URL."""
    if "pubmed/" in url:
        return url.split("pubmed/")[-1].rstrip("/")
    return url


def load_bioasq13(json_path: Path) -> list:
    """Load BioASQ13 JSON and convert to flat rows."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    rows = []

    for q in questions:
        question_text = q.get("body", "").strip()
        if not question_text:
            continue

        # Get answer
        ideal = q.get("ideal_answer", [])
        if isinstance(ideal, list):
            answer_text = " ".join(ideal).strip()
        else:
            answer_text = str(ideal).strip()

        # Get document URLs
        doc_urls = q.get("documents", [])

        # Get snippets
        snippets = q.get("snippets", [])

        # Build passages from snippets
        passages = []
        seen_texts = set()
        for snip in snippets:
            text = snip.get("text", "").strip()
            doc_url = snip.get("document", "")
            pmid = extract_pmid(doc_url)
            if text and text not in seen_texts:
                seen_texts.add(text)
                passages.append({
                    "title": f"PMID:{pmid}",
                    "text": text,
                })

        # If no snippets, try to use document URLs as placeholders
        if not passages:
            for url in doc_urls[:5]:
                pmid = extract_pmid(url)
                passages.append({
                    "title": f"PMID:{pmid}",
                    "text": f"[Abstract from PMID {pmid}]",
                })

        if not passages:
            continue

        rows.append({
            "id": q.get("id", ""),
            "question": question_text,
            "type": q.get("type", ""),
            "answer": answer_text,
            "context": passages,
            "documents": doc_urls,
        })

    return rows


def collect_all_passages(rows: list) -> list:
    """Collect all passages for distractor sampling."""
    all_passages = []
    for row in rows:
        for p in row.get("context", []):
            text = p.get("text", "").strip()
            if text and len(text) > 20:
                all_passages.append(text)
    return all_passages


def convert_to_musique(rows: list, all_passages: list, max_queries: int = 500) -> list:
    """Convert BioASQ rows to MuSiQue format with gold/distractors.

    Gold detection strategy for BioASQ:
    1. Check if answer text appears verbatim in snippet (exact match)
    2. Check if answer keywords appear in snippet (partial match)
    3. If no snippet matches, mark the first snippet from each document as gold
       (BioASQ documents are curated to be relevant to the answer)
    """
    entries = []
    n_written = 0

    for row in rows:
        if n_written >= max_queries:
            break

        question = row["question"]
        answer_text = row["answer"]
        answer_lower = answer_text.lower()
        doc_urls = row.get("documents", [])

        # Build paragraphs from snippets
        paragraphs = []
        for i, p in enumerate(row["context"]):
            text = p["text"]
            title = p["title"]
            # Exact match
            is_gold = answer_lower and answer_lower in text.lower()
            paragraphs.append({
                "idx": i,
                "title": title,
                "paragraph_text": text,
                "is_supporting": is_gold,
            })

        # Partial keyword match if no exact match found
        if not any(p["is_supporting"] for p in paragraphs) and answer_text:
            # Extract significant keywords from answer (3+ words)
            answer_words = [w for w in answer_lower.split() if len(w) > 3]
            if answer_words:
                best_score = 0
                best_idx = -1
                for i, p in enumerate(paragraphs):
                    snippet_lower = p["paragraph_text"].lower()
                    score = sum(1 for w in answer_words if w in snippet_lower)
                    if score > best_score:
                        best_score = score
                        best_idx = i
                # Mark as gold if at least 30% of answer keywords match
                if best_idx >= 0 and best_score >= len(answer_words) * 0.3:
                    paragraphs[best_idx]["is_supporting"] = True

        # If still no gold, mark all snippets as gold (curated BioASQ documents)
        # BioASQ provides relevant documents for each question
        if not any(p["is_supporting"] for p in paragraphs) and paragraphs:
            for p in paragraphs:
                p["is_supporting"] = True

        # Add distractor passages if fewer than 20
        n_existing = len(paragraphs)
        if n_existing < 20 and all_passages:
            distractor_texts = random.sample(
                all_passages, min(20 - n_existing, len(all_passages))
            )
            for dt in distractor_texts:
                paragraphs.append({
                    "idx": len(paragraphs),
                    "title": "distractor",
                    "paragraph_text": dt,
                    "is_supporting": False,
                })

        entries.append({
            "id": row.get("id", f"bioasq_{n_written:04d}"),
            "question": question,
            "answer": answer_text,
            "qa_type": row.get("type", ""),
            "paragraphs": paragraphs,
        })
        n_written += 1

    return entries


def main():
    print("=" * 60)
    print("BioASQ13 -> MuSiQue Format Converter")
    print("=" * 60)

    # Load training data
    training_path = RAW_DIR / "BioASQ-training13b" / "training13b.json"
    print(f"\nLoading training data: {training_path}")
    rows = load_bioasq13(training_path)
    print(f"  Loaded {len(rows)} questions")

    # Load golden test data
    golden_dir = RAW_DIR / "Task13BGoldenEnriched"
    golden_rows = []
    for gf in sorted(golden_dir.glob("*.json")):
        print(f"  Loading golden test: {gf.name}")
        golden_rows.extend(load_bioasq13(gf))
    print(f"  Total golden test questions: {len(golden_rows)}")

    # Combine (training + test)
    all_rows = rows + golden_rows
    print(f"\nTotal questions: {len(all_rows)}")

    # Collect all passages for distractors
    all_passages = collect_all_passages(all_rows)
    print(f"Total unique passages for distractors: {len(all_passages)}")

    # Convert to MuSiQue format (500 queries max)
    entries = convert_to_musique(all_rows, all_passages, max_queries=500)
    print(f"\nConverted {len(entries)} queries to MuSiQue format")

    # Write JSONL
    out_path = OUTPUT_DIR / "bioasq.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Stats
    gold_counts = [sum(1 for p in e["paragraphs"] if p["is_supporting"]) for e in entries]
    para_counts = [len(e["paragraphs"]) for e in entries]
    stats = {
        "num_queries": len(entries),
        "total_rows_in_source": len(all_rows),
        "avg_paragraphs": sum(para_counts) / len(para_counts),
        "avg_gold_per_query": sum(gold_counts) / len(gold_counts),
        "queries_with_gold": sum(1 for g in gold_counts if g > 0),
    }
    with open(out_path.with_suffix(".stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nOutput: {out_path}")
    print(f"Stats: {json.dumps(stats, indent=2)}")

    # Show sample
    if entries:
        sample = entries[0]
        print(f"\nSample entry:")
        print(f"  ID: {sample['id']}")
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  Answer: {sample['answer'][:100]}...")
        print(f"  Paragraphs: {len(sample['paragraphs'])}")
        print(f"  Gold: {sum(1 for p in sample['paragraphs'] if p['is_supporting'])}")


if __name__ == "__main__":
    main()

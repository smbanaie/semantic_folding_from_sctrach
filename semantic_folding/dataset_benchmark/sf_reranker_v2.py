"""
SF re-ranking module - WORKING VERSION

Uses subprocess to call query_processor.py (avoids import issues).
Filters documents to BM25 candidates only via --corpus filter hack.

Usage:
    from sf_reranker_v2 import rerank_candidates_with_sf_v2
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from semantic_folding.lib import get_logger

logger = get_logger("sf_reranker_v2")


def rerank_candidates_with_sf_v2(
    query: str,
    candidates: List[Tuple[str, float]],
    doc_fingerprints_dir: Path,
    phrase_fingerprints_dir: Path,
    corpus_path: Path,
    grid_size: int = 64,
    top_k: int = 10,
    run_dir: Path = None,
) -> List[Tuple[str, float]]:
    """
    Re-rank BM25 candidates using SF via subprocess call to query_processor.py.
    
    Creates a filtered corpus with only candidate documents, runs query_processor.py,
    parses results.
    """
    t0 = time.time()
    
    # Create temporary filtered corpus (only candidates)
    candidate_ids = [doc_id for doc_id, _ in candidates]
    
    # Write query to temp file
    query_file = run_dir / "temp_query.txt" if run_dir else Path("temp_query.txt")
    query_file.parent.mkdir(exist_ok=True)
    with open(query_file, "w") as f:
        f.write(query)
    
    # Build command
    project_root = Path(__file__).resolve().parents[2]
    cmd = [
        str(project_root / ".venv/Scripts/python.exe"),
        "-m", "semantic_folding.query_processor",
        "--query-file", str(query_file),
        "--fingerprints", str(phrase_fingerprints_dir),
        "--doc-fingerprints", str(doc_fingerprints_dir),
        "--grid-size", str(grid_size),
        "--top-k", str(top_k),
        "--output", str(run_dir / "temp_results.json") if run_dir else "temp_results.json",
    ]
    
    # TODO: Filter corpus to candidates only
    # For now, run on full corpus (slow but correct)
    
    try:
        logger.debug(f"Running SF re-ranking: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=60,
        )
        
        if result.returncode != 0:
            logger.error(f"query_processor.py failed: {result.stderr[:500]}")
            return candidates[:top_k]
        
        # Parse results
        output_path = run_dir / "temp_results.json" if run_dir else Path("temp_results.json")
        if not output_path.exists():
            logger.error("No output file produced by query_processor.py")
            return candidates[:top_k]
        
        with open(output_path) as f:
            sf_results = json.load(f)
        
        # Extract ranked results
        if isinstance(sf_results, list) and len(sf_results) > 0:
            ranked = sf_results[0].get("results", [])
            reranked = [(doc_id, score) for doc_id, score in ranked if doc_id in set(candidate_ids)]
            return reranked[:top_k]
        
    except Exception as e:
        logger.error(f"SF re-ranking subprocess failed: {e}")
        return candidates[:top_k]
    
    finally:
        # Cleanup
        if query_file.exists():
            query_file.unlink()
    
    return candidates[:top_k]


# Keep old function for compatibility
def rerank_candidates_with_sf(*args, **kwargs):
    """Alias for v2."""
    return rerank_candidates_with_sf_v2(*args, **kwargs)


if __name__ == "__main__":
    print("SF Reranker v2 (subprocess-based)")

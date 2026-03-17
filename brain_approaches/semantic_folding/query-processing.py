import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.ndimage import gaussian_filter, convolve
from lib import load_fingerprint_cache, load_context_fingerprint_cache, load_phrases
from phrase_extractor import extract_phrases_general
from loguru import logger


# ── Spreading strategies ──────────────────────────────────────────────────────

def spread_fingerprint(
    fingerprint: np.ndarray,
    strategy: str = "gaussian",
    sigma: float = 1.0,
    radius: int = 1
) -> np.ndarray:
    """
    Apply neighbourhood spreading to a 2D fingerprint before cosine comparison.

    Parameters
    ----------
    fingerprint : 2D array of shape (grid_size, grid_size)
    strategy    : one of "gaussian" | "uniform" | "distance"
    sigma       : used by gaussian strategy
    radius      : used by uniform and distance strategies
    """
    fp = fingerprint.astype(np.float64)

    if strategy == "gaussian":
        return gaussian_filter(fp, sigma=sigma)

    elif strategy == "uniform":
        kernel_size = 2 * radius + 1
        from scipy.ndimage import uniform_filter
        return uniform_filter(fp, size=kernel_size)

    elif strategy == "distance":
        kernel = build_distance_kernel(radius)
        return convolve(fp, kernel, mode='constant', cval=0.0)

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Choose from gaussian | uniform | distance")


def build_distance_kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    center = radius
    kernel = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            kernel[i, j] = 1.0 / (1.0 + dist)
    kernel /= kernel.sum()
    return kernel


# ── Similarity ────────────────────────────────────────────────────────────────

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity with zero-magnitude guard."""
    dot = np.dot(vec1, vec2)
    mag1, mag2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot / (mag1 * mag2)


# ── Query fingerprint ─────────────────────────────────────────────────────────

def generate_query_fingerprint(
    query_text: str,
    fingerprint_cache: Dict[str, np.ndarray],
    phrases: List[str],
    grid_size: int
) -> np.ndarray:
    query_fingerprint = np.zeros((grid_size, grid_size), dtype=np.int32)
    query_phrases = extract_phrases_general(query_text)
    logger.info(f"Query Phrases: {query_phrases}")

    for phrase in phrases:
        if phrase in query_phrases:
            phrase_fp = fingerprint_cache.get(phrase, np.zeros((grid_size, grid_size)))
            query_fingerprint += phrase_fp

    return query_fingerprint


# ── Search ────────────────────────────────────────────────────────────────────

def neighbourhood_cosine_search(
    query_fingerprint: np.ndarray,
    context_fingerprint_cache: Dict[str, np.ndarray],
    top_k: int = 3,
    strategy: str = "gaussian",
    sigma: float = 1.0,
    radius: int = 1
) -> List[Tuple[str, float]]:
    """
    Cosine similarity search with neighbourhood spreading.

    Both query and context fingerprints are spread before comparison,
    so nearby activations contribute to the similarity score.

    The spreading makes the metric tolerant of small positional
    shifts in the 2D semantic space:

        sim = cos( spread(Q), spread(C) )
            = [spread(Q) · spread(C)] / [||spread(Q)|| ||spread(C)||]
    """
    # Spread and flatten query once
    spread_query = spread_fingerprint(query_fingerprint, strategy, sigma, radius)
    query_vector = spread_query.flatten()

    similarities = []
    for context_id, fingerprint in context_fingerprint_cache.items():
        spread_ctx = spread_fingerprint(fingerprint, strategy, sigma, radius)
        context_vector = spread_ctx.flatten()
        sim = cosine_similarity(query_vector, context_vector)
        similarities.append((context_id, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neighbourhood-aware query matching")
    parser.add_argument("--query_text",           required=True)
    parser.add_argument("--fingerprints_dir",      required=True)
    parser.add_argument("--doc_fingerprints_dir",  required=True)
    parser.add_argument("--phrases_path",          required=True)
    parser.add_argument("--grid_size",   type=int,   default=32)
    parser.add_argument("--top_k",       type=int,   default=3)
    parser.add_argument("--strategy",    default="gaussian",
                        choices=["gaussian", "uniform", "distance"],
                        help="Neighbourhood spreading strategy")
    parser.add_argument("--sigma",       type=float, default=1.0,
                        help="Gaussian sigma (used with --strategy gaussian)")
    parser.add_argument("--radius",      type=int,   default=1,
                        help="Neighbourhood radius (used with uniform/distance)")
    args = parser.parse_args()

    phrases = load_phrases(args.phrases_path)
    doc_fingerprint_cache    = load_context_fingerprint_cache(Path(args.doc_fingerprints_dir))
    phrase_fingerprint_cache = load_fingerprint_cache(Path(args.fingerprints_dir), phrases, args.grid_size)

    query_fingerprint = generate_query_fingerprint(
        args.query_text, phrase_fingerprint_cache, phrases, args.grid_size
    )

    logger.info(f"Non-zero query cells (before spread): {np.count_nonzero(query_fingerprint)}")

    results = neighbourhood_cosine_search(
        query_fingerprint,
        doc_fingerprint_cache,
        top_k=args.top_k,
        strategy=args.strategy,
        sigma=args.sigma,
        radius=args.radius
    )

    print(f"\nTop {args.top_k} results  [strategy={args.strategy}]:")
    for rank, (context_id, sim) in enumerate(results, start=1):
        print(f"  #{rank}  {context_id:<35}  similarity: {sim:.4f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


def paragraphize(text: str) -> List[str]:
    """Split text into paragraphs using blank lines as delimiters."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Further split long paragraphs by line breaks if needed
    result: List[str] = []
    for p in paras:
        # normalize whitespace
        p = "\n".join([ln.strip() for ln in p.splitlines() if ln.strip()])
        if p:
            result.append(p)
    return result


def _ensure_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is required for embeddings. Install with `pip install sentence-transformers`."
        ) from e


def compute_embeddings(
    texts: Iterable[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> List[List[float]]:
    """Compute embeddings for a list of texts using sentence-transformers.

    Returns a list of embedding vectors (as lists of floats).
    """
    SentenceTransformer = _ensure_sentence_transformer()
    model = SentenceTransformer(model_name)
    texts = list(texts)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    # Convert to native lists
    return [list(map(float, emb)) for emb in embeddings]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    # lightweight cosine similarity
    import math

    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def choose_representative_samples(
    paragraphs: List[str],
    embeddings: List[List[float]],
    top_k: int = 3,
) -> List[Tuple[int, str, float]]:
    """Pick top_k paragraphs closest to the centroid as representative samples.

    Returns list of tuples: (index, paragraph_text, score)
    """
    # compute centroid
    if not embeddings:
        return []
    dim = len(embeddings[0])
    centroid = [0.0] * dim
    for emb in embeddings:
        for i, v in enumerate(emb):
            centroid[i] += v
    n = len(embeddings)
    centroid = [v / n for v in centroid]

    scores = [cosine_similarity(e, centroid) for e in embeddings]
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(i, paragraphs[i], float(scores[i])) for i in idxs]

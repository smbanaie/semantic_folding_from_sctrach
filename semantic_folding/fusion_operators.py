"""Fusion operators for hybrid retrieval.

Combines the per-query scores/rankings of two retrievers (A, B) over a
candidate set. Definitions follow the canonical IR fusion literature:

- RRF      : Cormack, Clarke & Buettcher (2009), SIGIR. Reciprocal Rank Fusion.
- CombSUM  : Fox & Shaw (1994), TREC-2. Sum of raw scores.
- CombMNZ  : Fox & Shaw (1994). Sum of raw scores x number of retrievers hit.
- Borda    : rank-aggregation tally (N - rank + 1 summed across retrievers).
- Linear   : alpha-weighted combination of per-retriever normalized scores.
- zscore   : z-score normalization, then alpha-weighted linear combination.
- minmax   : min-max normalization, then alpha-weighted linear combination.

The ``fuse`` entry point is intentionally retriever-agnostic: callers pass two
plain ``{doc_id: score}`` dicts. Ranking inside each retriever is derived from
the scores (higher = better), so the same code serves SF+SPLADE, BM25+DPR, etc.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence


# ── helpers ────────────────────────────────────────────────────────────────

def rank_from_scores(scores: Dict[str, float]) -> Dict[str, int]:
    """Return {doc_id: rank} with rank 1 = highest score. Ties share the
    position of their first occurrence (stable by doc_id)."""
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return {doc_id: i + 1 for i, (doc_id, _) in enumerate(ordered)}


def _normalize(scores: Dict[str, float], kind: str) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    if kind == "none":
        return dict(scores)
    if kind == "max":
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {d: 1.0 for d in scores}
        return {d: (v - lo) / (hi - lo) for d, v in scores.items()}
    if kind == "minmax":
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {d: 0.0 for d in scores}
        return {d: (v - lo) / (hi - lo) for d, v in scores.items()}
    if kind == "zscore":
        n = len(vals)
        mu = sum(vals) / n
        var = sum((v - mu) ** 2 for v in vals) / n
        sd = math.sqrt(var) if var > 0 else 0.0
        if sd == 0.0:
            return {d: 0.0 for d in scores}
        return {d: (v - mu) / sd for d, v in scores.items()}
    if kind == "l2":
        norm = math.sqrt(sum(v * v for v in vals))
        if norm == 0.0:
            return {d: 0.0 for d in scores}
        return {d: v / norm for d, v in scores.items()}
    raise ValueError(f"unknown normalization: {kind}")


# ── individual operators ─────────────────────────────────────────────────────

def rrf(scores_a: Dict[str, float], scores_b: Dict[str, float],
        k: int = 60) -> Dict[str, float]:
    rank_a = rank_from_scores(scores_a)
    rank_b = rank_from_scores(scores_b)
    out: Dict[str, float] = {}
    for d in set(rank_a) | set(rank_b):
        ra = rank_a.get(d, len(rank_a) + 1)
        rb = rank_b.get(d, len(rank_b) + 1)
        out[d] = 1.0 / (k + ra) + 1.0 / (k + rb)
    return out


def borda(scores_a: Dict[str, float], scores_b: Dict[str, float],
          n_docs: int | None = None) -> Dict[str, float]:
    rank_a = rank_from_scores(scores_a)
    rank_b = rank_from_scores(scores_b)
    n = n_docs or max(len(rank_a), len(rank_b), 1)
    out: Dict[str, float] = {}
    for d in set(rank_a) | set(rank_b):
        ra = rank_a.get(d, n + 1)
        rb = rank_b.get(d, n + 1)
        out[d] = (n - ra + 1) + (n - rb + 1)
    return out


def combsum(scores_a: Dict[str, float], scores_b: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for d in set(scores_a) | set(scores_b):
        out[d] = scores_a.get(d, 0.0) + scores_b.get(d, 0.0)
    return out


def combmnz(scores_a: Dict[str, float], scores_b: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for d in set(scores_a) | set(scores_b):
        sa = scores_a.get(d, 0.0)
        sb = scores_b.get(d, 0.0)
        m = (1 if d in scores_a else 0) + (1 if d in scores_b else 0)
        out[d] = (sa + sb) * m
    return out


def _linear(scores_a: Dict[str, float], scores_b: Dict[str, float],
            alpha: float, norm_kind: str) -> Dict[str, float]:
    na = _normalize(scores_a, norm_kind)
    nb = _normalize(scores_b, norm_kind)
    out: Dict[str, float] = {}
    for d in set(na) | set(nb):
        out[d] = alpha * na.get(d, 0.0) + (1.0 - alpha) * nb.get(d, 0.0)
    return out


def linear(scores_a: Dict[str, float], scores_b: Dict[str, float],
           alpha: float = 0.3) -> Dict[str, float]:
    return _linear(scores_a, scores_b, alpha, "max")


def minmax_linear(scores_a: Dict[str, float], scores_b: Dict[str, float],
                  alpha: float = 0.3) -> Dict[str, float]:
    return _linear(scores_a, scores_b, alpha, "minmax")


def zscore_linear(scores_a: Dict[str, float], scores_b: Dict[str, float],
                  alpha: float = 0.3) -> Dict[str, float]:
    return _linear(scores_a, scores_b, alpha, "zscore")


def l2_linear(scores_a: Dict[str, float], scores_b: Dict[str, float],
              alpha: float = 0.3) -> Dict[str, float]:
    return _linear(scores_a, scores_b, alpha, "l2")


# ── dispatch ─────────────────────────────────────────────────────────────────

OPERATORS = {
    "rrf": rrf,
    "borda": borda,
    "combsum": combsum,
    "combmnz": combmnz,
    "linear": linear,
    "minmax": minmax_linear,
    "zscore": zscore_linear,
    "l2": l2_linear,
}

# Operators that discard absolute scores and fuse on rank only.
RANK_SPACE = {"rrf", "borda"}
# Operators that fuse on raw / normalized scores (magnitude-bearing).
SCORE_SPACE = {"combsum", "combmnz", "linear", "minmax", "zscore", "l2"}


def fuse(operator: str,
         scores_a: Dict[str, float],
         scores_b: Dict[str, float],
         **params) -> Dict[str, float]:
    """Fuse two retrievers' per-doc scores with the named operator.

    operator : one of OPERATORS keys
    scores_a / scores_b : {doc_id: float} native scores (higher = more relevant)
    params   : operator-specific (k for rrf, alpha for linear-family)
    """
    if operator not in OPERATORS:
        raise ValueError(
            f"unknown operator '{operator}'. choices: {sorted(OPERATORS)}"
        )
    if operator == "rrf":
        return rrf(scores_a, scores_b, k=params.get("k", 60))
    if operator == "borda":
        return borda(scores_a, scores_b, n_docs=params.get("n_docs"))
    if operator == "combsum":
        return combsum(scores_a, scores_b)
    if operator == "combmnz":
        return combmnz(scores_a, scores_b)
    # linear family
    alpha = params.get("alpha", 0.3)
    if operator == "linear":
        return linear(scores_a, scores_b, alpha)
    if operator == "minmax":
        return minmax_linear(scores_a, scores_b, alpha)
    if operator == "zscore":
        return zscore_linear(scores_a, scores_b, alpha)
    if operator == "l2":
        return l2_linear(scores_a, scores_b, alpha)
    raise ValueError(f"operator {operator} not wired")


def available_operators() -> List[str]:
    return sorted(OPERATORS)

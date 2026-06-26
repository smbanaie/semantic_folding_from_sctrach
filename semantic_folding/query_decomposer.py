#!/usr/bin/env python3
"""
query_decomposer.py — Multi-hop query decomposition for Semantic Folding

Decomposes complex queries into atomic sub-queries, scores each independently,
and combines results via reciprocal rank fusion (RRF).

This addresses SF's inability to compose facts across passages.
"""

from __future__ import annotations

import re
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


# Patterns that indicate multi-hop queries
MULTI_HOP_PATTERNS = [
    # "Who was the X of Y?" patterns
    r"(who|what|which)\s+(was|is|are)\s+the\s+.{2,30}\s+of\s+",
    # "What is the X of Y?" patterns
    r"what\s+(is|are)\s+the\s+.{2,30}\s+of\s+",
    # "Where is the X located?" patterns
    r"where\s+(is|are)\s+.{2,30}\s+(located|found|situated)",
    # "When did X happen to Y?" patterns
    r"when\s+(did|do|does)\s+.{2,30}\s+(happen|occur|start|end)",
    # "Why does X cause Y?" patterns
    r"why\s+(does|do|did)\s+.{2,30}\s+(cause|lead|result)",
    # "How does X affect Y?" patterns
    r"how\s+(does|do|did)\s+.{2,30}\s+(affect|impact|influence)",
    # Compound queries with "and"
    r"\bwhich\b.{10,}\band\b.{5,}\bwhich\b",
    # "What is the effect of X on Y in Z?" patterns
    r"what\s+(is|are)\s+the\s+effect.{10,}\bof\b.{5,}\bon\b",
]

# Relationship indicators for decomposition
RELATIONSHIP_INDICATORS = {
    "of": ["belongs to", "part of", "member of"],
    "in": ["located in", "found in", "part of"],
    "for": "used for",
    "by": "created by",
    "on": "affects",
    "with": "associated with",
}


def is_multi_hop_query(query: str) -> bool:
    """Detect if a query requires multi-hop reasoning."""
    query_lower = query.lower()
    for pattern in MULTI_HOP_PATTERNS:
        if re.search(pattern, query_lower):
            return True
    return False


def decompose_query(query: str) -> List[str]:
    """
    Decompose a complex query into atomic sub-queries.
    
    Strategy:
    1. Try to split on relationship indicators ("of", "in", "for", "on")
    2. If no split found, use the original query as a single sub-query
    
    Returns:
        List of sub-queries, ordered by dependency
    """
    query_lower = query.lower()
    
    # Try to split on "of" relationship
    # Pattern: "X of Y" -> sub-query 1: "What is Y?", sub-query 2: "What is the X of [Y]?"
    of_match = re.search(r"(.+?)\s+of\s+(.+?)(?:\?|$)", query_lower)
    if of_match:
        main_part = of_match.group(1).strip()
        of_part = of_match.group(2).strip().rstrip("?")
        
        # Extract the relationship (e.g., "spouse" from "spouse of")
        relationship_words = ["spouse", "parent", "child", "founder", "creator", "inventor", 
                             "discoverer", "author", "director", "president", "capital",
                             "currency", "language", "population", "area", "location"]
        
        relationship = None
        for word in relationship_words:
            if word in main_part:
                relationship = word
                break
        
        if relationship:
            sub_query_1 = f"What is {of_part}?"
            sub_query_2 = f"What is the {relationship} of [{{}}]?"
            logger.info(f"  [DECOMPOSE] Split on '{relationship}': {sub_query_1} -> {sub_query_2}")
            return [sub_query_1, sub_query_2]
    
    # Try to split on "in" relationship
    in_match = re.search(r"(.+?)\s+in\s+(.+?)(?:\?|$)", query_lower)
    if in_match:
        main_part = in_match.group(1).strip()
        in_part = in_match.group(2).strip().rstrip("?")
        
        if len(main_part) > 10 and len(in_part) > 5:
            sub_query_1 = f"What is {in_part}?"
            sub_query_2 = f"{main_part} in [{{}}]?"
            logger.info(f"  [DECOMPOSE] Split on 'in': {sub_query_1} -> {sub_query_2}")
            return [sub_query_1, sub_query_2]
    
    # No decomposition possible - return original query
    logger.debug(f"  [DECOMPOSE] No decomposition possible for: {query[:60]}...")
    return [query]


def fill_sub_query(sub_query: str, previous_result: str) -> str:
    """Fill in the placeholder in a sub-query with the previous result."""
    if "[{}]" in sub_query:
        return sub_query.replace("[{}]", previous_result)
    return sub_query


def combine_results_rsf(
    all_results: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Combine results from multiple sub-queries using Reciprocal Score Fusion (RSF).
    
    RSF formula: score(d) = sum(1 / (k + rank_i(d))) for each sub-query i
    
    Args:
        all_results: List of result lists from each sub-query
        k: RSF constant (default 60)
    
    Returns:
        Fused and sorted list of (doc_id, score) tuples
    """
    doc_scores = {}
    
    for results in all_results:
        for rank, (doc_id, _) in enumerate(results):
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
            doc_scores[doc_id] += 1.0 / (k + rank + 1)
    
    # Sort by fused score
    fused = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def decompose_and_score(
    query: str,
    score_fn,
    top_k: int = 5,
    max_hops: int = 2,
) -> List[Tuple[str, float]]:
    """
    Decompose a query and score each sub-query, then combine results.
    
    Args:
        query: Original query text
        score_fn: Function that takes a query string and returns List[(doc_id, score)]
        top_k: Number of results to return
        max_hops: Maximum number of hops to decompose
    
    Returns:
        Fused results list
    """
    sub_queries = decompose_query(query)
    
    if len(sub_queries) == 1:
        # No decomposition - score directly
        logger.info(f"  [DECOMPOSE] Single query, scoring directly")
        return score_fn(query)[:top_k]
    
    logger.info(f"  [DECOMPOSE] {len(sub_queries)} sub-queries:")
    for i, sq in enumerate(sub_queries):
        logger.info(f"    {i+1}. {sq}")
    
    all_results = []
    
    for i, sub_query in enumerate(sub_queries):
        if i == 0:
            # First sub-query - score directly
            results = score_fn(sub_query)
            all_results.append(results[:top_k])
            logger.info(f"  [DECOMPOSE] Sub-query {i+1}: {len(results)} results")
        else:
            # Try to use previous result to fill placeholder
            # For now, just score the sub-query as-is
            results = score_fn(sub_query)
            all_results.append(results[:top_k])
            logger.info(f"  [DECOMPOSE] Sub-query {i+1}: {len(results)} results")
    
    # Combine results using RSF
    fused = combine_results_rsf(all_results)
    
    logger.info(f"  [DECOMPOSE] Fused {len(all_results)} result sets into {len(fused)} unique documents")
    
    return fused[:top_k]

#!/usr/bin/env python3
"""
query_decomposer.py — Multi-hop query decomposition for Semantic Folding

Decomposes complex queries into atomic sub-queries using spaCy NER and
dependency parsing, scores each independently, and combines results via
Reciprocal Rank Fusion (RRF).

Improved version: uses NER + dependency parse instead of regex-only.
"""

from __future__ import annotations

import re
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# Try to load spaCy
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False
    _nlp = None


# ── Multi-hop detection ──────────────────────────────────────────────────────

MULTI_HOP_PATTERNS = [
    r"(who|what|which)\s+(was|is|are)\s+the\s+.{2,30}\s+of\s+",
    r"what\s+(is|are)\s+the\s+.{2,30}\s+of\s+",
    r"where\s+(is|are)\s+.{2,30}\s+(located|found|situated)",
    r"when\s+(did|do|does)\s+.{2,30}\s+(happen|occur|start|end)",
    r"why\s+(does|do|did)\s+.{2,30}\s+(cause|lead|result)",
    r"how\s+(does|do|did)\s+.{2,30}\s+(affect|impact|influence)",
    r"\bwhich\b.{10,}\band\b.{5,}\bwhich\b",
    r"what\s+(is|are)\s+the\s+effect.{10,}\bof\b.{5,}\bon\b",
]


def is_multi_hop_query(query: str) -> bool:
    """Detect if a query requires multi-hop reasoning."""
    query_lower = query.lower()
    for pattern in MULTI_HOP_PATTERNS:
        if re.search(pattern, query_lower):
            return True
    return False


# ── Entity and relation extraction ───────────────────────────────────────────

def extract_entities(query: str) -> List[Dict]:
    """Extract named entities from query using spaCy NER.

    Returns list of {text, label, start, end} dicts.
    """
    if not HAS_SPACY:
        return []
    doc = _nlp(query)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })
    return entities


def extract_relationship(query: str) -> Optional[str]:
    """Extract the main relationship/relation word from a query.

    Uses dependency parse to find the subject and its relationship.
    E.g., "Who was the spouse of X?" -> "spouse"
          "What is the capital of Y?" -> "capital"
    """
    if not HAS_SPACY:
        return None
    doc = _nlp(query)

    # Strategy 1: Find noun subject of "was/is/are" + prepositional phrase
    for tok in doc:
        if tok.dep_ == "nsubj" and tok.head.pos_ in ("AUX", "VERB"):
            # Found subject like "spouse" in "Who was the spouse of X?"
            # Check if it has a "of" prep
            for child in tok.children:
                if child.dep_ == "prep" and child.text.lower() == "of":
                    return tok.lemma_.lower()
            # Also check parent verb's children
            for child in tok.head.children:
                if child.dep_ == "prep" and child.text.lower() == "of":
                    return tok.lemma_.lower()

    # Strategy 2: Find "X of Y" pattern via dependency parse
    for tok in doc:
        if tok.text.lower() == "of" and tok.dep_ == "prep":
            # The head of "of" is the relationship noun
            head = tok.head
            if head.pos_ in ("NOUN", "PROPN"):
                return head.lemma_.lower()

    return None


def decompose_query(query: str) -> List[str]:
    """
    Decompose a complex query into atomic sub-queries.

    Strategy:
    1. Extract entities using spaCy NER
    2. Extract relationship using dependency parse
    3. Generate sub-queries from (entity, relationship) pairs
    4. If no decomposition possible, return original query

    Returns:
        List of sub-queries, ordered by dependency
    """
    entities = extract_entities(query)
    relationship = extract_relationship(query)

    # If we have entities and a relationship, decompose
    if entities and relationship:
        sub_queries = []
        for ent in entities:
            # Sub-query 1: Just the entity (for entity-focused retrieval)
            sub_queries.append(ent["text"])
            # Sub-query 2: Relationship + entity
            sub_queries.append(f"{relationship} {ent['text']}")

        if sub_queries:
            logger.info(f"  [DECOMPOSE] NER+dep: entity={entities[0]['text']} "
                        f"({entities[0]['label']}), relation={relationship}")
            return sub_queries

    # Fallback: try regex-based decomposition on "X of Y"
    of_match = re.search(r"(.+?)\s+of\s+(.+?)(?:\?|$)", query.lower())
    if of_match:
        main_part = of_match.group(1).strip()
        of_part = of_match.group(2).strip().rstrip("?")

        relationship_words = [
            "spouse", "parent", "child", "founder", "creator", "inventor",
            "discoverer", "author", "director", "president", "capital",
            "currency", "language", "population", "area", "location",
        ]

        relationship = None
        for word in relationship_words:
            if word in main_part:
                relationship = word
                break

        if relationship:
            sub_q1 = f"What is {of_part}?"
            sub_q2 = f"What is the {relationship} of {{}}?"
            logger.info(f"  [DECOMPOSE] Regex: split on '{relationship}'")
            return [sub_q1, sub_q2]

    # No decomposition possible
    logger.debug(f"  [DECOMPOSE] No decomposition for: {query[:60]}...")
    return [query]


def fill_sub_query(sub_query: str, previous_result: str) -> str:
    """Fill in the placeholder in a sub-query with the previous result."""
    if "[{}]" in sub_query:
        return sub_query.replace("[{}]", previous_result)
    return sub_query


def combine_results_rrf(
    all_results: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Combine results from multiple sub-queries using Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each sub-query i

    Args:
        all_results: List of result lists from each sub-query
        k: RRF constant (default 60)

    Returns:
        Fused and sorted list of (doc_id, score) tuples
    """
    doc_scores = {}

    for results in all_results:
        for rank, (doc_id, _) in enumerate(results):
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
            doc_scores[doc_id] += 1.0 / (k + rank + 1)

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
        logger.info(f"  [DECOMPOSE] Single query, scoring directly")
        return score_fn(query)[:top_k]

    logger.info(f"  [DECOMPOSE] {len(sub_queries)} sub-queries:")
    for i, sq in enumerate(sub_queries):
        logger.info(f"    {i+1}. {sq}")

    all_results = []

    for i, sub_query in enumerate(sub_queries):
        results = score_fn(sub_query)
        all_results.append(results[:top_k])
        logger.info(f"  [DECOMPOSE] Sub-query {i+1}: {len(results)} results")

    # Combine results using RRF
    fused = combine_results_rrf(all_results)

    logger.info(f"  [DECOMPOSE] Fused {len(all_results)} result sets into "
                f"{len(fused)} unique documents")

    return fused[:top_k]

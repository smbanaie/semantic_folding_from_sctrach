"""
negation_handler.py - Negation-Aware Query Processing

Handles negation in queries by:
1. Detecting negation cues (not, no, never, neither, nor, without, etc.)
2. Extracting negated concepts (what's being negated)
3. Creating negation-aware scoring for document retrieval

Key insight: Negation words are stripped as stopwords during phrase extraction,
so "not in France" produces the same fingerprint as "in France". This module
provides post-retrieval negation-aware scoring to fix this.

Author: [Your Name]
Date: 2026-06-18
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from loguru import logger


# Negation cues ordered by specificity (most specific first)
NEGATION_CUES = [
    # Multi-word cues (check first to avoid partial matches)
    "does not", "do not", "did not", "has not", "have not", "had not",
    "is not", "are not", "was not", "were not", "will not", "would not",
    "could not", "should not", "cannot", "can not", "must not",
    "lack of", "absence of", "instead of", "other than",
    "rather than", "as opposed to",
    # Single-word cues
    "not", "no", "never", "neither", "nor", "without",
    # Explicit exclusion patterns
    "except", "excluding", "apart from",
]

# Pattern to extract negated concept (word(s) after negation cue)
# Example: "not in France" -> "in France"
# Example: "does not contain water" -> "contain water"
NEGATED_CONCEPT_PATTERN = re.compile(
    r'(?:' + '|'.join(re.escape(cue) for cue in NEGATION_CUES) + r')\s+',
    re.IGNORECASE
)

# Pattern for "X except Y" or "X excluding Y"
EXCLUSION_PATTERN = re.compile(
    r'(.+?)\s+(?:except|excluding|apart from)\s+(.+?)(?:\?|$)',
    re.IGNORECASE
)

# Pattern for "Which of the following is not..."
WHICH_NOT_PATTERN = re.compile(
    r'which\s+(?:of\s+the\s+following\s+)?is\s+not',
    re.IGNORECASE
)


class NegationHandler:
    """Handles negation-aware query processing and document scoring."""

    def __init__(self):
        self.negation_cues = NEGATION_CUES

    def detect_negation(self, query: str) -> Dict:
        """
        Detect and analyze negation in a query.

        Returns:
            Dict with keys:
                - has_negation: bool
                - negation_type: str ('negated_concept', 'exclusion', 'which_not', 'none')
                - negated_concepts: List[str] - words/concepts being negated
                - positive_concepts: List[str] - what IS being asked about
                - negation_cues_found: List[str] - which cues were detected
        """
        query_lower = query.lower()
        result = {
            "has_negation": False,
            "negation_type": "none",
            "negated_concepts": [],
            "positive_concepts": [],
            "negation_cues_found": [],
        }

        # Check for "Which of the following is not..." pattern
        if WHICH_NOT_PATTERN.search(query):
            result["has_negation"] = True
            result["negation_type"] = "which_not"
            # Extract what comes after "not"
            match = WHICH_NOT_PATTERN.search(query)
            if match:
                remaining = query_lower[match.end():].strip().rstrip('?').strip()
                if remaining:
                    result["negated_concepts"] = remaining.split()[:3]
            return result

        # Check for exclusion patterns (except, excluding)
        excl_match = EXCLUSION_PATTERN.search(query)
        if excl_match:
            result["has_negation"] = True
            result["negation_type"] = "exclusion"
            result["positive_concepts"] = excl_match.group(1).split()[:3]
            result["negated_concepts"] = excl_match.group(2).split()[:3]
            return result

        # Check for negation cues
        cues_found = []
        negated_concepts = []

        for cue in self.negation_cues:
            # Use word boundary matching for single-word cues
            if cue in ["not", "no", "never", "neither", "nor", "without"]:
                pattern = r'\b' + re.escape(cue) + r'\b'
                if re.search(pattern, query_lower):
                    cues_found.append(cue)
                    # Extract concept after negation cue
                    match = re.search(pattern + r'\s+(.+?)(?:\?|$)', query_lower)
                    if match:
                        concept_words = match.group(1).split()[:3]
                        negated_concepts.extend(concept_words)
            else:
                if cue in query_lower:
                    cues_found.append(cue)
                    # Extract concept after negation cue
                    idx = query_lower.find(cue) + len(cue)
                    remaining = query_lower[idx:].strip().rstrip('?').strip()
                    if remaining:
                        concept_words = remaining.split()[:3]
                        negated_concepts.extend(concept_words)

        if cues_found:
            result["has_negation"] = True
            result["negation_type"] = "negated_concept"
            result["negation_cues_found"] = list(set(cues_found))  # Deduplicate cues
            # Deduplicate and limit negated concepts
            seen_concepts = set()
            unique_concepts = []
            for c in negated_concepts:
                if c not in seen_concepts:
                    seen_concepts.add(c)
                    unique_concepts.append(c)
            result["negated_concepts"] = unique_concepts[:5]  # Limit to 5 concepts

            # Extract positive concepts (what's NOT negated)
            # Remove negation cues and negated concepts from query
            positive_parts = query_lower
            for cue in cues_found:
                positive_parts = positive_parts.replace(cue, '')
            if unique_concepts:
                for concept in unique_concepts:
                    positive_parts = positive_parts.replace(concept, '')
            # Clean up and get remaining words
            positive_words = re.findall(r'\b[a-z]+\b', positive_parts)
            result["positive_concepts"] = [w for w in positive_words if len(w) > 2][:5]

        return result

    def score_documents(
        self,
        query: str,
        results: List[Tuple[str, float]],
        doc_texts: Dict[str, str],
        negation_boost: float = 0.3,
        negation_penalty: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Apply negation-aware scoring to ranked results.

        Args:
            query: Original query string
            results: List of (doc_id, score) tuples
            doc_texts: Dict mapping doc_id to document text
            negation_boost: Boost for documents with properly negated concepts
            negation_penalty: Penalty for documents with negated concepts but no negation context

        Returns:
            Re-ranked results with negation-aware scoring
        """
        neg_info = self.detect_negation(query)

        if not neg_info["has_negation"] or not neg_info["negated_concepts"]:
            logger.debug("  [NEGATION] no negation detected or no negated concepts")
            return results

        logger.info(
            f"  [NEGATION] type={neg_info['negation_type']}, "
            f"cues={neg_info['negation_cues_found']}, "
            f"negated={neg_info['negated_concepts']}"
        )

        adjusted_results = []
        for doc_id, score in results:
            doc_text = doc_texts.get(doc_id, "").lower()

            if not doc_text:
                adjusted_results.append((doc_id, score))
                continue

            # Check if document contains negated concepts
            has_negated_concept = False
            has_proper_negation = False

            for concept in neg_info["negated_concepts"]:
                if concept in doc_text:
                    has_negated_concept = True

                    # Check if the concept is properly negated in the document
                    # Look for negation cues near the concept (within 50 chars)
                    for cue in neg_info["negation_cues_found"]:
                        # Find all occurrences of the cue
                        for match in re.finditer(re.escape(cue), doc_text):
                            cue_pos = match.start()
                            concept_pos = doc_text.find(concept)

                            # If negation cue is within 50 chars before concept
                            if 0 < concept_pos - cue_pos <= 50:
                                has_proper_negation = True
                                break

                    if has_proper_negation:
                        break

            # Apply scoring adjustments
            if has_negated_concept and has_proper_negation:
                # Document properly negates the concept - boost it
                adjusted_score = score * (1 + negation_boost)
                logger.debug(
                    f"  [NEGATION BOOST] doc={doc_id}, "
                    f"score={score:.4f} -> {adjusted_score:.4f}"
                )
            elif has_negated_concept and not has_proper_negation:
                # Document mentions negated concept without proper context - penalty
                adjusted_score = score * (1 - negation_penalty)
                logger.debug(
                    f"  [NEGATION PENALTY] doc={doc_id}, "
                    f"score={score:.4f} -> {adjusted_score:.4f}"
                )
            else:
                adjusted_score = score

            adjusted_results.append((doc_id, adjusted_score))

        # Re-sort by adjusted scores
        adjusted_results.sort(key=lambda x: x[1], reverse=True)

        return adjusted_results

    def create_negation_aware_query(
        self,
        query: str,
        phrase_vocab: Set[str],
    ) -> Tuple[str, List[str]]:
        """
        Create a negation-aware query representation for fingerprint construction.

        Instead of stripping negation words, this preserves them to create
        a query fingerprint that represents both what IS asked AND what is negated.

        Args:
            query: Original query string
            phrase_vocab: Set of vocabulary phrases

        Returns:
            Tuple of (modified_query, negation_markers) where:
                - modified_query: Query with negation markers preserved
                - negation_markers: List of negation markers to track
        """
        neg_info = self.detect_negation(query)

        if not neg_info["has_negation"]:
            return query, []

        # Add negation markers to preserve negation information
        # Example: "Which city is not in France?" -> "Which city is not_neg in France?"
        markers = []
        modified_query = query

        for cue in neg_info["negation_cues_found"]:
            # Add a marker after the negation cue
            marker = f"{cue}_neg"
            markers.append(marker)
            # Replace cue with cue+marker in query
            modified_query = modified_query.replace(cue, marker, 1)

        return modified_query, markers


# Global instance for reuse
_negation_handler = None


def get_negation_handler() -> NegationHandler:
    """Get or create the global negation handler instance."""
    global _negation_handler
    if _negation_handler is None:
        _negation_handler = NegationHandler()
    return _negation_handler

"""
ontology_expander.py - Ontology-Guided Query Expansion

Enhances query expansion using MeSH/UMLS-style ontologies by:
1. Weighted synonym expansion (synonyms get lower weight)
2. Multi-word term expansion (expand phrases as units)
3. MeSH tree-based expansion (find related terms via hierarchy)
4. Context-aware expansion (only expand semantically relevant terms)

Author: [Your Name]
Date: 2026-06-18
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from loguru import logger


class OntologyExpander:
    """Enhanced query expansion using MeSH/UMLS ontologies."""

    def __init__(self, glossary_path: Optional[str] = None):
        """
        Initialize the ontology expander.

        Args:
            glossary_path: Path to glossary JSON file (MeSH/UMLS format)
        """
        self.glossary: Dict[str, List[str]] = {}
        self.term_weights: Dict[str, float] = {}
        self.mesh_tree: Dict[str, List[str]] = {}  # parent -> children mapping

        if glossary_path:
            self.load_glossary(glossary_path)

    def load_glossary(self, glossary_path: str) -> None:
        """Load glossary from JSON file."""
        path = Path(glossary_path)
        if not path.exists():
            logger.warning(f"  [ONTOLOGY] glossary not found: {glossary_path}")
            return

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Flatten domains into term→synonyms mapping
        for domain_name, domain in data.get("domains", {}).items():
            for category, terms in domain.items():
                for canonical, synonyms in terms.items():
                    self.glossary[canonical.lower()] = [s.lower() for s in synonyms]
                    # Assign weights: canonical=1.0, synonyms=0.5
                    self.term_weights[canonical.lower()] = 1.0
                    for syn in synonyms:
                        self.term_weights[syn.lower()] = 0.5

        logger.info(f"  [ONTOLOGY] loaded {len(self.glossary)} terms from {glossary_path}")

    def detect_terms(self, query: str) -> List[Tuple[str, List[str], float]]:
        """
        Detect ontology terms in a query.

        Returns:
            List of (canonical_term, synonyms, weight) tuples
        """
        query_lower = query.lower()
        detected = []

        # Sort by length (longest first) to match multi-word terms first
        sorted_terms = sorted(self.glossary.keys(), key=len, reverse=True)

        for term in sorted_terms:
            if term in query_lower:
                synonyms = self.glossary[term]
                weight = self.term_weights.get(term, 1.0)
                detected.append((term, synonyms, weight))
                # Remove matched term from query to avoid partial matches
                query_lower = query_lower.replace(term, "", 1)

        return detected

    def expand_query_weighted(
        self,
        query: str,
        synonym_weight: float = 0.5,
        max_synonyms: int = 3,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Expand query with weighted synonyms.

        Args:
            query: Original query
            synonym_weight: Weight for synonym terms (0-1)
            max_synonyms: Maximum number of synonyms to add per term

        Returns:
            Tuple of (expanded_query, term_weights) where term_weights
            maps each term to its weight in the expanded query
        """
        detected = self.detect_terms(query)

        if not detected:
            return query, {}

        term_weights = {}
        expanded_parts = [query]

        for canonical, synonyms, weight in detected:
            # Add canonical term with its weight
            term_weights[canonical] = weight

            # Add top synonyms with reduced weight
            for syn in synonyms[:max_synonyms]:
                term_weights[syn] = synonym_weight
                expanded_parts.append(syn)

        expanded_query = " ".join(expanded_parts)
        logger.info(
            f"  [ONTOLOGY] expanded query: +{len(detected)} terms, "
            f"+{sum(len(s) for _, s, _ in detected)} synonyms"
        )

        return expanded_query, term_weights

    def expand_with_mesh_tree(
        self,
        query: str,
        mesh_tree: Optional[Dict[str, List[str]]] = None,
        depth: int = 1,
    ) -> str:
        """
        Expand query using MeSH tree hierarchy.

        Args:
            query: Original query
            mesh_tree: Optional custom tree (parent -> children)
            depth: How many levels to traverse

        Returns:
            Expanded query with MeSH-related terms
        """
        if mesh_tree:
            self.mesh_tree = mesh_tree

        if not self.mesh_tree:
            return query

        detected = self.detect_terms(query)
        expanded_terms = [query]

        for canonical, _, _ in detected:
            # Find related terms via tree traversal
            related = self._traverse_tree(canonical, depth)
            for term in related[:2]:  # Limit to 2 related terms
                if term not in query.lower():
                    expanded_terms.append(term)

        return " ".join(expanded_terms)

    def _traverse_tree(self, term: str, depth: int) -> List[str]:
        """Traverse MeSH tree to find related terms."""
        if depth <= 0 or term not in self.mesh_tree:
            return []

        related = []
        children = self.mesh_tree[term]
        related.extend(children)

        # Recurse if depth allows
        if depth > 1:
            for child in children:
                related.extend(self._traverse_tree(child, depth - 1))

        return related

    def create_phrase_weights(
        self,
        query_phrases: List[str],
        synonym_weight: float = 0.5,
    ) -> Dict[str, float]:
        """
        Create weight mapping for query phrases based on ontology.

        Args:
            query_phrases: List of extracted query phrases
            synonym_weight: Weight for synonym-derived phrases

        Returns:
            Dict mapping phrase to weight
        """
        weights = {}

        for phrase in query_phrases:
            phrase_lower = phrase.lower()

            # Check if phrase is a canonical term
            if phrase_lower in self.term_weights:
                weights[phrase] = self.term_weights[phrase_lower]
            # Check if phrase is a synonym
            elif any(phrase_lower in syns for syns in self.glossary.values()):
                weights[phrase] = synonym_weight
            else:
                weights[phrase] = 1.0  # Default weight for non-ontology terms

        return weights


# Global instance
_ontology_expander = None


def get_ontology_expander(glossary_path: Optional[str] = None) -> OntologyExpander:
    """Get or create the global ontology expander instance."""
    global _ontology_expander
    if _ontology_expander is None or glossary_path:
        _ontology_expander = OntologyExpander(glossary_path)
    return _ontology_expander

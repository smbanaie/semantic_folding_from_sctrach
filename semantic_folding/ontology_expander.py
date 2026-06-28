"""
ontology_expander.py - Ontology-Guided Query Expansion

Enhances query expansion using MeSH/UMLS-style ontologies by:
1. Weighted synonym expansion (synonyms get lower weight)
2. Multi-word term expansion (expand phrases as units)
3. MeSH tree-based expansion (find related terms via hierarchy)
4. Context-aware expansion (only expand semantically relevant terms)
5. Corpus-level glossary injection (append synonyms to corpus for indexing)
6. Query-time phrase normalization (map synonyms to canonical before vocab lookup)

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
        self._syn_to_canon: Optional[Dict[str, str]] = None

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

    # ------------------------------------------------------------------
    # P1.3: Corpus-level glossary injection + query normalization
    # ------------------------------------------------------------------

    def build_synonym_to_canonical(self) -> Dict[str, str]:
        """Build reverse mapping: synonym -> canonical term.

        This is used to normalize query phrases to their canonical form
        before vocabulary lookup.
        """
        mapping = {}
        for canonical, synonyms in self.glossary.items():
            mapping[canonical] = canonical
            for syn in synonyms:
                # Only map if synonym doesn't conflict with another canonical
                if syn not in self.glossary:
                    mapping[syn] = canonical
        return mapping

    def normalize_phrase_through_glossary(self, phrase: str) -> str:
        """Normalize a phrase by mapping synonyms to canonical form.

        E.g., 'heart attack' -> 'myocardial infarction'
              'mirna' -> 'microrna'
        """
        phrase_lower = phrase.lower()
        if self._syn_to_canon is None:
            self._syn_to_canon = self.build_synonym_to_canonical()
        if phrase_lower in self._syn_to_canon:
            return self._syn_to_canon[phrase_lower]
        return phrase

    def normalize_query_phrases(self, phrases: List[str]) -> List[str]:
        """Normalize a list of query phrases through the glossary.

        Maps each synonym to its canonical form. If the canonical form
        is in the phrase vocabulary, it will match; the synonym may not be.
        """
        if not self.glossary:
            return phrases
        return [self.normalize_phrase_through_glossary(p) for p in phrases]

    def expand_corpus_with_glossary(self, corpus_lines: List[str]) -> List[str]:
        """Append glossary synonyms to each corpus line where canonical terms appear.

        For each line, if a canonical term or its synonyms appear in the text,
        append all other forms (synonyms or canonical) to the end of the line.
        This ensures all synonym variants get picked up by phrase extraction
        and receive fingerprints in the semantic grid.

        Args:
            corpus_lines: List of corpus lines (doc_id, title text)

        Returns:
            Expanded corpus lines with appended synonyms
        """
        if not self.glossary:
            return corpus_lines

        # Build a combined mapping: every variant -> set of all other variants
        variant_groups = []
        for canonical, synonyms in self.glossary.items():
            all_variants = {canonical} | set(synonyms)
            variant_groups.append(all_variants)

        expanded_lines = []
        total_injections = 0

        for line in corpus_lines:
            line_lower = line.lower()
            append_terms = []

            for variants in variant_groups:
                # Check if any variant appears in the line
                found_variant = None
                for v in variants:
                    if v in line_lower:
                        found_variant = v
                        break

                if found_variant:
                    # Append all OTHER variants that aren't in the text
                    for v in variants:
                        if v != found_variant and v not in line_lower and v not in append_terms:
                            append_terms.append(v)

            if append_terms:
                expanded_line = line + " " + " ".join(append_terms)
                expanded_lines.append(expanded_line)
                total_injections += len(append_terms)
            else:
                expanded_lines.append(line)

        logger.info(
            f"  [ONTOLOGY] corpus expansion: {total_injections} synonym injections "
            f"across {len(corpus_lines)} lines"
        )
        return expanded_lines


# Global instance
_ontology_expander = None


def get_ontology_expander(glossary_path: Optional[str] = None) -> OntologyExpander:
    """Get or create the global ontology expander instance."""
    global _ontology_expander
    if _ontology_expander is None or glossary_path:
        _ontology_expander = OntologyExpander(glossary_path)
    return _ontology_expander


def expand_corpus_with_glossary(corpus_lines: List[str], glossary_path: str) -> List[str]:
    """Module-level convenience function for corpus expansion."""
    expander = get_ontology_expander(glossary_path)
    return expander.expand_corpus_with_glossary(corpus_lines)
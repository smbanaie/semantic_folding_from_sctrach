"""
Reviewer Agent - Phase 5 of the knowledge graph pipeline.

Validates, normalizes, and deduplicates triples extracted by the Extractor agent.
"""

import json
import logging
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Set

from tqdm import tqdm

from src.config import get_config
from src.models.data_models import GraphState, Triple
from src.utils.openrouter_client import OpenRouterClient
from src.utils.io_utils import save_agent_output

logger = logging.getLogger(__name__)


def _similarity(a: str, b: str) -> float:
    """
    Calculate similarity between two strings.

    Args:
        a: First string
        b: Second string

    Returns:
        Similarity score (0.0-1.0)
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _normalize_entity(entity: str, entity_cache: Dict[str, str] | None = None) -> str:
    """
    Enhanced entity normalization with shortening for lengthy phrases.
    
    Note: This is a synchronous wrapper. For LLM-based shortening,
    use _normalize_entities_batch() which handles async operations.

    Args:
        entity: Entity string to normalize
        entity_cache: Optional cache dictionary for shortened entities

    Returns:
        Normalized entity string
    """
    # Check cache first
    if entity_cache and entity in entity_cache:
        return entity_cache[entity]
    
    # Remove extra whitespace
    entity = " ".join(entity.split())
    
    # Shorten common lengthy phrases
    entity = _shorten_common_phrases(entity)
    
    # Capitalize first letter of each word (for proper nouns)
    # But preserve acronyms (all caps) and known acronyms
    if entity.isupper() or len(entity) <= 3:
        result = entity
    else:
        # Preserve known acronyms in title case
        known_acronyms = ['RAG', 'LLM', 'AI', 'NER', 'API', 'GPU', 'CPU', 'ML', 'DL', 'NLP']
        words = entity.split()
        for i, word in enumerate(words):
            if word.upper() in known_acronyms:
                words[i] = word.upper()
        
        result = ' '.join(words)
    
    # Cache result
    if entity_cache is not None:
        entity_cache[entity] = result
    
    return result


def _shorten_common_phrases(entity: str) -> str:
    """
    Shorten common lengthy phrases in entities.
    
    Args:
        entity: Entity string to shorten
        
    Returns:
        Shortened entity string
    """
    # Common phrase mappings
    phrase_mappings = {
        # Technology terms
        "Large Language Models": "LLMs",
        "Knowledge Graphs": "Knowledge Graphs",  # Keep as is, already concise
        "Vector Embeddings": "Vector Embeddings",  # Keep as is
        "Retrieval-Augmented Generation": "RAG",
        "Graph-Augmented Retrieval": "GraphRAG",
        "Named Entity Recognition": "NER",
        "Relationship Extraction": "Relation Extraction",
        "Entity Linking": "Entity Linking",  # Keep as is
        "Multi-Hop Reasoning": "Multi-Hop Reasoning",  # Keep as is
        
        # Common descriptive phrases
        "Rich Relational Structure": "Relational Structure",
        "Complex Relationships": "Complex Relationships",  # Keep as is
        "Explicit Relational Structure": "Relational Structure",
        "Semantic Similarity": "Semantic Similarity",  # Keep as is
        "Contextual Connections": "Contextual Connections",  # Keep as is
        "Hierarchical Structures": "Hierarchical Structures",  # Keep as is
        "Temporal Relationships": "Temporal Relationships",  # Keep as is
        "Causal Relationships": "Causal Relationships",  # Keep as is
        "Semantic Associations": "Semantic Associations",  # Keep as is
        
        # System descriptions
        "Traditional RAG Approaches": "Traditional RAG",
        "GraphRAG Systems": "GraphRAG Systems",  # Keep as is
        "Knowledge Graph Construction": "Knowledge Graph Construction",  # Keep as is
        "Graph-Enhanced Retrieval": "Graph-Enhanced Retrieval",  # Keep as is
        "Augmented Generation": "Augmented Generation",  # Keep as is
        
        # Quality descriptions
        "High-Quality Knowledge Graphs": "Knowledge Graphs",
        "Accurate Entity Recognition": "Entity Recognition",
        "Correct Relationship Extraction": "Relationship Extraction",
        "Proper Entity Disambiguation": "Entity Disambiguation",
        
        # Process descriptions
        "Entity Extraction": "Entity Extraction",  # Keep as is
        "Relationship Identification": "Relationship Identification",  # Keep as is
        "Query Expansion": "Query Expansion",  # Keep as is
        "Graph Traversal Algorithms": "Graph Traversal",
        "Vector Similarity Search": "Vector Search",
        "Entity Disambiguation": "Entity Disambiguation",  # Keep as is
        "Entity Linking Techniques": "Entity Linking",
        
        # Result descriptions
        "Relevant Subgraphs": "Subgraphs",
        "Vector Search Results": "Search Results",
        "Knowledge Graph Edges": "Graph Edges",
        "Canonical Entity Representations": "Entity Representations",
        
        # Capability descriptions
        "More Sophisticated Reasoning": "Sophisticated Reasoning",
        "Multi-Hop Reasoning": "Multi-Hop Reasoning",  # Keep as is
        "Handling Complex Queries": "Complex Queries",
        "Integrating Knowledge Graphs": "Knowledge Graph Integration",
        "Creating Opportunities": "Opportunities",
        "Explainable AI": "Explainable AI",  # Keep as is
        "Computational Cost": "Computational Cost",  # Keep as is
        "Substantial Computational Resources": "Computational Resources",
        "Underlying Extraction Models": "Extraction Models",
        "Errors That Propagate": "Propagation Errors",
        
        # Structure descriptions
        "Interconnected Network Of Information": "Information Network",
        "Underlying Knowledge Representation Structure": "Knowledge Structure",
        "Flat Vector Embeddings": "Vector Embeddings",
        "Rich Relational Structure": "Relational Structure",
        "Explicit Relational Structure": "Relational Structure",
        "Semantic Similarity": "Semantic Similarity",  # Keep as is
        "Contextual Connections Between Concepts": "Concept Connections",
        
        # Performance descriptions
        "System Performance": "Performance",
        "Quality Impacts": "Quality Impact",
        "Accurate Entity Recognition": "Entity Recognition",
        "Correct Relationship Extraction": "Relationship Extraction",
        "Proper Entity Disambiguation": "Entity Disambiguation",
        
        # Connection descriptions
        "Mentions Of The Same Entity": "Entity Mentions",
        "Canonical Entity Representations": "Entity Representations",
        "Text Mentions": "Text Mentions",
        
        # Query descriptions
        "Related Entities": "Related Entities",  # Keep as is
        "Query Includes": "Query Includes",  # Keep as is
        
        # Long descriptive phrases that can be shortened
        "Questions That Require Connecting Information From Multiple Sources": "Complex Questions",
        "Opportunities For Explainable AI": "Explainable AI Opportunities",
        "Explanations By Showing Graph Paths": "Graph Path Explanations",
        "Several Challenges In Implementation And Deployment": "Implementation Challenges",
        "Computational Cost Of Building And Maintaining Knowledge Graphs": "Knowledge Graph Costs",
        "Quality Of Extracted Knowledge": "Knowledge Quality",
        "Errors That Propagate Through The System": "System Errors",
        "Substantial Computational Resources": "Computational Resources",
        "Contextual Connections Between Concepts": "Concept Connections",
    }
    
    # Apply mappings
    entity_lower = entity.lower()
    for long_phrase, short_phrase in phrase_mappings.items():
        if entity_lower == long_phrase.lower():
            return short_phrase
    
    return entity


async def _shorten_entities_batch(
    entities: List[str],
    client: OpenRouterClient,
    max_words: int = 3,
) -> Dict[str, str]:
    """
    Shorten a batch of entities using LLM to ensure they are ≤ max_words.
    
    Args:
        entities: List of entity strings to shorten
        client: OpenRouter client for LLM calls
        max_words: Maximum number of words allowed (default: 3)
    
    Returns:
        Dictionary mapping original entities to shortened versions
    """
    # Filter entities that need shortening
    entities_to_shorten = [
        e for e in entities 
        if len(e.split()) > max_words
    ]
    
    if not entities_to_shorten:
        return {e: e for e in entities}
    
    logger.info(f"Shortening {len(entities_to_shorten)} entities with LLM (max {max_words} words)")
    
    # Create optimized prompt for entity shortening
    shortening_prompt = f"""You are an entity name optimizer for knowledge graphs. Your task is to shorten entity names while preserving their semantic meaning.

ENTITY NAMES TO SHORTEN:
{json.dumps(entities_to_shorten, indent=2)}

REQUIREMENTS:
1. Shorten each entity to AT MOST {max_words} words
2. Preserve the core semantic meaning and key concepts
3. Prioritize the most important words that convey the entity's meaning
4. Use acronyms for common phrases when appropriate (e.g., "Large Language Models" → "LLMs")
5. Maintain readability for graph visualization
6. Keep technical terms intact when they are essential (e.g., "Knowledge Graph", "RAG", "LLM")
7. Remove filler words like "of", "the", "that", "from", "in", "and" when possible without losing meaning

EXAMPLES:
- "Questions That Require Connecting Information From Multiple Sources" → "Multi-Source Queries" or "Complex Queries"
- "Computational Cost Of Building And Maintaining Knowledge Graphs" → "Knowledge Graph Costs" or "Graph Construction Costs"
- "Underlying Knowledge Representation Structure" → "Knowledge Structure" or "Knowledge Representation"
- "Several Challenges In Implementation And Deployment" → "Implementation Challenges" or "Deployment Challenges"
- "Large Language Models" → "LLMs"
- "Retrieval-Augmented Generation" → "RAG"

Return a JSON object mapping each original entity to its shortened version:
{{
  "shortened_entities": {{
    "original entity 1": "shortened version 1",
    "original entity 2": "shortened version 2",
    ...
  }}
}}

IMPORTANT:
- Each shortened entity MUST be ≤ {max_words} words
- Preserve semantic meaning - the shortened version should convey the same concept
- Use consistent shortening patterns (e.g., always shorten "Large Language Models" to "LLMs")
- If an entity is already ≤ {max_words} words, return it unchanged

Now shorten the entities:"""

    try:
        system_prompt = "You are an entity name optimizer. Always respond with valid JSON only, no additional text."
        response = await client.generate(
            prompt=shortening_prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for more consistent shortening
            response_format={"type": "json_object"},
        )
        
        # Parse response
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
                result = json.loads(response)
            else:
                logger.warning("Could not parse shortening response, using original entities")
                return {e: e for e in entities}
        
        shortened_map = result.get("shortened_entities", {})
        
        # Build result dictionary with all entities (shortened + unchanged)
        result_map: Dict[str, str] = {}
        for entity in entities:
            if entity in shortened_map:
                shortened = shortened_map[entity]
                # Validate word count
                word_count = len(shortened.split())
                if word_count <= max_words:
                    result_map[entity] = shortened
                    logger.debug(f"Shortened: '{entity}' → '{shortened}' ({word_count} words)")
                else:
                    logger.warning(
                        f"Shortened entity '{shortened}' still has {word_count} words "
                        f"(max {max_words}), keeping original"
                    )
                    result_map[entity] = entity
            else:
                # Entity not in response, keep original
                result_map[entity] = entity
        
        return result_map
        
    except Exception as e:
        logger.error(f"Error shortening entities with LLM: {e}", exc_info=True)
        # Fallback: return original entities
        return {e: e for e in entities}


async def reviewer_agent(state: GraphState) -> GraphState:
    """
    Validate, normalize, and deduplicate triples.

    This agent:
    1. Takes raw_triples from state
    2. Normalizes entity names
    3. Detects and merges duplicates
    4. Validates triple consistency
    5. Corrects errors using LLM

    Args:
        state: Current graph state with raw_triples

    Returns:
        Updated state with validated_triples and corrections_made
    """
    config = get_config()
    raw_triples: List[Triple] = state.get("raw_triples", [])

    if not raw_triples:
        logger.warning("No raw triples to review")
        state["validated_triples"] = []
        state["corrections_made"] = []
        return state

    logger.info("Starting reviewer agent")
    logger.info(f"Number of raw triples to review: {len(raw_triples)}")

    # Step 1: Basic normalization and deduplication
    logger.info("Step 1: Normalizing entities and detecting duplicates...")
    
    # Collect all unique entities for normalization
    entity_shortening_cache: Dict[str, str] = {}
    
    # Extract all unique entities from triples
    all_entities: Set[str] = set()
    for triple in raw_triples:
        all_entities.add(triple.subject)
        all_entities.add(triple.object)
    
    # Apply rule-based normalization first (always)
    rule_normalized: Dict[str, str] = {}
    for entity in all_entities:
        rule_normalized[entity] = _normalize_entity(entity)
    
    if config.use_llm_entity_shortening:
        # Check which entities still need LLM shortening
        # Map normalized entities to their originals
        normalized_to_original: Dict[str, str] = {}
        entities_needing_shortening_normalized = []
        
        for original, normalized in rule_normalized.items():
            if len(normalized.split()) > config.max_entity_words:
                entities_needing_shortening_normalized.append(normalized)
                normalized_to_original[normalized] = original
        
        if entities_needing_shortening_normalized:
            logger.info(
                f"Found {len(entities_needing_shortening_normalized)} entities needing LLM shortening "
                f"(>{config.max_entity_words} words)"
            )
            
            # Create client for entity shortening
            shortening_client = OpenRouterClient(
                api_key=config.openrouter_api_key,
                model=config.reviewer_model,
                base_url=config.openrouter_base_url,
            )
            
            try:
                async with shortening_client:
                    # Batch process normalized entities for shortening
                    batch_size_shortening = 20  # Smaller batches for shortening
                    shortened_batch_normalized = {}
                    
                    for i in range(0, len(entities_needing_shortening_normalized), batch_size_shortening):
                        batch = entities_needing_shortening_normalized[i:i + batch_size_shortening]
                        batch_shortened = await _shorten_entities_batch(
                            batch,
                            shortening_client,
                            max_words=config.max_entity_words,
                        )
                        shortened_batch_normalized.update(batch_shortened)
                    
                    # Map shortened normalized entities back to original entities
                    for normalized, shortened in shortened_batch_normalized.items():
                        original = normalized_to_original.get(normalized, normalized)
                        entity_shortening_cache[original] = shortened
                    
                    # Also cache rule-normalized entities that don't need shortening
                    for entity, normalized in rule_normalized.items():
                        if entity not in entity_shortening_cache:
                            entity_shortening_cache[entity] = normalized
                    
                    logger.info(
                        f"LLM shortening completed: {len(shortened_batch_normalized)} entities processed"
                    )
            except Exception as e:
                logger.error(f"Error during LLM entity shortening: {e}", exc_info=True)
                logger.warning("Falling back to rule-based normalization only")
                # Fallback: use rule-normalized entities
                entity_shortening_cache = rule_normalized
        else:
            logger.info("All entities are already ≤ {} words, skipping LLM shortening".format(
                config.max_entity_words
            ))
            # Use rule-normalized entities
            entity_shortening_cache = rule_normalized
    else:
        # LLM shortening disabled, use rule-based normalization only
        entity_shortening_cache = rule_normalized
    
    # Normalize triples using cache
    normalized_triples = _normalize_triples_with_cache(raw_triples, entity_shortening_cache)
    deduplicated_triples = _deduplicate_triples(normalized_triples)

    logger.info(f"After normalization and deduplication: {len(deduplicated_triples)} triples")
    
    # Log statistics about entity lengths
    if config.use_llm_entity_shortening:
        entity_lengths = []
        for triple in deduplicated_triples:
            entity_lengths.append(len(triple.subject.split()))
            entity_lengths.append(len(triple.object.split()))
        
        if entity_lengths:
            max_length = max(entity_lengths)
            avg_length = sum(entity_lengths) / len(entity_lengths)
            logger.info(
                f"Entity length statistics: max={max_length} words, "
                f"avg={avg_length:.1f} words, target≤{config.max_entity_words} words"
            )

    # Step 2: LLM-based validation and correction (in batches)
    logger.info("Step 2: Validating triples with LLM...")
    batch_size = 50
    validated_triples: List[Triple] = []
    corrections_made: List[Dict] = []

    client = OpenRouterClient(
        api_key=config.openrouter_api_key,
        model=config.reviewer_model,
        base_url=config.openrouter_base_url,
    )

    # Process in batches
    for i in tqdm(range(0, len(deduplicated_triples), batch_size), desc="Validating batches"):
        batch = deduplicated_triples[i : i + batch_size]
        try:
            async with client:
                logger.debug(
                    f"Validating batch {i // batch_size}: size={len(batch)}, model={client.model}"
                )
                logger.debug("Calling OpenRouter API for reviewer batch validation")
                batch_validated, batch_corrections = await _validate_batch(batch, client)
                validated_triples.extend(batch_validated)
                corrections_made.extend(batch_corrections)
        except Exception as e:
            logger.error(f"Error validating batch {i // batch_size}: {e}")
            # Fallback: use triples as-is
            validated_triples.extend(batch)

    logger.info(f"Validation completed: {len(validated_triples)} validated triples")
    logger.info(f"Corrections made: {len(corrections_made)}")

    # Update state
    state["validated_triples"] = validated_triples
    state["corrections_made"] = corrections_made

    # Save reviewer output for inspection
    try:
        save_agent_output(
            "reviewer",
            {
                "validated_triples": [t.dict() for t in validated_triples],
                "corrections_made": corrections_made,
            },
        )
    except Exception:
        logger.exception("Failed to save reviewer output")

    return state


def _normalize_triples(triples: List[Triple]) -> List[Triple]:
    """
    Normalize entity names in triples (without cache).

    Args:
        triples: List of triples to normalize

    Returns:
        List of normalized triples
    """
    return _normalize_triples_with_cache(triples, None)


def _normalize_triples_with_cache(
    triples: List[Triple],
    entity_cache: Dict[str, str] | None = None,
) -> List[Triple]:
    """
    Normalize entity names in triples using optional cache.

    Args:
        triples: List of triples to normalize
        entity_cache: Optional cache dictionary for shortened entities

    Returns:
        List of normalized triples
    """
    normalized: List[Triple] = []
    for triple in triples:
        # Use cache if available, otherwise use rule-based normalization
        if entity_cache:
            subject = entity_cache.get(triple.subject, triple.subject)
            object_entity = entity_cache.get(triple.object, triple.object)
        else:
            subject = _normalize_entity(triple.subject)
            object_entity = _normalize_entity(triple.object)
        
        normalized_triple = Triple(
            subject=subject,
            predicate=triple.predicate.upper().strip(),
            object=object_entity,
            confidence=triple.confidence,
            source_chunk_id=triple.source_chunk_id,
            metadata=triple.metadata,
        )
        normalized.append(normalized_triple)
    return normalized


def _deduplicate_triples(triples: List[Triple]) -> List[Triple]:
    """
    Remove duplicate triples.

    Args:
        triples: List of triples to deduplicate

    Returns:
        List of unique triples
    """
    # Use (subject, predicate, object) as key
    seen = set()
    unique_triples: List[Triple] = []

    for triple in triples:
        key = (triple.subject.lower(), triple.predicate.lower(), triple.object.lower())
        if key not in seen:
            seen.add(key)
            unique_triples.append(triple)
        else:
            # Merge metadata from duplicate
            for existing in unique_triples:
                if (
                    existing.subject.lower() == triple.subject.lower()
                    and existing.predicate.lower() == triple.predicate.lower()
                    and existing.object.lower() == triple.object.lower()
                ):
                    # Update confidence (average)
                    existing.confidence = (existing.confidence + triple.confidence) / 2
                    # Merge source chunk IDs
                    if "source_chunks" not in existing.metadata:
                        existing.metadata["source_chunks"] = [existing.source_chunk_id]
                    if triple.source_chunk_id not in existing.metadata["source_chunks"]:
                        existing.metadata["source_chunks"].append(triple.source_chunk_id)
                    break

    return unique_triples


async def _validate_batch(
    batch: List[Triple],
    client: OpenRouterClient,
) -> tuple[List[Triple], List[Dict]]:
    """
    Validate a batch of triples using LLM.

    Args:
        batch: List of triples to validate
        client: OpenRouter client

    Returns:
        Tuple of (validated_triples, corrections_made)
    """
    # Convert triples to JSON for LLM
    triples_json = [
        {
            "subject": t.subject,
            "predicate": t.predicate,
            "object": t.object,
        }
        for t in batch
    ]

    validation_prompt = f"""You are a knowledge graph validator. Review and correct the following triples.

TRIPLES TO VALIDATE:
{json.dumps(triples_json, indent=2)}

TASK:
1. Check each triple for:
   - Entity name consistency (e.g., "NYC" should be "New York City" if that's the canonical form)
   - Predicate accuracy and clarity
   - Factual correctness (based on common knowledge)
   - Format consistency

2. For each triple, either:
   - Keep it as-is if correct
   - Correct entity names, predicates, or objects if needed
   - Mark as invalid if it's clearly wrong or nonsensical

Return a JSON object with this structure:
{{
  "validated_triples": [
    {{
      "subject": "Corrected subject",
      "predicate": "Corrected predicate",
      "object": "Corrected object",
      "confidence": 0.95,
      "correction": "What was corrected (or 'none' if unchanged)"
    }},
    ...
  ]
}}

GUIDELINES:
- Normalize entity names to canonical forms
- Use consistent predicate naming (UPPER_CASE)
- Remove or correct invalid triples
- Preserve valid triples as-is
- Confidence should reflect certainty (0.0-1.0)

Now validate the triples:"""

    try:
        system_prompt = "You are a knowledge graph validator. Always respond with valid JSON only, no additional text."
        response = await client.generate(
            prompt=validation_prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        # Parse response
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
                result = json.loads(response)
            else:
                logger.warning("Could not parse validation response, using original triples")
                return batch, []

        validated_data = result.get("validated_triples", [])
        validated_triples: List[Triple] = []
        corrections: List[Dict] = []

        for i, validated_item in enumerate(validated_data):
            if i < len(batch):
                original = batch[i]
                correction = validated_item.get("correction", "none")

                validated_triple = Triple(
                    subject=validated_item.get("subject", original.subject),
                    predicate=validated_item.get("predicate", original.predicate),
                    object=validated_item.get("object", original.object),
                    confidence=validated_item.get("confidence", original.confidence),
                    source_chunk_id=original.source_chunk_id,
                    metadata=original.metadata,
                )

                validated_triples.append(validated_triple)

                if correction and correction.lower() != "none":
                    corrections.append({
                        "original": {
                            "subject": original.subject,
                            "predicate": original.predicate,
                            "object": original.object,
                        },
                        "corrected": {
                            "subject": validated_triple.subject,
                            "predicate": validated_triple.predicate,
                            "object": validated_triple.object,
                        },
                        "correction": correction,
                    })

        return validated_triples, corrections

    except Exception as e:
        logger.error(f"Error in batch validation: {e}", exc_info=True)
        # Fallback: return original triples
        return batch, []



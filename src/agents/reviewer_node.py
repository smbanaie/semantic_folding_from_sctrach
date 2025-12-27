"""
Reviewer Agent - Phase 5 of the knowledge graph pipeline.

Validates, normalizes, and deduplicates triples extracted by the Extractor agent.
"""

import json
import logging
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List

from tqdm import tqdm

from src.config import get_config
from src.models.data_models import GraphState, Triple
from src.utils.openrouter_client import OpenRouterClient

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


def _normalize_entity(entity: str) -> str:
    """
    Basic entity normalization.

    Args:
        entity: Entity string to normalize

    Returns:
        Normalized entity string
    """
    # Remove extra whitespace
    entity = " ".join(entity.split())
    # Capitalize first letter of each word (for proper nouns)
    # But preserve acronyms (all caps)
    if entity.isupper() or len(entity) <= 3:
        return entity
    return entity.title()


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
    normalized_triples = _normalize_triples(raw_triples)
    deduplicated_triples = _deduplicate_triples(normalized_triples)

    logger.info(f"After normalization and deduplication: {len(deduplicated_triples)} triples")

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

    return state


def _normalize_triples(triples: List[Triple]) -> List[Triple]:
    """
    Normalize entity names in triples.

    Args:
        triples: List of triples to normalize

    Returns:
        List of normalized triples
    """
    normalized: List[Triple] = []
    for triple in triples:
        normalized_triple = Triple(
            subject=_normalize_entity(triple.subject),
            predicate=triple.predicate.upper().strip(),
            object=_normalize_entity(triple.object),
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



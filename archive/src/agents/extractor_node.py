"""
Extractor Agent - Phase 4 of the knowledge graph pipeline.

Extracts knowledge triples from chunks in parallel, using LLM
to identify entities and relationships.
"""

import asyncio
import json
import logging
from typing import Dict, List

from tqdm import tqdm

from src.config import get_config
from src.models.data_models import Chunk, GraphState, Triple
from src.utils.openrouter_client import OpenRouterClient
from src.utils.io_utils import save_agent_output

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


async def extract_entities_openie(
    text: str,
    client: OpenRouterClient,
) -> List[str]:
    """
    First step of OpenIE: Extract named entities from text.
    This follows HippoRAG2's methodology of separate entity extraction.

    Args:
        text: Text to extract entities from
        client: OpenRouter client

    Returns:
        List of extracted entity names
    """
    entity_extraction_prompt = f"""Extract all named entities and key concepts from the following text.

TASK: Identify all proper nouns, organizations, locations, persons, technologies, and other significant entities mentioned in the text.

Return a JSON array of entity strings. Focus on:
- Proper nouns (people, organizations, products, locations)
- Technical terms and concepts
- Domain-specific entities
- Avoid common nouns unless they represent specific concepts

EXAMPLE:
Text: "Professor Thomas researches Alzheimer's at Stanford University using advanced neural networks."
Output: ["Professor Thomas", "Alzheimer's", "Stanford University", "neural networks"]

TEXT TO ANALYZE:
{text}

Return only a JSON array of strings:"""

    try:
        system_prompt = "You are an entity extraction expert. Always respond with valid JSON array of entity strings only."

        response = await client.generate(
            prompt=entity_extraction_prompt,
            system_prompt=system_prompt,
            temperature=0.1,  # Low temperature for consistency
            response_format={"type": "json_object"},
        )

        # Parse response
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                entities = parsed
            elif isinstance(parsed, dict) and "entities" in parsed:
                entities = parsed["entities"]
            else:
                entities = []
        except json.JSONDecodeError:
            logger.warning(f"Could not parse entities JSON: {response[:200]}")
            entities = []

        # Filter and clean entities
        cleaned_entities = []
        for entity in entities:
            if isinstance(entity, str) and len(entity.strip()) > 0:
                cleaned_entity = entity.strip()
                # Skip very short entities (likely noise)
                if len(cleaned_entity) > 2:
                    cleaned_entities.append(cleaned_entity)

        return cleaned_entities

    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        return []


async def extract_relations_openie(
    text: str,
    entities: List[str],
    extraction_prompts: Dict[str, str],
    client: OpenRouterClient,
) -> List[Dict]:
    """
    Second step of OpenIE: Extract relations between entities.
    Uses the extracted entities as context for more accurate relation extraction.

    Args:
        text: Original text
        entities: List of entities extracted in first step
        extraction_prompts: Prompts for relationship extraction
        client: OpenRouter client

    Returns:
        List of triple dictionaries
    """
    entities_str = ", ".join(f'"{e}"' for e in entities)

    relationship_prompt = extraction_prompts.get(
        "relationship_extraction", "Extract relationships between entities."
    )

    relation_extraction_prompt = f"""Extract knowledge triples using Open Information Extraction (OpenIE) methodology.

IDENTIFIED ENTITIES: [{entities_str}]

RELATIONSHIP EXTRACTION GUIDELINES:
{relationship_prompt}

TEXT TO ANALYZE:
{text}

TASK:
Extract all knowledge triples (subject, predicate, object) where the subject and object are from the identified entities or related concepts in the text.

Requirements:
- Subjects and objects should be noun phrases (entities or concepts)
- Predicates should be clear relationship types in UPPER_CASE
- Only extract relations that are explicitly stated or clearly implied in the text
- Use the identified entities as anchors but also extract additional relevant concepts
- Focus on factual relationships, not opinions or speculations

Return a JSON array of triples with this structure:
[
  {{
    "subject": "Entity or concept from text",
    "predicate": "RELATIONSHIP_TYPE",
    "object": "Related entity or concept"
  }}
]

EXAMPLE:
Entities: ["OpenAI", "GPT-4", "large language model"]
Text: "OpenAI developed GPT-4, a large language model that many companies use."
Output:
[
  {{"subject": "OpenAI", "predicate": "DEVELOPED", "object": "GPT-4"}},
  {{"subject": "GPT-4", "predicate": "IS_TYPE_OF", "object": "large language model"}},
  {{"subject": "GPT-4", "predicate": "USED_BY", "object": "many companies"}}
]

Now extract triples from the provided text:"""

    try:
        system_prompt = "You are a relation extraction expert using OpenIE methodology. Always respond with valid JSON array of triples only."

        response = await client.generate(
            prompt=relation_extraction_prompt,
            system_prompt=system_prompt,
            temperature=0.2,  # Slightly higher for relation creativity
            response_format={"type": "json_object"},
        )

        # Parse response
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                triples_data = parsed
            elif isinstance(parsed, dict) and "triples" in parsed:
                triples_data = parsed["triples"]
            else:
                triples_data = []
        except json.JSONDecodeError:
            logger.warning(f"Could not parse relations JSON: {response[:200]}")
            triples_data = []

        return triples_data

    except Exception as e:
        logger.error(f"Error extracting relations: {e}")
        return []


async def extract_from_chunk_openie(
    chunk: Chunk,
    extraction_prompts: Dict[str, str],
    client: OpenRouterClient,
) -> List[Triple]:
    """
    OpenIE-based triple extraction from a chunk using HippoRAG2's two-step methodology.

    Step 1: Extract entities
    Step 2: Extract relations using entities as context

    Args:
        chunk: Chunk to extract from
        extraction_prompts: Prompts for entity and relationship extraction
        client: OpenRouter client

    Returns:
        List of extracted triples
    """
    try:
        # Step 1: Extract entities
        entities = await extract_entities_openie(chunk.content, client)

        if not entities:
            logger.warning(f"No entities found in chunk {chunk.id}, falling back to standard extraction")
            # Fallback to original method if no entities found
            return await extract_from_chunk(chunk, extraction_prompts, client)

        logger.debug(f"Extracted {len(entities)} entities from chunk {chunk.id}: {entities[:5]}...")

        # Step 2: Extract relations using entities as context
        triples_data = await extract_relations_openie(
            chunk.content, entities, extraction_prompts, client
        )

        # Convert to Triple objects
        triples: List[Triple] = []
        for triple_data in triples_data:
            if not isinstance(triple_data, dict):
                continue

            subject = triple_data.get("subject", "").strip()
            predicate = triple_data.get("predicate", "").strip()
            obj = triple_data.get("object", "").strip()

            if subject and predicate and obj:
                triple = Triple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence=1.0,
                    source_chunk_id=chunk.id,
                    metadata={
                        "extraction_method": "openie",
                        "entities_used": entities,
                    },
                )
                triples.append(triple)

        logger.debug(f"Extracted {len(triples)} triples using OpenIE from chunk {chunk.id}")
        return triples

    except Exception as e:
        logger.error(f"Error in OpenIE extraction from chunk {chunk.id}: {e}", exc_info=True)
        # Fallback to original method on error
        return await extract_from_chunk(chunk, extraction_prompts, client)


async def extract_from_chunk(
    chunk: Chunk,
    extraction_prompts: Dict[str, str],
    client: OpenRouterClient,
) -> List[Triple]:
    """
    Extract triples from a single chunk.

    Args:
        chunk: Chunk to extract from
        extraction_prompts: Prompts for entity and relationship extraction
        client: OpenRouter client

    Returns:
        List of extracted triples
    """
    entity_prompt = extraction_prompts.get("entity_extraction", "Extract entities from the text.")
    relationship_prompt = extraction_prompts.get(
        "relationship_extraction", "Extract relationships between entities."
    )

    # Build extraction prompt
    extraction_prompt = f"""You are a knowledge extraction expert. Extract all knowledge triples from the following text.

ENTITY EXTRACTION GUIDELINES:
{entity_prompt}

RELATIONSHIP EXTRACTION GUIDELINES:
{relationship_prompt}

TEXT TO ANALYZE:
{chunk.content}

TASK:
Extract all knowledge triples in the format (subject, predicate, object) from the text.

Return a JSON array of triples with this structure:
[
  {{
    "subject": "Entity or concept name",
    "predicate": "Relationship type (e.g., DEVELOPED, LOCATED_IN, WORKS_FOR, RELATED_TO)",
    "object": "Related entity or concept"
  }},
  ...
]

GUIDELINES:
- Extract only factual, explicit relationships mentioned in the text
- Use clear, canonical entity names (avoid abbreviations unless standard)
- Predicates should be action verbs or relationship types in UPPER_CASE
- Be specific and accurate
- Include all relevant triples, even if they seem obvious
- If no triples can be extracted, return an empty array []

EXAMPLE:
Text: "OpenAI developed GPT-4, a large language model. GPT-4 is used by many companies."
Output:
[
  {{"subject": "OpenAI", "predicate": "DEVELOPED", "object": "GPT-4"}},
  {{"subject": "GPT-4", "predicate": "IS_TYPE_OF", "object": "large language model"}},
  {{"subject": "GPT-4", "predicate": "USED_BY", "object": "many companies"}}
]

Now extract triples from the provided text:"""

    try:
        system_prompt = "You are a knowledge extraction expert. Always respond with valid JSON array of triples only, no additional text."
        logger.debug(f"Extract-from-chunk: chunk_id={chunk.id}, prompt_len={len(extraction_prompt)} chars, model={client.model}")
        logger.debug("Calling OpenRouter API for extract_from_chunk")
        response = await client.generate(
            prompt=extraction_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        # Parse response
        # Response might be wrapped in JSON object with "triples" key or be direct array
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                triples_data = parsed
            elif isinstance(parsed, dict):
                # Try common keys
                triples_data = parsed.get("triples", parsed.get("data", []))
                if not isinstance(triples_data, list):
                    triples_data = []
            else:
                triples_data = []
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
                parsed = json.loads(response)
                triples_data = parsed if isinstance(parsed, list) else parsed.get("triples", [])
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
                parsed = json.loads(response)
                triples_data = parsed if isinstance(parsed, list) else parsed.get("triples", [])
            else:
                logger.warning(f"Could not parse JSON from chunk {chunk.id}: {response[:200]}")
                return []

        # Convert to Triple objects
        triples: List[Triple] = []
        for triple_data in triples_data:
            if not isinstance(triple_data, dict):
                continue

            subject = triple_data.get("subject", "").strip()
            predicate = triple_data.get("predicate", "").strip()
            obj = triple_data.get("object", "").strip()

            if subject and predicate and obj:
                triple = Triple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence=1.0,
                    source_chunk_id=chunk.id,
                    metadata={},
                )
                triples.append(triple)

        return triples

    except Exception as e:
        logger.error(f"Error extracting from chunk {chunk.id}: {e}", exc_info=True)
        return []


async def extractor_agent(state: GraphState) -> GraphState:
    """
    Extract triples from all chunks in parallel.

    This agent:
    1. Takes chunks and extraction_prompts from state
    2. Processes all chunks in parallel using asyncio.gather
    3. Aggregates results into raw_triples
    4. Adds extraction statistics

    Args:
        state: Current graph state with chunks and extraction_prompts

    Returns:
        Updated state with raw_triples and extraction_stats
    """
    config = get_config()
    chunks: List[Chunk] = state.get("chunks", [])
    extraction_prompts: Dict[str, str] = state.get("extraction_prompts", {})

    if not chunks:
        raise ValueError("No chunks found in state")

    logger.info("Starting extractor agent")
    logger.info(f"Number of chunks to process: {len(chunks)}")

    # Initialize client
    client = OpenRouterClient(
        api_key=config.openrouter_api_key,
        model=config.extractor_model,
        base_url=config.openrouter_base_url,
    )

    # Process chunks in parallel with rate limiting
    semaphore = asyncio.Semaphore(config.max_parallel_extractions)
    raw_triples: List[Triple] = []
    extraction_stats = {
        "total_chunks": len(chunks),
        "chunks_processed": 0,
        "chunks_failed": 0,
        "total_triples": 0,
        "errors": [],
    }

    async def extract_with_semaphore(chunk: Chunk):
        """Extract from chunk with semaphore for rate limiting."""
        async with semaphore:
            try:
                # Use OpenIE methodology for enhanced triple extraction
                triples = await extract_from_chunk_openie(chunk, extraction_prompts, client)
                return triples, None
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.id}: {e}")
                return [], str(e)

    # Process all chunks with progress bar
    logger.info("Processing chunks in parallel...")

    results = []
    # Manage a single client session for all concurrent requests to avoid closing the session
    async with client:
        tasks = [extract_with_semaphore(chunk) for chunk in chunks]

        # Use tqdm for progress tracking
        with tqdm(total=len(chunks), desc="Extracting triples", unit="chunk") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

    # Aggregate results
    for triples, error in results:
        if error:
            extraction_stats["chunks_failed"] += 1
            extraction_stats["errors"].append(error)
        else:
            extraction_stats["chunks_processed"] += 1
            raw_triples.extend(triples)

    extraction_stats["total_triples"] = len(raw_triples)

    logger.info(f"Extraction completed:")
    logger.info(f"  - Chunks processed: {extraction_stats['chunks_processed']}/{extraction_stats['total_chunks']}")
    logger.info(f"  - Chunks failed: {extraction_stats['chunks_failed']}")
    logger.info(f"  - Total triples extracted: {extraction_stats['total_triples']}")

    # Update state
    state["raw_triples"] = raw_triples
    state["extraction_stats"] = extraction_stats

    # Save extractor output for inspection
    try:
        save_agent_output(
            "extractor",
            {
                "extraction_stats": extraction_stats,
                "raw_triples": [t.dict() for t in raw_triples],
            },
        )
    except Exception:
        logger.exception("Failed to save extractor output")

    return state



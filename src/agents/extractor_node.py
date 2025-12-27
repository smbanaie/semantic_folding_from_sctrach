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

logger = logging.getLogger(__name__)


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
                async with client:
                    triples = await extract_from_chunk(chunk, extraction_prompts, client)
                    return triples, None
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.id}: {e}")
                return [], str(e)

    # Process all chunks with progress bar
    logger.info("Processing chunks in parallel...")
    tasks = [extract_with_semaphore(chunk) for chunk in chunks]

    # Use tqdm for progress tracking
    results = []
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

    return state



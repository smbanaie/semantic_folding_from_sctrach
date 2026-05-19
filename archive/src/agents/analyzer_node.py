"""
Analyzer Agent - Phase 1 of the knowledge graph pipeline.

Analyzes the corpus to understand its domain, identify key entity types,
and generate context-aware extraction prompts for downstream agents.
"""

import json
import logging
from typing import Dict

from src.config import get_config
from src.models.data_models import GraphState
from src.utils.openrouter_client import OpenRouterClient
from src.utils.io_utils import save_agent_output

logger = logging.getLogger(__name__)


async def analyzer_agent(state: GraphState) -> GraphState:
    """
    Analyze corpus and generate extraction strategy.

    This agent:
    1. Analyzes the corpus to determine domain/topic
    2. Identifies key entity types likely present
    3. Generates domain-specific extraction prompts
    4. Creates extraction strategy for downstream agents

    Args:
        state: Current graph state containing corpus

    Returns:
        Updated state with extraction_strategy, extraction_prompts, and domain_context
    """
    config = get_config()
    corpus = state.get("corpus", "")
    corpus_metadata = state.get("corpus_metadata", {})

    if not corpus:
        raise ValueError("Corpus is empty in state")

    logger.info("Starting analyzer agent")
    logger.info(f"Corpus length: {len(corpus)} characters")

    # Prefer using representative samples from sections when available to limit LLM input size
    sections = state.get("sections", [])
    sample_text = None
    try:
        if sections:
            collected = []
            # take up to 2 representative samples per section, stop when reaching ~4000 chars
            max_chars = 4000
            for s in sections:
                reps = None
                try:
                    # pydantic model attribute
                    reps = s.representative_samples
                except Exception:
                    # dict-like section
                    reps = s.get("representative_samples") if isinstance(s, dict) else None
                if reps:
                    for r in reps[:2]:
                        if sum(len(x) for x in collected) + len(r) > max_chars:
                            break
                        collected.append(r)
                if sum(len(x) for x in collected) >= max_chars:
                    break
            if collected:
                sample_text = "\n\n".join(collected)
    except Exception:
        sample_text = None

    if sample_text:
        analysis_corpus_snippet = sample_text
        snippet_note = f"SAMPLES FROM SECTIONS (up to {len(analysis_corpus_snippet)} chars)"
    else:
        analysis_corpus_snippet = corpus[:5000]
        snippet_note = "CORPUS (first 5000 characters)"

    # Build analysis prompt
    analysis_prompt = f"""You are an expert knowledge extraction analyst. Analyze the following text corpus samples and provide a comprehensive extraction strategy.

{snippet_note}:
{analysis_corpus_snippet}

CORPUS STATISTICS:
- Total length: {len(corpus)} characters
- Estimated words: ~{len(corpus.split())} words

TASK:
Analyze this corpus and provide a JSON response with the following structure:
{{
  "domain": "scientific|news|literature|technical|general",
  "topic": "Brief topic description",
  "key_entity_types": ["Person", "Organization", "Location", "Technology", "Concept", ...],
  "common_relationship_types": ["DEVELOPED", "LOCATED_IN", "WORKS_FOR", "RELATED_TO", ...],
  "extraction_strategy": "Detailed paragraph explaining how to extract knowledge from this corpus",
  "entity_extraction_prompt": "Specific prompt for extracting entities from this domain",
  "relationship_extraction_prompt": "Specific prompt for extracting relationships from this domain",
  "domain_context": "Brief context about the domain and its characteristics"
}}

GUIDELINES:
- Think step-by-step about the corpus content
- Identify the primary domain and topic
- List entity types that are likely to appear (e.g., Person, Organization, Technology, Concept)
- Identify common relationship patterns (e.g., DEVELOPED, LOCATED_IN, WORKS_FOR)
- Create domain-specific prompts that will help extractors understand what to look for
- The extraction prompts should be clear, specific, and tailored to this corpus

Now analyze the provided corpus samples and return your JSON response:"""

    # Initialize OpenRouter client
    client = OpenRouterClient(
        api_key=config.openrouter_api_key,
        model=config.analyzer_model,
        base_url=config.openrouter_base_url,
    )

    # Debug: log prompt size after it's built
    logger.debug(f"Analyzer prompt length: {len(analysis_prompt)} chars; model={config.analyzer_model}")

    try:
        async with client:
            logger.info(f"Calling {config.analyzer_model} for analysis")
            logger.debug("Calling OpenRouter API for analyzer_agent")
            system_prompt = "You are an expert knowledge extraction analyst. Always respond with valid JSON only, no additional text."
            response = await client.generate(
                prompt=analysis_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            # Parse JSON response
            try:
                analysis_result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks if present
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    response = response[json_start:json_end].strip()
                    analysis_result = json.loads(response)
                elif "```" in response:
                    json_start = response.find("```") + 3
                    json_end = response.find("```", json_start)
                    response = response[json_start:json_end].strip()
                    analysis_result = json.loads(response)
                else:
                    raise ValueError(f"Could not parse JSON from response: {response[:200]}")

            logger.info("Analysis completed successfully")
            logger.info(f"Domain: {analysis_result.get('domain', 'unknown')}")
            logger.info(f"Topic: {analysis_result.get('topic', 'unknown')}")

            # Update state
            state["extraction_strategy"] = analysis_result.get(
                "extraction_strategy", "Extract entities and relationships from the corpus."
            )
            state["extraction_prompts"] = {
                "entity_extraction": analysis_result.get(
                    "entity_extraction_prompt",
                    "Extract all entities from the text.",
                ),
                "relationship_extraction": analysis_result.get(
                    "relationship_extraction_prompt",
                    "Extract relationships between entities.",
                ),
            }
            state["domain_context"] = analysis_result.get(
                "domain_context", "General domain corpus."
            )

            # Add analysis metadata to corpus_metadata
            corpus_metadata["analysis"] = {
                "domain": analysis_result.get("domain"),
                "topic": analysis_result.get("topic"),
                "key_entity_types": analysis_result.get("key_entity_types", []),
                "common_relationship_types": analysis_result.get(
                    "common_relationship_types", []
                ),
            }
            state["corpus_metadata"] = corpus_metadata

            # Save analyzer output for inspection
            try:
                save_agent_output(
                    "analyzer",
                    {
                        "analysis_result": analysis_result,
                        "extraction_strategy": state["extraction_strategy"],
                        "extraction_prompts": state["extraction_prompts"],
                        "domain_context": state["domain_context"],
                        "corpus_metadata": state.get("corpus_metadata", {}),
                    },
                )
            except Exception:
                logger.exception("Failed to save analyzer output")

            return state

    except Exception as e:
        logger.error(f"Error in analyzer agent: {e}", exc_info=True)
        # Fallback to default strategy
        logger.warning("Using fallback extraction strategy")
        state["extraction_strategy"] = (
            "Extract entities and relationships from the corpus. "
            "Focus on named entities (people, organizations, locations) and their relationships."
        )
        state["extraction_prompts"] = {
            "entity_extraction": "Extract all entities from the text, including people, organizations, locations, and concepts.",
            "relationship_extraction": "Extract relationships between entities. Common patterns include: works_for, located_in, related_to, developed_by.",
        }
        state["domain_context"] = "General domain corpus."
        return state



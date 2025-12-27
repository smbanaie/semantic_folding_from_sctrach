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

    # Build analysis prompt
    analysis_prompt = f"""You are an expert knowledge extraction analyst. Analyze the following text corpus and provide a comprehensive extraction strategy.

CORPUS (first 5000 characters):
{corpus[:5000]}

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

EXAMPLE OUTPUT FORMAT:
{{
  "domain": "technical",
  "topic": "Artificial Intelligence and Machine Learning",
  "key_entity_types": ["Technology", "Organization", "Person", "Concept", "Method"],
  "common_relationship_types": ["DEVELOPED", "USES", "IMPLEMENTS", "RELATED_TO", "CREATED_BY"],
  "extraction_strategy": "Focus on extracting technologies, organizations developing them, key researchers, and relationships between concepts. Pay attention to technical terminology and domain-specific relationships.",
  "entity_extraction_prompt": "Extract all entities from the text, focusing on: technologies (AI models, frameworks, tools), organizations (companies, research labs), people (researchers, developers), and concepts (algorithms, methodologies). Use full canonical names when possible.",
  "relationship_extraction_prompt": "Extract relationships between entities. Common patterns include: technologies DEVELOPED by organizations, concepts USED in technologies, people WORK_FOR organizations, technologies IMPLEMENT concepts. Be specific and accurate.",
  "domain_context": "This corpus discusses AI/ML technologies, their development, applications, and relationships between various components of the AI ecosystem."
}}

Now analyze the provided corpus and return your JSON response:"""

    # Initialize OpenRouter client
    client = OpenRouterClient(
        api_key=config.openrouter_api_key,
        model=config.analyzer_model,
        base_url=config.openrouter_base_url,
    )

    try:
        async with client:
            logger.info(f"Calling {config.analyzer_model} for analysis")
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



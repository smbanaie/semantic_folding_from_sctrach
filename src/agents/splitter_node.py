"""
Splitter Agent - Phase 2 of the knowledge graph pipeline.

Divides the corpus into logical sections based on semantic boundaries,
enabling parallel processing in downstream stages.
"""

import json
import logging
from typing import List

from src.config import get_config
from src.models.data_models import GraphState, Section
from src.utils.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


async def splitter_agent(state: GraphState) -> GraphState:
    """
    Split corpus into logical sections.

    This agent:
    1. Analyzes corpus to identify logical boundaries
    2. Splits corpus into 3-10 sections (depending on length)
    3. Creates Section objects with metadata
    4. Ensures sections maintain semantic coherence

    Args:
        state: Current graph state with corpus and extraction_strategy

    Returns:
        Updated state with sections populated
    """
    config = get_config()
    corpus = state.get("corpus", "")
    extraction_strategy = state.get("extraction_strategy", "")

    if not corpus:
        raise ValueError("Corpus is empty in state")

    logger.info("Starting splitter agent")
    logger.info(f"Corpus length: {len(corpus)} characters")

    # Determine target number of sections based on corpus length
    # Target: 3-10 sections, roughly 5000-15000 chars per section
    corpus_length = len(corpus)
    if corpus_length < 10000:
        target_sections = 3
    elif corpus_length < 50000:
        target_sections = 5
    elif corpus_length < 100000:
        target_sections = 7
    else:
        target_sections = 10

    logger.info(f"Target sections: {target_sections}")

    # Build splitting prompt
    split_prompt = f"""You are an expert text analyzer. Divide the following text into logical sections.

EXTRACTION STRATEGY:
{extraction_strategy}

TEXT TO SPLIT:
{corpus}

TASK:
Analyze the text and identify {target_sections} logical section boundaries. Sections should:
- Be semantically coherent (each section covers a distinct topic or theme)
- Maintain natural boundaries (at paragraph breaks when possible)
- Be roughly balanced in length (aim for similar sizes)
- Not break mid-sentence or mid-paragraph

Return a JSON response with this structure:
{{
  "sections": [
    {{
      "id": "section_0",
      "content": "Full text content of section 0...",
      "topic": "Brief topic description",
      "estimated_entities": 10,
      "position": 0
    }},
    {{
      "id": "section_1",
      "content": "Full text content of section 1...",
      "topic": "Brief topic description",
      "estimated_entities": 12,
      "position": 1
    }}
  ]
}}

IMPORTANT:
- Include the COMPLETE text content for each section (no truncation)
- Ensure all sections together contain the ENTIRE original text (no content loss)
- Section IDs should be: section_0, section_1, section_2, etc.
- Position should be 0, 1, 2, etc. (sequential)

Now analyze and split the text:"""

    # Initialize OpenRouter client
    client = OpenRouterClient(
        api_key=config.openrouter_api_key,
        model=config.splitter_model,
        base_url=config.openrouter_base_url,
    )

    sections: List[Section] = []
    section_metadata: List[dict] = []

    try:
        async with client:
            logger.info(f"Calling {config.splitter_model} for splitting")
            system_prompt = "You are an expert text analyzer. Always respond with valid JSON only. Ensure all text content is included in sections."
            response = await client.generate(
                prompt=split_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            # Parse JSON response
            try:
                split_result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    response = response[json_start:json_end].strip()
                    split_result = json.loads(response)
                elif "```" in response:
                    json_start = response.find("```") + 3
                    json_end = response.find("```", json_start)
                    response = response[json_start:json_end].strip()
                    split_result = json.loads(response)
                else:
                    raise ValueError(f"Could not parse JSON from response: {response[:200]}")

            # Create Section objects
            section_list = split_result.get("sections", [])
            if not section_list:
                raise ValueError("No sections returned from splitter")

            current_pos = 0
            for idx, section_data in enumerate(section_list):
                section_id = section_data.get("id", f"section_{idx}")
                content = section_data.get("content", "")
                topic = section_data.get("topic", f"Section {idx}")
                estimated_entities = section_data.get("estimated_entities", 0)
                position = section_data.get("position", idx)

                if not content:
                    logger.warning(f"Section {section_id} has empty content, skipping")
                    continue

                # Calculate positions in original corpus
                start_pos = current_pos
                end_pos = current_pos + len(content)

                section = Section(
                    id=section_id,
                    content=content,
                    metadata={
                        "topic": topic,
                        "estimated_entities": estimated_entities,
                        "position": position,
                    },
                    start_pos=start_pos,
                    end_pos=end_pos,
                )

                sections.append(section)
                section_metadata.append({
                    "id": section_id,
                    "topic": topic,
                    "estimated_entities": estimated_entities,
                    "position": position,
                    "length": len(content),
                })

                current_pos = end_pos

            logger.info(f"Successfully created {len(sections)} sections")
            logger.info(f"Total section content length: {sum(len(s.content) for s in sections)} chars")
            logger.info(f"Original corpus length: {len(corpus)} chars")

            # Validate: ensure we didn't lose content
            total_section_length = sum(len(s.content) for s in sections)
            if total_section_length < len(corpus) * 0.8:  # Allow 20% tolerance for formatting
                logger.warning(
                    f"Section content ({total_section_length} chars) is significantly less than "
                    f"original corpus ({len(corpus)} chars). Content may have been lost."
                )

    except Exception as e:
        logger.error(f"Error in splitter agent: {e}", exc_info=True)
        # Fallback: split by character count
        logger.warning("Using fallback character-based splitting")
        sections = _fallback_split(corpus, target_sections)

    # Update state
    state["sections"] = sections
    state["section_metadata"] = section_metadata

    return state


def _fallback_split(corpus: str, target_sections: int) -> List[Section]:
    """
    Fallback splitting method using character count.

    Args:
        corpus: Text to split
        target_sections: Target number of sections

    Returns:
        List of Section objects
    """
    logger.info("Using fallback character-based splitting")
    sections: List[Section] = []
    section_size = len(corpus) // target_sections

    for idx in range(target_sections):
        start_idx = idx * section_size
        if idx == target_sections - 1:
            # Last section gets remaining content
            end_idx = len(corpus)
        else:
            # Try to break at paragraph boundary
            end_idx = start_idx + section_size
            # Look for paragraph break within 500 chars
            for i in range(end_idx, min(end_idx + 500, len(corpus))):
                if corpus[i : i + 2] == "\n\n":
                    end_idx = i + 2
                    break

        content = corpus[start_idx:end_idx].strip()
        if content:
            section = Section(
                id=f"section_{idx}",
                content=content,
                metadata={
                    "topic": f"Section {idx}",
                    "estimated_entities": 0,
                    "position": idx,
                },
                start_pos=start_idx,
                end_pos=end_idx,
            )
            sections.append(section)

    return sections



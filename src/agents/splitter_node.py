"""
Splitter Agent - Phase 2 of the knowledge graph pipeline.

Divides the corpus into logical sections based on semantic boundaries,
enabling parallel processing in downstream stages.
"""

import json
import logging
from pathlib import Path
from typing import List

from src.config import get_config
from src.models.data_models import GraphState, Section
from src.utils.openrouter_client import OpenRouterClient
from src.utils.io_utils import save_agent_output
from src.utils.embeddings import paragraphize, compute_embeddings
from src.utils.emb_cache import EmbeddingCache
from src.utils.io_utils import get_run_id

# Import KMeans for semantic splitting fallback
try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

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

    # Debug: log prompt size after it's built
    logger.debug(f"Splitter prompt length: {len(split_prompt)} chars; model={config.splitter_model}")

    sections: List[Section] = []
    section_metadata: List[dict] = []

    # Use quick boundary-based splitting by default for large textbooks
    # This method is much faster and doesn't require resource-intensive embeddings
    if config.use_quick_splitting:
        try:
            sections = _quick_split_by_boundaries(corpus, target_sections)
            logger.info(f"Quick splitter created {len(sections)} sections using boundary analysis")
        except Exception as e:
            logger.warning(f"Quick boundary splitting failed: {e}", exc_info=True)
            # Fall through to semantic splitting below
            sections = []
    else:
        sections = []
        
        # Create section metadata
        for i, section in enumerate(sections):
            section_metadata.append({
                "id": section.id,
                "topic": section.metadata.get("topic", f"Section {i}"),
                "estimated_entities": section.metadata.get("estimated_entities", 0),
                "position": section.metadata.get("position", i),
                "length": len(section.content),
                "boundary_type": section.metadata.get("boundary_type", "unknown"),
            })
            
        try:
            save_agent_output("splitter", {"sections": [s.dict() for s in sections], "section_metadata": section_metadata})
        except Exception:
            logger.exception("Failed to save splitter output after quick split")

    # If quick splitting wasn't used or failed, fall back to semantic splitting
    if not sections:
        try:
            logger.info("Falling back to semantic splitting with embeddings")
            paras = paragraphize(corpus)
            if not paras:
                raise ValueError("No paragraphs found for semantic splitting")

            # Map paragraph positions in original corpus
            para_positions = []
            search_pos = 0
            for p in paras:
                idx = corpus.find(p, search_pos)
                if idx == -1:
                    idx = search_pos
                para_positions.append((idx, idx + len(p)))
                search_pos = idx + len(p)

            # Prepare embedding cache for this run
            run_id = get_run_id()
            cache_path = Path(__file__).resolve().parents[2] / "agents" / "output" / run_id / "splitter_emb_cache.json"
            emb_cache = EmbeddingCache(cache_path)

            # Collect embeddings, using cache where possible
            texts_to_compute = []
            compute_idxs = []
            cached_embs = [None] * len(paras)
            for i, p in enumerate(paras):
                e = emb_cache.get(p)
                if e is None:
                    texts_to_compute.append(p)
                    compute_idxs.append(i)
                else:
                    cached_embs[i] = e

            if texts_to_compute:
                new_embs = compute_embeddings(texts_to_compute)
                for idx, emb in zip(compute_idxs, new_embs):
                    emb_cache.set(paras[idx], emb)
                    cached_embs[idx] = emb
                emb_cache.save()

            if any(e is None for e in cached_embs):
                raise ValueError("Failed to obtain embeddings for all paragraphs")

            embeddings = cached_embs  # type: ignore[var-annotated]

            # Number of clusters
            n_clusters = min(max(3, target_sections), len(paras))
            if KMeans is None:
                raise RuntimeError("sklearn not available for clustering")

            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(embeddings)

            # Group paragraph indices by cluster
            clusters: dict[int, list[int]] = {}
            for i, lbl in enumerate(labels):
                clusters.setdefault(int(lbl), []).append(i)

            built_sections: List[Section] = []
            for cid, idxs in clusters.items():
                idxs_sorted = sorted(idxs, key=lambda i: para_positions[i][0])
                content_parts = [paras[i] for i in idxs_sorted]
                content = "\n\n".join(content_parts)
                start_pos = para_positions[idxs_sorted[0]][0]
                end_pos = para_positions[idxs_sorted[-1]][1]

                cluster_embs = [embeddings[i] for i in idxs_sorted]
                samples = choose_representative_samples([paras[i] for i in idxs_sorted], cluster_embs, top_k=3)
                rep_texts = [s[1] for s in samples]
                rep_meta = [{"paragraph_index": idxs_sorted[s[0]], "score": s[2]} for s in samples]

                section = Section(
                    id=f"section_{cid}",
                    content=content,
                    metadata={"topic": f"Cluster {cid}", "estimated_entities": 0, "position": cid},
                    start_pos=start_pos,
                    end_pos=end_pos,
                    representative_samples=rep_texts,
                    sample_metadata=rep_meta,
                )
                built_sections.append(section)
                section_metadata.append({
                    "id": section.id,
                    "topic": section.metadata.get("topic"),
                    "estimated_entities": 0,
                    "position": section.metadata.get("position"),
                    "length": len(section.content),
                })

            sections = sorted(built_sections, key=lambda s: s.start_pos)
            logger.info(f"Semantic splitter created {len(sections)} sections using embeddings and clustering")
            try:
                save_agent_output("splitter", {"sections": [s.dict() for s in sections], "section_metadata": section_metadata})
            except Exception:
                logger.exception("Failed to save splitter output after semantic split")

        except Exception as e2:
            logger.warning(f"Semantic splitting also failed: {e2}", exc_info=True)
            
            # Final fallback to LLM-based splitter
            try:
                async with client:
                    logger.info(f"Falling back to LLM splitter {config.splitter_model}")
                    response = await client.generate(
                        prompt=split_prompt,
                        system_prompt="You are an expert text analyzer. Always respond with valid JSON only. Ensure all text content is included in sections.",
                        temperature=0.3,
                        response_format={"type": "json_object"},
                    )
                    try:
                        split_result = json.loads(response)
                    except json.JSONDecodeError:
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
                            raise

                    section_list = split_result.get("sections", [])
                    if not section_list:
                        raise ValueError("No sections returned from LLM fallback")

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

                        start_pos = current_pos
                        end_pos = current_pos + len(content)

                        section = Section(
                            id=section_id,
                            content=content,
                            metadata={"topic": topic, "estimated_entities": estimated_entities, "position": position},
                            start_pos=start_pos,
                            end_pos=end_pos,
                        )
                        sections.append(section)
                        section_metadata.append({"id": section_id, "topic": topic, "estimated_entities": estimated_entities, "position": position, "length": len(content)})
                        current_pos = end_pos
            except Exception:
                logger.exception("LLM fallback failed — using character-based fallback")
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


def _quick_split_by_boundaries(corpus: str, target_sections: int) -> List[Section]:
    """
    Quick splitting method using paragraph and header boundaries.
    
    This method is designed for large textbooks and documents where:
    - Headers/chapters provide natural boundaries
    - Paragraph breaks indicate logical sections
    - No resource-intensive embeddings needed
    
    Args:
        corpus: Text to split
        target_sections: Target number of sections
        
    Returns:
        List of Section objects
    """
    logger.info(f"Using quick boundary-based splitting for {target_sections} sections")
    
    # First, identify all potential boundaries
    boundaries = []
    
    # 1. Find chapter/section headers (lines that are short and capitalized)
    lines = corpus.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
            
        # Check if this looks like a header
        # - Short length (typically < 100 chars)
        # - Contains capital letters or numbers
        # - Not ending with punctuation (usually)
        # - Followed by blank line or content
        if (len(stripped) < 100 and 
            (stripped.isupper() or any(c.isupper() for c in stripped) or any(c.isdigit() for c in stripped)) and
            not stripped.endswith(('.', '!', '?', ':')) and
            i < len(lines) - 1 and (not lines[i+1].strip() or len(lines[i+1].strip()) > 20)):
            
            # Find the actual position in the corpus
            pos = corpus.find(line)
            if pos != -1:
                boundaries.append((pos, 'header', line))
    
    # 2. Find paragraph boundaries (double newlines)
    para_positions = []
    search_pos = 0
    while True:
        pos = corpus.find('\n\n', search_pos)
        if pos == -1:
            break
        para_positions.append(pos)
        search_pos = pos + 2
    
    # 3. Combine boundaries and sort by position
    all_boundaries = []
    
    # Add header boundaries
    for pos, boundary_type, content in boundaries:
        all_boundaries.append((pos, boundary_type, content))
    
    # Add paragraph boundaries, but only if they're not too close to headers
    for pos in para_positions:
        # Check if this paragraph boundary is too close to a header
        too_close = False
        for header_pos, _, _ in boundaries:
            if abs(pos - header_pos) < 500:  # Within 500 characters
                too_close = True
                break
        
        if not too_close:
            all_boundaries.append((pos, 'paragraph', ''))
    
    # Sort all boundaries by position
    all_boundaries.sort(key=lambda x: x[0])
    
    # 4. Create sections based on boundaries
    sections: List[Section] = []
    current_pos = 0
    
    if not all_boundaries:
        # Fallback to character-based splitting if no boundaries found
        return _fallback_split(corpus, target_sections)
    
    # Calculate how many boundaries to use based on target sections
    # We want roughly target_sections sections, so we need target_sections-1 boundaries
    if len(all_boundaries) < target_sections - 1:
        # Not enough boundaries, use all of them
        selected_boundaries = all_boundaries
    else:
        # Select boundaries evenly spaced
        step = len(all_boundaries) // (target_sections - 1)
        selected_boundaries = [all_boundaries[i * step] for i in range(target_sections - 1)]
    
    # Create sections
    for i, (boundary_pos, boundary_type, content) in enumerate(selected_boundaries):
        # Find a good break point - don't break mid-sentence
        break_pos = boundary_pos
        
        # Look for a good sentence boundary near the break point
        # Search backwards for a period, question mark, or exclamation point
        for j in range(boundary_pos, max(0, boundary_pos - 200), -1):
            if corpus[j:j+1] in '.!?':
                # Check if followed by whitespace or newline
                if j + 1 < len(corpus) and corpus[j + 1] in ' \n':
                    break_pos = j + 1
                    break
        
        # Get content for this section
        section_content = corpus[current_pos:break_pos]
        
        if section_content.strip():  # Only add non-empty sections (but preserve whitespace)
            section = Section(
                id=f"section_{len(sections)}",
                content=section_content,
                metadata={
                    "topic": f"Section {len(sections)}",
                    "estimated_entities": 0,
                    "position": len(sections),
                    "boundary_type": boundary_type,
                    "boundary_content": content[:50] if content else ""
                },
                start_pos=current_pos,
                end_pos=break_pos,
            )
            sections.append(section)
        
        current_pos = break_pos
    
    # Add the final section with remaining content
    final_content = corpus[current_pos:]
    if final_content.strip():
        section = Section(
            id=f"section_{len(sections)}",
            content=final_content,
            metadata={
                "topic": f"Section {len(sections)}",
                "estimated_entities": 0,
                "position": len(sections),
                "boundary_type": "end",
                "boundary_content": ""
            },
            start_pos=current_pos,
            end_pos=len(corpus),
        )
        sections.append(section)
    
    # If we have too many or too few sections, adjust
    if len(sections) > target_sections * 1.5:  # Too many sections
        # Merge small sections
        sections = _merge_small_sections(sections, target_sections)
    elif len(sections) < target_sections * 0.5:  # Too few sections
        # Split large sections
        sections = _split_large_sections(sections, target_sections)
    
    logger.info(f"Quick splitter created {len(sections)} sections using {len(selected_boundaries)} boundaries")
    return sections


def _merge_small_sections(sections: List[Section], target_count: int) -> List[Section]:
    """Merge small sections to reduce total count."""
    if len(sections) <= target_count:
        return sections
    
    # Sort sections by length
    sections.sort(key=lambda s: len(s.content))
    
    merged_sections = []
    i = 0
    while i < len(sections):
        current_section = sections[i]
        
        # Try to merge with next section if current is small
        if len(current_section.content) < 2000 and i + 1 < len(sections):
            next_section = sections[i + 1]
            merged_content = current_section.content + "\n\n" + next_section.content
            
            merged_section = Section(
                id=current_section.id,
                content=merged_content,
                metadata={
                    "topic": f"{current_section.metadata['topic']} + {next_section.metadata['topic']}",
                    "estimated_entities": current_section.metadata['estimated_entities'] + next_section.metadata['estimated_entities'],
                    "position": current_section.metadata['position'],
                    "boundary_type": "merged",
                    "boundary_content": ""
                },
                start_pos=current_section.start_pos,
                end_pos=next_section.end_pos,
            )
            merged_sections.append(merged_section)
            i += 2  # Skip next section since we merged it
        else:
            merged_sections.append(current_section)
            i += 1
    
    return merged_sections


def _split_large_sections(sections: List[Section], target_count: int) -> List[Section]:
    """Split large sections to increase total count."""
    if len(sections) >= target_count:
        return sections
    
    split_sections = []
    for section in sections:
        if len(section.content) > 15000 and len(split_sections) < target_count - 1:
            # Split this section in half
            mid_point = len(section.content) // 2
            
            # Find a good break point around the middle
            break_pos = mid_point
            for i in range(mid_point, max(0, mid_point - 1000), -1):
                if section.content[i:i+2] == '\n\n':
                    break_pos = i
                    break
            
            # Create first half
            first_half = Section(
                id=f"{section.id}_part1",
                content=section.content[:break_pos].strip(),
                metadata={
                    "topic": f"{section.metadata['topic']} (Part 1)",
                    "estimated_entities": section.metadata['estimated_entities'] // 2,
                    "position": len(split_sections),
                    "boundary_type": "split",
                    "boundary_content": ""
                },
                start_pos=section.start_pos,
                end_pos=section.start_pos + break_pos,
            )
            split_sections.append(first_half)
            
            # Create second half
            second_half = Section(
                id=f"{section.id}_part2",
                content=section.content[break_pos:].strip(),
                metadata={
                    "topic": f"{section.metadata['topic']} (Part 2)",
                    "estimated_entities": section.metadata['estimated_entities'] - first_half.metadata['estimated_entities'],
                    "position": len(split_sections),
                    "boundary_type": "split",
                    "boundary_content": ""
                },
                start_pos=section.start_pos + break_pos,
                end_pos=section.end_pos,
            )
            split_sections.append(second_half)
        else:
            split_sections.append(section)
    
    return split_sections



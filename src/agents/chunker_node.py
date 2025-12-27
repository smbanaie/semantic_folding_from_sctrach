"""
Chunker Agent - Phase 3 of the knowledge graph pipeline.

Creates optimal context windows (chunks) from sections with overlap
to maintain context across boundaries.
"""

import logging
import re
from typing import List

import tiktoken

from src.config import get_config
from src.models.data_models import Chunk, GraphState, Section

logger = logging.getLogger(__name__)

# Try to get encoding, fallback to cl100k_base (GPT-4 tokenizer)
try:
    encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    encoding = None
    logger.warning("tiktoken encoding not available, using character-based estimation")


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    if encoding:
        return len(encoding.encode(text))
    # Fallback: estimate 1 token ≈ 4 characters
    return len(text) // 4


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    # Pattern to match sentence endings
    # Matches: . ! ? followed by space or end of string
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])(?=\n\n)'
    sentences = re.split(sentence_pattern, text)
    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]


async def chunker_agent(state: GraphState) -> GraphState:
    """
    Create chunks from sections with overlap.

    This agent:
    1. Takes sections from state
    2. Splits each section into chunks of ~1000 tokens
    3. Applies 15% overlap between consecutive chunks
    4. Creates Chunk objects with proper IDs

    Args:
        state: Current graph state with sections

    Returns:
        Updated state with chunks populated
    """
    config = get_config()
    sections: List[Section] = state.get("sections", [])

    if not sections:
        raise ValueError("No sections found in state")

    logger.info("Starting chunker agent")
    logger.info(f"Number of sections: {len(sections)}")

    chunks: List[Chunk] = []
    chunk_mapping: dict = {}  # chunk_id -> section_id

    target_chunk_size = config.chunk_size  # tokens
    overlap_ratio = config.chunk_overlap

    for section in sections:
        section_chunks = _chunk_section(
            section=section,
            target_size=target_chunk_size,
            overlap_ratio=overlap_ratio,
        )
        chunks.extend(section_chunks)

        # Update mapping
        for chunk in section_chunks:
            chunk_mapping[chunk.id] = chunk.section_id

    logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
    logger.info(f"Average chunk size: {sum(_estimate_tokens(c.content) for c in chunks) / len(chunks):.0f} tokens")

    # Update state
    state["chunks"] = chunks
    state["chunk_mapping"] = chunk_mapping

    return state


def _chunk_section(
    section: Section,
    target_size: int,
    overlap_ratio: float,
) -> List[Chunk]:
    """
    Chunk a section into smaller pieces with overlap.

    Args:
        section: Section to chunk
        target_size: Target chunk size in tokens
        overlap_ratio: Overlap ratio (0.0-0.5)

    Returns:
        List of Chunk objects
    """
    content = section.content
    section_id = section.id

    if not content:
        return []

    # Split into sentences
    sentences = _split_sentences(content)
    if not sentences:
        # Fallback: split by paragraphs
        paragraphs = content.split("\n\n")
        sentences = [p.strip() for p in paragraphs if p.strip()]

    if not sentences:
        # Last resort: single chunk
        token_count = _estimate_tokens(content)
        chunk = Chunk(
            id=f"{section_id}_chunk_0",
            content=content,
            section_id=section_id,
            chunk_index=0,
            overlap_with_next=0,
            token_count=token_count,
        )
        return [chunk]

    chunks: List[Chunk] = []
    current_chunk_sentences: List[str] = []
    current_tokens = 0
    chunk_index = 0

    # Calculate overlap size
    overlap_tokens = int(target_size * overlap_ratio)

    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)

        # If adding this sentence would exceed target, finalize current chunk
        if current_tokens + sentence_tokens > target_size and current_chunk_sentences:
            # Create chunk
            chunk_content = " ".join(current_chunk_sentences)
            chunk = Chunk(
                id=f"{section_id}_chunk_{chunk_index}",
                content=chunk_content,
                section_id=section_id,
                chunk_index=chunk_index,
                overlap_with_next=0,  # Will be set after next chunk is created
                token_count=_estimate_tokens(chunk_content),
            )
            chunks.append(chunk)
            chunk_index += 1

            # Start new chunk with overlap
            # Take last sentences that fit in overlap
            overlap_sentences: List[str] = []
            overlap_token_count = 0
            for s in reversed(current_chunk_sentences):
                s_tokens = _estimate_tokens(s)
                if overlap_token_count + s_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_token_count += s_tokens
                else:
                    break

            current_chunk_sentences = overlap_sentences + [sentence]
            current_tokens = overlap_token_count + sentence_tokens

            # Update previous chunk's overlap
            if chunks:
                chunks[-1].overlap_with_next = len(" ".join(overlap_sentences))
        else:
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens

    # Add final chunk
    if current_chunk_sentences:
        chunk_content = " ".join(current_chunk_sentences)
        chunk = Chunk(
            id=f"{section_id}_chunk_{chunk_index}",
            content=chunk_content,
            section_id=section_id,
            chunk_index=chunk_index,
            overlap_with_next=0,
            token_count=_estimate_tokens(chunk_content),
        )
        chunks.append(chunk)

    # Validate chunks
    validated_chunks: List[Chunk] = []
    for chunk in chunks:
        token_count = chunk.token_count or _estimate_tokens(chunk.content)
        # Min 200 tokens, max 2000 tokens
        if token_count < 200:
            logger.warning(
                f"Chunk {chunk.id} is too small ({token_count} tokens), "
                "considering merging with next chunk"
            )
            # Try to merge with next chunk if available
            if validated_chunks:
                last_chunk = validated_chunks[-1]
                merged_content = last_chunk.content + " " + chunk.content
                merged_tokens = _estimate_tokens(merged_content)
                if merged_tokens <= 2000:
                    last_chunk.content = merged_content
                    last_chunk.token_count = merged_tokens
                    last_chunk.overlap_with_next = chunk.overlap_with_next
                    continue
        elif token_count > 2000:
            logger.warning(
                f"Chunk {chunk.id} is too large ({token_count} tokens), "
                "may cause issues with LLM"
            )

        validated_chunks.append(chunk)

    logger.info(
        f"Section {section_id}: Created {len(validated_chunks)} chunks "
        f"(avg {sum(c.token_count or _estimate_tokens(c.content) for c in validated_chunks) / len(validated_chunks):.0f} tokens)"
    )

    return validated_chunks




"""
Pydantic data models for the knowledge graph builder.

Defines the data structures used throughout the pipeline:
- Section: Document sections created by Splitter
- Chunk: Text chunks created by Chunker
- Triple: Knowledge triples extracted by Extractor
- GraphState: LangGraph state schema
"""

from typing import Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


class Section(BaseModel):
    """
    Represents a logical section of the corpus.

    Created by the Splitter agent to divide the corpus into
    manageable, semantically coherent sections.
    """

    id: str = Field(..., description="Unique section identifier (e.g., 'section_0')")
    content: str = Field(..., description="Section text content")
    metadata: Dict = Field(
        default_factory=dict,
        description="Section metadata (topic, estimated_entities, position, etc.)",
    )
    start_pos: int = Field(
        default=0,
        description="Starting character position in original corpus",
    )
    end_pos: int = Field(
        default=0,
        description="Ending character position in original corpus",
    )
    representative_samples: Optional[List[str]] = Field(
        default_factory=list,
        description="Representative sample paragraphs chosen for this section (for Analyzer/Extractor context)",
    )
    sample_metadata: Optional[List[Dict]] = Field(
        default_factory=list,
        description="Optional metadata for each representative sample (e.g., paragraph_index, score)",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "id": "section_0",
                "content": "This is a section about artificial intelligence...",
                "metadata": {
                    "topic": "AI Overview",
                    "estimated_entities": 15,
                    "position": 0,
                },
                "start_pos": 0,
                "end_pos": 1500,
            }
        }


class Chunk(BaseModel):
    """
    Represents a text chunk for extraction.

    Created by the Chunker agent with optimal size for LLM processing.
    Includes overlap information to maintain context across boundaries.
    """

    id: str = Field(
        ...,
        description="Unique chunk identifier (e.g., 'section_0_chunk_0')",
    )
    content: str = Field(..., description="Chunk text content")
    section_id: str = Field(..., description="Parent section identifier")
    chunk_index: int = Field(
        default=0,
        description="Index of chunk within its section",
    )
    overlap_with_next: int = Field(
        default=0,
        description="Number of characters overlapping with next chunk",
    )
    token_count: Optional[int] = Field(
        default=None,
        description="Estimated token count for the chunk",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "id": "section_0_chunk_0",
                "content": "Artificial intelligence (AI) is transforming healthcare...",
                "section_id": "section_0",
                "chunk_index": 0,
                "overlap_with_next": 150,
                "token_count": 987,
            }
        }


class Triple(BaseModel):
    """
    Represents a knowledge triple (subject, predicate, object).

    Extracted by the Extractor agent and validated by the Reviewer agent.
    """

    subject: str = Field(..., description="Subject entity of the triple")
    predicate: str = Field(..., description="Relationship/predicate")
    object: str = Field(..., description="Object entity of the triple")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0)",
    )
    source_chunk_id: str = Field(
        ...,
        description="ID of the chunk from which this triple was extracted",
    )
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional metadata (entity types, context, etc.)",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "subject": "OpenAI",
                "predicate": "DEVELOPED",
                "object": "GPT-4",
                "confidence": 0.95,
                "source_chunk_id": "section_0_chunk_0",
                "metadata": {
                    "subject_type": "Organization",
                    "object_type": "Technology",
                },
            }
        }


class GraphState(TypedDict):
    """
    LangGraph state schema.

    This TypedDict defines the state structure passed between nodes
    in the LangGraph workflow. All agents read from and update this state.
    """

    # Input
    corpus: str
    corpus_metadata: Dict

    # Analyzer outputs
    extraction_strategy: str
    extraction_prompts: Dict[str, str]
    domain_context: str

    # Splitter outputs
    sections: List[Section]
    section_metadata: List[Dict]

    # Chunker outputs
    chunks: List[Chunk]
    chunk_mapping: Dict[str, str]  # chunk_id -> section_id

    # Extractor outputs
    raw_triples: List[Triple]
    extraction_stats: Dict

    # Reviewer outputs
    validated_triples: List[Triple]
    corrections_made: List[Dict]

    # Storage outputs
    graph_stats: Dict
    storage_status: str


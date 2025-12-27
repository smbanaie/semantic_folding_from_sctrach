"""
LangGraph workflow for knowledge graph builder.

Orchestrates the multi-agent pipeline:
analyzer → splitter → chunker → extractor → reviewer → storage
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from src.agents import (
    analyzer_agent,
    chunker_agent,
    extractor_agent,
    reviewer_agent,
    splitter_agent,
)
from src.config import get_config
from src.models.data_models import GraphState
from src.storage.memgraph_client import MemgraphClient

logger = logging.getLogger(__name__)


async def storage_node(state: GraphState) -> GraphState:
    """
    Storage node: Write validated triples to Memgraph.

    This node:
    1. Takes validated_triples from state
    2. Connects to Memgraph
    3. Creates nodes and relationships
    4. Updates graph statistics

    Args:
        state: Current graph state with validated_triples

    Returns:
        Updated state with graph_stats and storage_status
    """
    config = get_config()
    validated_triples = state.get("validated_triples", [])

    if not validated_triples:
        logger.warning("No validated triples to store")
        state["graph_stats"] = {"nodes": 0, "edges": 0}
        state["storage_status"] = "skipped_no_triples"
        return state

    logger.info(f"Storing {len(validated_triples)} triples to Memgraph")

    try:
        client = MemgraphClient(
            uri=config.memgraph_uri,
            user=config.memgraph_user,
            password=config.memgraph_password,
        )

        with client:
            # Bulk insert triples
            client.bulk_insert_triples(validated_triples)

            # Get graph statistics
            stats = client.get_stats()

            logger.info(f"Successfully stored triples:")
            logger.info(f"  - Nodes: {stats.get('nodes', 0)}")
            logger.info(f"  - Edges: {stats.get('edges', 0)}")

            state["graph_stats"] = stats
            state["storage_status"] = "success"

    except Exception as e:
        logger.error(f"Error storing to Memgraph: {e}", exc_info=True)
        state["graph_stats"] = {"nodes": 0, "edges": 0, "error": str(e)}
        state["storage_status"] = f"error: {str(e)}"

    return state


def should_retry_extraction(state: GraphState) -> Literal["reviewer", "chunker"]:
    """
    Conditional edge function: Determine if extraction should be retried.

    If extraction has >30% error rate, route back to chunker.
    Otherwise, proceed to reviewer.

    Args:
        state: Current graph state

    Returns:
        Next node name
    """
    extraction_stats = state.get("extraction_stats", {})
    total_chunks = extraction_stats.get("total_chunks", 0)
    chunks_failed = extraction_stats.get("chunks_failed", 0)

    if total_chunks > 0:
        error_rate = chunks_failed / total_chunks
        if error_rate > 0.3:  # 30% error rate
            logger.warning(
                f"High error rate ({error_rate:.1%}), routing back to chunker for retry"
            )
            return "chunker"

    logger.info("Extraction successful, proceeding to reviewer")
    return "reviewer"


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph workflow.

    The workflow follows this sequence:
    1. analyzer: Analyze corpus and generate extraction strategy
    2. splitter: Divide corpus into sections
    3. chunker: Create chunks with overlap
    4. extractor: Extract triples in parallel
    5. reviewer: Validate and normalize triples
    6. storage: Write to Memgraph

    Returns:
        Compiled LangGraph application
    """
    logger.info("Building LangGraph workflow")

    # Create StateGraph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("analyzer", analyzer_agent)
    workflow.add_node("splitter", splitter_agent)
    workflow.add_node("chunker", chunker_agent)
    workflow.add_node("extractor", extractor_agent)
    workflow.add_node("reviewer", reviewer_agent)
    workflow.add_node("storage", storage_node)

    # Set entry point
    workflow.set_entry_point("analyzer")

    # Add sequential edges
    workflow.add_edge("analyzer", "splitter")
    workflow.add_edge("splitter", "chunker")
    workflow.add_edge("chunker", "extractor")

    # Conditional edge: extractor → reviewer or chunker (retry)
    workflow.add_conditional_edges(
        "extractor",
        should_retry_extraction,
        {
            "reviewer": "reviewer",
            "chunker": "chunker",  # Retry chunking if too many errors
        },
    )

    # Continue to storage after reviewer
    workflow.add_edge("reviewer", "storage")
    workflow.add_edge("storage", END)

    # Compile the graph
    app = workflow.compile()

    logger.info("LangGraph workflow compiled successfully")
    logger.info("Workflow: analyzer → splitter → chunker → extractor → reviewer → storage")

    return app


def visualize_graph() -> str:
    """
    Generate a visualization of the workflow graph.

    Returns:
        Graph visualization in Mermaid format
    """
    mermaid = """
    graph TD
        START([Start]) --> analyzer[Analyzer Agent]
        analyzer --> splitter[Splitter Agent]
        splitter --> chunker[Chunker Agent]
        chunker --> extractor[Extractor Agent<br/>Parallel Processing]
        extractor -->|Success| reviewer[Reviewer Agent]
        extractor -->|High Error Rate| chunker
        reviewer --> storage[Storage Node<br/>Memgraph]
        storage --> END([End])
        
        style analyzer fill:#e1f5ff
        style splitter fill:#fff4e1
        style chunker fill:#fff4e1
        style extractor fill:#ffe1f5
        style reviewer fill:#e1f5ff
        style storage fill:#e1ffe1
    """
    return mermaid.strip()




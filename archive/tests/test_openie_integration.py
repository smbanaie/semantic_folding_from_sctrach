"""
Test script to validate OpenIE integration in the extractor agent.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.data_models import Chunk
from src.utils.openrouter_client import OpenRouterClient
from src.agents.extractor_node import extract_entities_openie, extract_relations_openie, extract_from_chunk_openie


async def test_openie_extraction():
    """Test OpenIE extraction with sample text."""

    # Sample text similar to HippoRAG2 examples
    sample_text = """
    Professor Thomas researches Alzheimer's disease at Stanford University.
    He uses advanced neural network models to study brain patterns.
    The research is funded by the National Institutes of Health.
    """

    # Initialize client (you'll need to set OPENROUTER_API_KEY)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        return

    client = OpenRouterClient(
        api_key=api_key,
        model="meta-llama/llama-3.1-8b-instruct",
        base_url="https://openrouter.ai/api/v1"
    )

    extraction_prompts = {
        "entity_extraction": "Extract named entities including persons, organizations, locations, and technical concepts.",
        "relationship_extraction": "Extract relationships between entities focusing on research, development, and organizational connections."
    }

    print("Testing OpenIE Entity Extraction...")
    print(f"Sample text: {sample_text.strip()}")

    try:
        # Test entity extraction
        async with client:
            entities = await extract_entities_openie(sample_text, client)
            print(f"\nExtracted entities: {entities}")

            # Test relation extraction
            triples_data = await extract_relations_openie(
                sample_text, entities, extraction_prompts, client
            )
            print(f"\nExtracted relations: {triples_data}")

            # Test full OpenIE extraction
            chunk = Chunk(
                id="test_chunk_1",
                content=sample_text,
                metadata={"source": "test"}
            )

            triples = await extract_from_chunk_openie(chunk, extraction_prompts, client)
            print(f"\nFinal triples ({len(triples)}):")
            for triple in triples:
                print(f"  ({triple.subject}, {triple.predicate}, {triple.object})")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_openie_extraction())
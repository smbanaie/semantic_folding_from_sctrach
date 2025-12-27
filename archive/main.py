#!/usr/bin/env python3
"""
KG-Builder: Knowledge Graph Generation using kg-gen library with OpenRouter
"""

import os
import sys
import argparse
from datetime import datetime
import re
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import yaml

# Configure loguru logging
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)
logger.add(
    "logs/kg-builder.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="7 days"
)

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

from kg_gen import KGGen
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def load_models():
    """Load available models from YAML file"""
    try:
        with open('models.yaml', 'r', encoding='utf-8') as f:
            models_data = yaml.safe_load(f)
        return models_data
    except FileNotFoundError:
        logger.error("models.yaml file not found")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing models.yaml: {e}")
        return None


def load_environment():
    """Load environment variables from .env file"""
    logger.info("Loading environment variables...")
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment variables.")
        logger.error("Please create a .env file with your OpenRouter API key.")
        logger.error("You can copy .env.sample to .env and fill in your key.")
        sys.exit(1)
    
    logger.info("Environment variables loaded successfully")
    return api_key


def setup_kg_gen(api_key: str, model: str = "openrouter/google/gemma-2-9b-it:free"):
    """Initialize KGGen with OpenRouter configuration"""
    # Normalize model to include provider prefix expected by LiteLLM (OpenRouter)
    normalized_model = model if model.startswith("openrouter/") else f"openrouter/{model}"
    if normalized_model != model:
        logger.debug(f"Normalized model provider: '{model}' -> '{normalized_model}'")
    logger.info(f"Setting up KGGen with model: {normalized_model}")
    
    # Set up environment variables for LiteLLM (used by kg-gen internally)
    os.environ["OPENROUTER_API_KEY"] = api_key
    logger.debug("Set OPENROUTER_API_KEY environment variable")
    
    try:
        logger.info("Initializing KGGen...")
        # Initialize KGGen with OpenRouter model
        kg = KGGen(
            model=normalized_model,
            temperature=0.1,  # Low temperature for consistent output
            api_key=api_key
        )
        logger.success("KGGen initialized successfully")
        return kg
    except Exception as e:
        logger.error(f"Failed to initialize KGGen: {e}")
        raise


def process_text_file(kg: KGGen, input_file: str, output_dir: str, chunk_size: int = 5000, cluster: bool = True):
    """Process a text file and generate knowledge graph"""
    logger.info(f"Processing file: {input_file}")
    
    try:
        # Read the input file
        logger.debug(f"Reading input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        logger.info(f"Text length: {len(text)} characters")
        
        # Generate knowledge graph
        logger.info("Generating knowledge graph...")
        logger.debug(f"Using chunk_size: {chunk_size}, cluster: {cluster}")
        
        graph = kg.generate(
            input_data=text,
            chunk_size=chunk_size,
            cluster=cluster
        )
        
        # Log results
        logger.success("Knowledge graph generated successfully!")
        logger.info(f"Entities: {len(graph.entities)}")
        logger.info(f"Relations: {len(graph.relations)}")
        logger.info(f"Edges: {len(graph.edges)}")
        
        if hasattr(graph, 'entity_clusters') and graph.entity_clusters:
            logger.info(f"Entity clusters: {len(graph.entity_clusters)}")
        
        if hasattr(graph, 'edge_clusters') and graph.edge_clusters:
            logger.info(f"Edge clusters: {len(graph.edge_clusters)}")
        
        # Save results
        logger.info(f"Saving results to: {output_dir}")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save graph data
        logger.debug("Saving graph.pkl...")
        import pickle
        with open(output_path / "graph.pkl", "wb") as f:
            pickle.dump(graph, f)
        
        # Save entities and relations as text files
        logger.debug("Saving entities.txt...")
        with open(output_path / "entities.txt", "w", encoding="utf-8") as f:
            for entity in sorted(graph.entities):
                f.write(f"{entity}\n")
        
        logger.debug("Saving relations.txt...")
        with open(output_path / "relations.txt", "w", encoding="utf-8") as f:
            for relation in sorted(graph.relations):
                f.write(f"{relation[0]} -> {relation[1]} -> {relation[2]}\n")
        
        # Generate visualization
        html_path = output_path / "graph_visualization.html"
        logger.info(f"Generating visualization: {html_path}")
        kg.visualize(graph, str(html_path), open_in_browser=False)
        
        logger.success(f"Results saved to: {output_path}")
        logger.success(f"Visualization: {html_path}")
        
        return graph
        
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {e}")
        raise


def main():
    logger.info("Starting KG-Builder application")
    
    parser = argparse.ArgumentParser(description="KG-Builder: Knowledge Graph Generation using kg-gen")
    parser.add_argument('--input', '-i', help='Input text file path')
    parser.add_argument('--output', '-o', default='./output', help='Base output directory (default: ./output)')
    parser.add_argument('--label', help='Optional label for this run; if omitted, a timestamped label will be generated')
    parser.add_argument('--model', '-m', default='openai/gpt-oss-20b:free', 
                       help='OpenRouter model to use (default: openai/gpt-oss-20b:free)')
    parser.add_argument('--chunk-size', '-c', type=int, default=5000, 
                       help='Chunk size for large texts (default: 5000)')
    parser.add_argument('--no-cluster', action='store_true', 
                       help='Disable entity and relation clustering')
    parser.add_argument('--list-models', action='store_true', 
                       help='List available OpenRouter models and exit')
    
    args = parser.parse_args()
    logger.debug(f"Command line arguments: {args}")
    
    # Validate arguments
    if not args.list_models and not args.input:
        logger.error("Input file is required when not listing models")
        parser.error("Input file is required when not listing models. Use --input or --list-models")
    
    # Load environment
    api_key = load_environment()
    
    if args.list_models:
        logger.info("Loading available OpenRouter models from YAML")
        models_data = load_models()
        
        if not models_data:
            print("❌ Error: Could not load models.yaml file")
            return
        
        console = Console()
        console.print(Panel.fit(Text("Available OpenRouter Free Models", style="bold magenta")))

        def render_group(title: str, models: list):
            if not models:
                return
            table = Table(title=title, show_lines=True)
            table.add_column("Name", style="bold cyan")
            table.add_column("ID", style="green")
            table.add_column("Provider", style="yellow")
            table.add_column("Context", style="blue")
            table.add_column("Best For", style="white")
            table.add_column("Reasoning", style="white")
            table.add_column("Tool Use", style="white")
            for m in models:
                table.add_row(
                    m.get("name", "-"),
                    m.get("id", "-"),
                    m.get("provider", "-"),
                    m.get("context", "-"),
                    m.get("best_for", "-"),
                    "✅" if m.get("reasoning") else "❌",
                    "✅" if m.get("tool_use") else "❌",
                )
            console.print(table)

        render_group("RECOMMENDED for Knowledge Graph Generation", models_data.get('models', {}).get('recommended', []))
        render_group("ALTERNATIVES", models_data.get('models', {}).get('alternatives', []))
        render_group("SPECIALIZED", models_data.get('models', {}).get('specialized', []))

        console.print(Panel.fit(Text("Usage Examples", style="bold green")))
        console.print("python main.py --input input/sample-text.txt --model deepseek/deepseek-v3.1:free")
        console.print("python main.py --input input/sample-text.txt --model z-ai/glm-4.5-air:free")
        return
    
    # Initialize KGGen
    try:
        logger.info("Initializing KGGen...")
        kg = setup_kg_gen(api_key, args.model)
        logger.success(f"Initialized KGGen with model: {args.model}")
    except Exception as e:
        logger.error(f"Error initializing KGGen: {e}")
        if "NotFoundError" in str(e) or "404" in str(e):
            logger.error("Model not found. Try using --list-models to see available models.")
            logger.error("Recommended: Use 'openrouter/google/gemma-2-9b-it:free'")
        sys.exit(1)
    
    # Process the input file
    try:
        logger.info("Starting knowledge graph generation...")

        # Build a unique run directory under the base output directory
        if args.label:
            run_label = args.label
        else:
            # Use ISO-like timestamp + sanitized model id, e.g., 2025-10-20T20-31
            ts = datetime.now().strftime('%Y-%m-%dT%H-%M')
            sanitized_model = re.sub(r'[^A-Za-z0-9_.-]+', '-', args.model)
            run_label = f"{ts}_{sanitized_model}"
        run_output_dir = str(Path(args.output) / run_label)
        logger.info(f"Run output directory (label='{run_label}'): {run_output_dir}")
        graph = process_text_file(
            kg, 
            args.input, 
            run_output_dir,
            chunk_size=args.chunk_size,
            cluster=not args.no_cluster
        )
        
        logger.success("Knowledge Graph generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

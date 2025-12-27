#!/usr/bin/env python3
"""
Knowledge Graph Builder - Main Entry Point

A LangGraph-based multi-agent system for building knowledge graphs from text corpora.
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn
import sys
import os

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

from src.config import get_config
from src.graph import build_graph, visualize_graph
from src.models.data_models import GraphState
from src.storage.memgraph_client import MemgraphClient

console = Console()


async def run_pipeline(
    corpus_text: str,
    corpus_files: list[Path],
    clear_graph: bool = False,
    output_path: str | None = None,
) -> GraphState:
    """
    Run the complete knowledge graph pipeline.

    Args:
        corpus_text: Combined corpus text
        corpus_files: List of corpus file paths
        clear_graph: Whether to clear Memgraph before running
        output_path: Custom output path for triples JSON

    Returns:
        Final graph state
    """
    # Clear graph if requested
    if clear_graph:
        console.print("[yellow]Clearing Memgraph...[/yellow]")
        config = get_config()
        try:
            with MemgraphClient(
                uri=config.memgraph_uri,
                user=config.memgraph_user,
                password=config.memgraph_password,
            ) as client:
                client.clear_graph()
            console.print("[green][OK] Graph cleared[/green]")
        except Exception as e:
            console.print(f"[red]Warning: Could not clear graph: {e}[/red]")

    # Build workflow
    console.print("[cyan]Building workflow...[/cyan]")
    app = build_graph()

    # Create initial state
    initial_state: GraphState = {
        "corpus": corpus_text,
        "corpus_metadata": {
            "files": [str(f) for f in corpus_files],
            "total_length": len(corpus_text),
            "word_count": len(corpus_text.split()),
        },
        "extraction_strategy": "",
        "extraction_prompts": {},
        "domain_context": "",
        "sections": [],
        "section_metadata": [],
        "chunks": [],
        "chunk_mapping": {},
        "raw_triples": [],
        "extraction_stats": {},
        "validated_triples": [],
        "corrections_made": [],
        "graph_stats": {},
        "storage_status": "",
    }

    # Run pipeline
    console.print("[bold green]Starting pipeline...[/bold green]")
    start_time = datetime.now()

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            disable=False,
        ) as progress:
            task = progress.add_task("Processing pipeline...", total=None)

            # Invoke the graph
            final_state = await app.ainvoke(initial_state)

            progress.update(task, completed=True)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        console.print(f"[green][OK] Pipeline completed in {duration:.1f} seconds[/green]")

        # Save results
        if output_path:
            output_file = Path(output_path)
        else:
            config = get_config()
            output_file = config.get_output_path("triples.json")

        # Also save timestamped copy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_file = output_file.parent / f"triples_{timestamp}.json"

        validated_triples = final_state.get("validated_triples", [])
        if validated_triples:
            # Convert triples to JSON-serializable format
            triples_data = [
                {
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "confidence": t.confidence,
                    "source_chunk_id": t.source_chunk_id,
                    "metadata": t.metadata,
                }
                for t in validated_triples
            ]

            # Save to both files
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(triples_data, f, indent=2, ensure_ascii=False)

            with open(timestamped_file, "w", encoding="utf-8") as f:
                json.dump(triples_data, f, indent=2, ensure_ascii=False)

            console.print(f"[green][OK] Saved {len(validated_triples)} triples to:[/green]")
            console.print(f"  - {output_file}")
            console.print(f"  - {timestamped_file}")

        return final_state

    except Exception as e:
        console.print(f"[red][ERROR] Pipeline failed: {e}[/red]")
        raise


def display_statistics(state: GraphState, duration: float):
    """
    Display pipeline statistics using Rich.

    Args:
        state: Final graph state
        duration: Pipeline duration in seconds
    """
    # Create statistics table
    table = Table(title="Pipeline Statistics", show_header=True, header_style="bold magenta")

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    # Corpus info
    corpus_metadata = state.get("corpus_metadata", {})
    table.add_row("Corpus Files", str(len(corpus_metadata.get("files", []))))
    table.add_row("Corpus Length", f"{corpus_metadata.get('total_length', 0):,} characters")
    table.add_row("Word Count", f"{corpus_metadata.get('word_count', 0):,} words")

    # Sections
    sections = state.get("sections", [])
    table.add_row("Sections Created", str(len(sections)))

    # Chunks
    chunks = state.get("chunks", [])
    table.add_row("Chunks Created", str(len(chunks)))

    # Extraction
    extraction_stats = state.get("extraction_stats", {})
    raw_triples = state.get("raw_triples", [])
    table.add_row("Raw Triples Extracted", str(len(raw_triples)))
    table.add_row(
        "Chunks Processed",
        f"{extraction_stats.get('chunks_processed', 0)}/{extraction_stats.get('total_chunks', 0)}",
    )
    table.add_row("Chunks Failed", str(extraction_stats.get("chunks_failed", 0)))

    # Validation
    validated_triples = state.get("validated_triples", [])
    corrections = state.get("corrections_made", [])
    table.add_row("Validated Triples", str(len(validated_triples)))
    table.add_row("Corrections Made", str(len(corrections)))

    # Graph stats
    graph_stats = state.get("graph_stats", {})
    table.add_row("Graph Nodes", str(graph_stats.get("nodes", 0)))
    table.add_row("Graph Edges", str(graph_stats.get("edges", 0)))

    # Storage
    storage_status = state.get("storage_status", "unknown")
    status_style = "green" if storage_status == "success" else "yellow"
    table.add_row("Storage Status", f"[{status_style}]{storage_status}[/{status_style}]")

    # Timing
    table.add_row("Duration", f"{duration:.1f} seconds")

    console.print()
    console.print(table)


def main():
    """Main entry point for the knowledge graph builder."""
    parser = argparse.ArgumentParser(
        description="Build knowledge graphs from text corpora using multi-agent pipeline"
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        help="Override CORPUS_DIRECTORY from config",
    )
    parser.add_argument(
        "--corpus-files",
        type=str,
        help="Specific files to process (comma-separated)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Custom output path for triples JSON",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show graph visualization",
    )
    parser.add_argument(
        "--clear-graph",
        action="store_true",
        help="Clear Memgraph before running",
    )

    args = parser.parse_args()

    # Show visualization if requested
    if args.visualize:
        console.print(Panel(visualize_graph(), title="Workflow Visualization", border_style="blue"))
        return

    # Display header
    console.print(
        Panel.fit(
            "[bold blue]Knowledge Graph Builder[/bold blue]\n"
            "[dim]Multi-agent pipeline for knowledge extraction[/dim]",
            border_style="blue",
        )
    )

    # Load configuration
    config = get_config()

    # Override corpus directory if provided
    if args.corpus_dir:
        config.corpus_directory = args.corpus_dir

    # Override corpus files if provided
    if args.corpus_files:
        config.corpus_files = [f.strip() for f in args.corpus_files.split(",")]

    # Load corpus
    try:
        console.print("[cyan]Loading corpus...[/cyan]")
        corpus_text = config.load_corpus()
        corpus_files = config.get_corpus_files()

        console.print(f"[green][OK] Loaded {len(corpus_files)} file(s)[/green]")
        console.print(f"  Total length: {len(corpus_text):,} characters")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print(
            "\n[yellow]Tip:[/yellow] Add .txt files to data/corpus/ or configure CORPUS_FILES in .env"
        )
        return
    except Exception as e:
        console.print(f"[red]Error loading corpus: {e}[/red]")
        return

    # Run pipeline
    start_time = datetime.now()
    try:
        final_state = asyncio.run(
            run_pipeline(
                corpus_text=corpus_text,
                corpus_files=corpus_files,
                clear_graph=args.clear_graph,
                output_path=args.output,
            )
        )

        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Display statistics
        display_statistics(final_state, duration)

        console.print("\n[bold green][OK] Pipeline completed successfully![/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red][ERROR] Pipeline failed: {e}[/red]")
        raise


if __name__ == "__main__":
    main()

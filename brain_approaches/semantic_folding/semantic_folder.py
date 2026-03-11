#!/usr/bin/env python3
"""
Semantic Folding Pipeline - Interactive TUI

Command-line Text User Interface for running the Semantic Folding pipeline
with configuration management, phase selection, and error checking.
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import loguru
from loguru import logger

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    logger.warning("PyYAML not available. Install with: uv add pyyaml")
    YAML_AVAILABLE = False

try:
    import questionary
    QUESTIONARY_AVAILABLE = True
except ImportError:
    logger.warning("questionary not available. Install with: uv add questionary")
    QUESTIONARY_AVAILABLE = False


class SemanticFoldingTUI:
    """Text User Interface for Semantic Folding Pipeline"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/semantic_folding.yml"
        self.config = self.load_config()
        self.output_base = Path(self.config.get('output_base', 'outputs'))
        self.resume_file = Path.home() / ".semantic_folding_resume.json"

        # Pipeline phases
        self.phases = {
            1: {"name": "Corpus Loading & Preprocessing", "script": "scratchpad.py", "completed": False},
            2: {"name": "Phrase Extraction", "script": "phrase_extractor.py", "completed": False},
            3: {"name": "Term-Context Matrix", "script": "term_context.py", "completed": False},
            4: {"name": "Semantic Space Construction", "script": "semantic_space.py", "completed": False},
            5: {"name": "Fingerprints Generation", "script": "phrase_fingerprints.py", "completed": False},
            6: {"name": "LanceDB Integration", "script": "lance_storage.py", "completed": False},
        }

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path(self.config_file)

        # Default configuration
        default_config = {
            'corpus_path': 'data/HippoRAG2/dataset/musique_corpus.json',
            'queries_path': 'data/HippoRAG2/dataset/musique.json',
            'output_base': 'outputs',
            'grid_size': 16,
            'log_level': 'INFO',
            'debug': False,
            'max_phrases': None,
            'max_docs': None,
        }

        if config_path.exists() and YAML_AVAILABLE:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f) or {}
                default_config.update(yaml_config)
                logger.success(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        else:
            logger.info(f"Config file {config_path} not found, using defaults")

        return default_config

    def save_config(self) -> None:
        """Save current configuration to YAML file"""
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not available, cannot save config")
            return

        config_path = Path(self.config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            logger.success(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def save_resume_state(self, last_output_dir: str, last_phase: int) -> None:
        """Save resume state to file"""
        resume_data = {
            'last_output_dir': last_output_dir,
            'last_phase': last_phase,
            'timestamp': str(datetime.now())
        }

        try:
            with open(self.resume_file, 'w', encoding='utf-8') as f:
                json.dump(resume_data, f, indent=2)
            logger.info(f"Saved resume state to {self.resume_file}")
        except Exception as e:
            logger.warning(f"Failed to save resume state: {e}")

    def load_resume_state(self) -> Optional[Dict[str, Any]]:
        """Load resume state from file"""
        if not self.resume_file.exists():
            return None

        try:
            with open(self.resume_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load resume state: {e}")
            return None

    def clear_resume_state(self) -> None:
        """Clear resume state file"""
        if self.resume_file.exists():
            try:
                self.resume_file.unlink()
                logger.info("Cleared resume state")
            except Exception as e:
                logger.warning(f"Failed to clear resume state: {e}")

    def show_progress_indicator(self, phase_name: str, duration_estimate: int = 30) -> None:
        """Show a progress indicator for long-running phases"""
        progress_chars = ["|", "/", "-", "\\"]
        start_time = time.time()

        def progress_loop():
            i = 0
            while getattr(threading.current_thread(), "do_run", True):
                elapsed = time.time() - start_time
                progress_char = progress_chars[i % len(progress_chars)]
                logger.info(f"\r{progress_char} {phase_name}... ({elapsed:.1f}s elapsed)", end="", flush=True)
                time.sleep(0.5)
                i += 1

        # Start progress thread
        progress_thread = threading.Thread(target=progress_loop, daemon=True)
        progress_thread.do_run = True
        progress_thread.start()

        return progress_thread

    def stop_progress_indicator(self, progress_thread: threading.Thread) -> None:
        """Stop the progress indicator"""
        if progress_thread and progress_thread.is_alive():
            progress_thread.do_run = False
            progress_thread.join(timeout=1.0)
            logger.info("\r" + " " * 50 + "\r", end="", flush=True)  # Clear the line

    def show_phase_completion_stats(self, phase_num: int, output_dir: str) -> None:
        """Show statistics about what was created in the phase"""
        output_path = Path(output_dir)

        try:
            if phase_num == 1:
                corpus_file = output_path / "corpus.txt"
                if corpus_file.exists():
                    with open(corpus_file, 'r', encoding='utf-8') as f:
                        lines = sum(1 for _ in f)
                    logger.success(f"   Documents processed: {lines}")

            elif phase_num == 2:
                phrases_file = output_path / "phrases.txt"
                if phrases_file.exists():
                    with open(phrases_file, 'r', encoding='utf-8') as f:
                        lines = sum(1 for _ in f)
                    logger.success(f"   Phrases extracted: {lines}")

            elif phase_num == 3:
                matrix_file = output_path / "term_context_matrix.npz"
                if matrix_file.exists():
                    # Try to get matrix stats from the json file
                    stats_file = output_path / "term_context_matrix.json"
                    if stats_file.exists():
                        with open(stats_file, 'r', encoding='utf-8') as f:
                            stats = json.load(f)
                        logger.success(f"   Matrix created: {stats.get('num_contexts', '?')} × {stats.get('num_phrases', '?')}")
                        logger.success(f"   Sparsity: {stats.get('density', 0):.4f} ({stats.get('entries', 0)} entries)")
                    else:
                        logger.success(f"   Matrix file created: {matrix_file.stat().st_size / (1024*1024):.1f} MB")

            elif phase_num == 4:
                coords_file = output_path / "context_coordinates.csv"
                if coords_file.exists():
                    with open(coords_file, 'r', encoding='utf-8') as f:
                        lines = sum(1 for _ in f) - 1  # Subtract header
                    logger.success(f"   Semantic space: {lines} contexts mapped to {self.config['grid_size']}×{self.config['grid_size']} grid")

            elif phase_num == 5:
                fp_dir = output_path / "fingerprints"
                doc_fp_dir = output_path / "doc_fingerprints"

                fp_count = len(list(fp_dir.glob("*.txt"))) if fp_dir.exists() else 0
                doc_fp_count = len(list(doc_fp_dir.glob("*_fingerlogger.info.txt"))) if doc_fp_dir.exists() else 0

                logger.success(f"   Phrase fingerprints: {fp_count}")
                logger.success(f"   Document fingerprints: {doc_fp_count}")

            elif phase_num == 6:
                lance_dir = output_path / "lance_db"
                if lance_dir.exists():
                    # Count files in lance directory
                    total_files = sum(1 for _ in lance_dir.rglob("*") if _.is_file())
                    logger.success(f"   LanceDB created: {total_files} database files")

        except Exception as e:
            # Don't fail if we can't read stats, just skip
            logger.debug(f"Could not read completion stats: {e}")
            pass

    def show_final_pipeline_stats(self) -> None:
        """Show final statistics for the completed pipeline"""
        last_run = self.check_last_run_status()
        if not last_run:
            return

        try:
            stats = {}

            # Corpus stats
            corpus_file = last_run / "corpus.txt"
            if corpus_file.exists():
                with open(corpus_file, 'r', encoding='utf-8') as f:
                    stats['documents'] = sum(1 for _ in f)

            # Phrase stats
            phrases_file = last_run / "phrases.txt"
            if phrases_file.exists():
                with open(phrases_file, 'r', encoding='utf-8') as f:
                    stats['phrases'] = sum(1 for _ in f)

            # Matrix stats
            matrix_stats_file = last_run / "term_context_matrix.json"
            if matrix_stats_file.exists():
                with open(matrix_stats_file, 'r', encoding='utf-8') as f:
                    matrix_data = json.load(f)
                    stats['matrix_entries'] = matrix_data.get('entries', 0)
                    stats['matrix_density'] = matrix_data.get('density', 0)

            # Fingerlogger.info stats
            fp_dir = last_run / "fingerprints"
            doc_fp_dir = last_run / "doc_fingerprints"

            stats['phrase_fingerprints'] = len(list(fp_dir.glob("*.txt"))) if fp_dir.exists() else 0
            stats['doc_fingerprints'] = len(list(doc_fp_dir.glob("*_fingerlogger.info.txt"))) if doc_fp_dir.exists() else 0

            # Output directory size
            total_size = sum(f.stat().st_size for f in last_run.rglob('*') if f.is_file())
            stats['total_size_mb'] = total_size / (1024 * 1024)

            # Display stats
            logger.success(f"   Documents processed: {stats.get('documents', '?')}")
            logger.success(f"   Phrases extracted: {stats.get('phrases', '?')}")
            logger.success(f"   Matrix entries: {stats.get('matrix_entries', '?'):,}")
            logger.success(f"   Matrix density: {stats.get('matrix_density', 0):.4f}")
            logger.success(f"   Phrase fingerprints: {stats.get('phrase_fingerprints', '?')}")
            logger.success(f"   Document fingerprints: {stats.get('doc_fingerprints', '?')}")
            logger.success(f"   Total output size: {stats.get('total_size_mb', 0):.1f} MB")
            logger.success(f"   Output directory: {last_run.name}")

        except Exception as e:
            logger.debug(f"Could not show final stats: {e}")
            logger.info("   📊 Pipeline completed (detailed stats unavailable)")

    def check_last_run_status(self) -> Optional[Path]:
        """Check for the most recent output directory and its status"""
        if not self.output_base.exists():
            return None

        # Find most recent output directory
        output_dirs = [d for d in self.output_base.iterdir() if d.is_dir() and d.name.startswith('musique_')]
        if not output_dirs:
            return None

        latest_dir = max(output_dirs, key=lambda x: x.stat().st_mtime)
        logger.success(f"Found latest run: {latest_dir.name}")
        return latest_dir

    def check_phase_completion(self, output_dir: Path) -> None:
        """Check which phases have been completed in the given output directory"""
        # Reset completion status
        for phase in self.phases.values():
            phase["completed"] = False

        # Check for key output files
        checks = {
            1: output_dir / "corpus.txt",
            2: output_dir / "phrases.txt",
            3: output_dir / "term_context_matrix.npz",
            4: output_dir / "context_coordinates.csv",
            5: [output_dir / "fingerprints", output_dir / "doc_fingerprints"],
            6: output_dir / "lance_db",
        }

        for phase_num, check_files in checks.items():
            if isinstance(check_files, list):
                # Check if any of the files/directories exist
                self.phases[phase_num]["completed"] = any(f.exists() for f in check_files)
            else:
                self.phases[phase_num]["completed"] = check_files.exists()

    def check_log_errors(self, output_dir: Path) -> List[str]:
        """Check log files for errors from the last run"""
        errors = []
        logs_dir = output_dir / "logs"

        if not logs_dir.exists():
            return errors

        # Check main pipeline log
        pipeline_log = logs_dir / f"pipeline_{output_dir.name.split('_', 1)[1]}.log"
        if pipeline_log.exists():
            try:
                with open(pipeline_log, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "ERROR" in content or "FAILED" in content:
                        errors.append(f"Pipeline log contains errors: {pipeline_log}")
            except Exception as e:
                errors.append(f"Could not read pipeline log: {e}")

        # Check error log
        error_log = logs_dir / f"errors_{output_dir.name.split('_', 1)[1]}.log"
        if error_log.exists() and error_log.stat().st_size > 0:
            errors.append(f"Error log contains entries: {error_log}")

        return errors

    def show_status(self) -> None:
        """Display current pipeline status"""
        logger.success("\n" + "="*60)
        logger.success("SEMANTIC FOLDING PIPELINE STATUS")
        logger.success("="*60)

        # Configuration
        logger.success("\nConfiguration:")
        logger.success(f"   Corpus: {self.config['corpus_path']}")
        logger.success(f"   Output: {self.output_base}")
        logger.success(f"   Grid Size: {self.config['grid_size']}x{self.config['grid_size']}")
        logger.success(f"   Log Level: {self.config['log_level']}")

        # Check last run
        last_run = self.check_last_run_status()
        if last_run:
            logger.info(f"\nLast Run: {last_run.name}")

            # Check for errors
            errors = self.check_log_errors(last_run)
            if errors:
                logger.info("ERRORS FOUND:")
                for error in errors:
                    logger.info(f"   WARNING: {error}")
            else:
                logger.info("No errors detected in logs")

            # Check phase completion
            self.check_phase_completion(last_run)
            logger.info("\nPhase Completion:")
            for phase_num, phase_info in self.phases.items():
                status = "COMPLETED" if phase_info["completed"] else "PENDING"
                logger.info(f"   {status}: Phase {phase_num} - {phase_info['name']}")
        else:
            logger.info("\nNo previous runs found")

        

    def configure_pipeline(self) -> None:
        """Interactive configuration"""
        if not QUESTIONARY_AVAILABLE:
            logger.info("ERROR: questionary not available for interactive configuration")
            return

        logger.info("\n" + "="*50)
        logger.info("PIPELINE CONFIGURATION")
        logger.info("="*50)

        # Corpus path
        corpus_path = questionary.text(
            "Corpus path:",
            default=self.config['corpus_path']
        ).ask()

        # Grid size
        grid_size = questionary.select(
            "Grid size:",
            choices=["8", "16", "32"],
            default=str(self.config['grid_size'])
        ).ask()

        # Log level
        log_level = questionary.select(
            "Log level:",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default=self.config['log_level']
        ).ask()

        # Update config
        self.config.update({
            'corpus_path': corpus_path,
            'grid_size': int(grid_size),
            'log_level': log_level,
        })

        # Save config
        if questionary.confirm("Save configuration to file?").ask():
            self.save_config()

        logger.info("✅ Configuration updated!")

    def run_pipeline_menu(self) -> None:
        """Main pipeline execution menu"""
        if not QUESTIONARY_AVAILABLE:
            logger.info("ERROR: questionary not available for interactive menu")
            return

        while True:
            logger.info("\n" + "="*50)
            logger.info("SEMANTIC FOLDING PIPELINE")
            logger.info("="*50)

            self.show_status()

            choices = [
                "Run All Phases",
                "Run Specific Phase",
                "View Output Files",
                "Configure Pipeline",
                "Clean Old Outputs",
                "Exit"
            ]

            choice = questionary.select(
                "Select action:",
                choices=choices
            ).ask()

            if choice == "Exit":
                break
            elif choice == "Run All Phases":
                self.run_all_phases()
            elif choice == "Run Specific Phase":
                self.run_specific_phase()
            elif choice == "View Output Files":
                self.view_output_files()
            elif choice == "Configure Pipeline":
                self.configure_pipeline()
            elif choice == "Clean Old Outputs":
                self.clean_outputs()

    def run_all_phases(self) -> None:
        """Run all pipeline phases"""
        logger.info("\nSTARTING COMPLETE SEMANTIC FOLDING PIPELINE")
        logger.info("=" * 60)

        # Check if we should clean old outputs
        last_run = self.check_last_run_status()
        if last_run and questionary.confirm("Remove previous output directory first?").ask():
            logger.info(f"Cleaning up previous run: {last_run.name}")
            shutil.rmtree(last_run)
            logger.info(f"Removed old output directory: {last_run}")

        # Show pipeline overview
        logger.info("Pipeline Overview:")
        logger.info(f"   Output Base: {self.output_base}")
        logger.info(f"   Corpus: {self.config['corpus_path']}")
        logger.info(f"   Grid Size: {self.config['grid_size']}×{self.config['grid_size']}")
        logger.info(f"   Total Phases: 6")
        logger.info(f"   Estimated Time: 5-15 minutes (depending on corpus size)")
        

        # Run all phases sequentially with progress tracking
        logger.info("Executing complete pipeline...")
        

        # Create a timestamped output directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{self.output_base}/pipeline_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {output_dir}")
        

        # Run all phases
        phases = [1, 2, 3, 4, 5, 6]
        for phase_num in phases:
            try:
                self.run_specific_phase_non_interactive(phase_num, str(output_dir))
            except SystemExit as e:
                logger.info(f"Pipeline failed at Phase {phase_num}")
                return

        logger.success("\nPipeline completed successfully!")
        
        logger.success("Final Results:")

        # Show final statistics
        self.show_final_pipeline_stats()

    def run_phase_1(self, output_dir: str) -> str:
        """Run Phase 1: Corpus loading and preprocessing"""
        from pathlib import Path
        import json

        # Use the provided output directory
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Create logs directory
        logs_dir = output_dir_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Load and process corpus directly
        corpus_path = Path(self.config['corpus_path'])
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)

        # Convert to corpus.txt format
        corpus_txt_path = output_dir_path / "corpus.txt"
        with open(corpus_txt_path, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(corpus):
                title = doc.get('title', f'doc_{i}')
                text = doc.get('text', '')
                f.write(f"{i},{title}: {text}\n")

            logger.success(f"Processed {len(corpus)} documents")
        return str(output_dir_path)

    def run_phase_5(self, output_dir: str) -> None:
        """Run Phase 5: Generate both phrase and document fingerprints"""
        # First, generate phrase fingerprints
        cmd1 = ["uv", "run", "python", "brain_approaches/semantic_folding/phrase_fingerprints.py",
                "--matrix_path", str(Path(output_dir) / "term_context_matrix.npz"),
                "--coordinates_path", str(Path(output_dir) / "context_coordinates.csv"),
                "--phrases_path", str(Path(output_dir) / "phrases.txt"),
                "--output_dir", output_dir]

        # Then, generate document fingerprints
        cmd2 = ["uv", "run", "python", "brain_approaches/semantic_folding/doc_fingerprints.py",
                "--corpus_path", str(Path(output_dir) / "corpus.txt"),
                "--phrases_path", str(Path(output_dir) / "phrases.txt"),
                "--fingerprints_dir", str(Path(output_dir) / "fingerprints"),
                "--output_dir", output_dir]

        # Execute both commands
        import subprocess
        logger.info("Generating phrase fingerprints...")
        result1 = subprocess.run(cmd1, cwd=os.getcwd(), capture_output=True, text=True)
        if result1.returncode != 0:
            logger.info(f"Phrase fingerprints failed: {result1.stderr}")
            raise RuntimeError("Phase 5a failed")

        logger.info("Generating document fingerprints...")
        result2 = subprocess.run(cmd2, cwd=os.getcwd(), capture_output=True, text=True)
        if result2.returncode != 0:
            logger.info(f"Document fingerprints failed: {result2.stderr}")
            raise RuntimeError("Phase 5b failed")

    def run_specific_phase_non_interactive(self, phase_num: int, output_dir: str) -> str:
        """Run a specific pipeline phase in non-interactive mode"""
        phase_name = self.phases[phase_num]['name']
        logger.info(f"\n>>> Phase {phase_num}: {phase_name}")
        logger.info("=" * 60)

        # Estimate duration for progress indicator
        duration_estimates = {
            1: 10,   # Corpus loading
            2: 120,  # Phrase extraction (can be slow)
            3: 60,   # Term-context matrix
            4: 30,   # Semantic space construction
            5: 180,  # Fingerlogger.info generation (can be slow)
            6: 60    # LanceDB integration
        }

        actual_output_dir = output_dir

        # Phase-specific commands
        if phase_num == 1:
            actual_output_dir = self.run_phase_1(output_dir)
            # For Phase 1, we don't need subprocess since we handled it directly
            logger.success("[SUCCESS] Phase 1 completed successfully!")
            self.show_phase_completion_stats(phase_num, actual_output_dir)
            self.phases[phase_num]["completed"] = True
            return actual_output_dir
        elif phase_num == 5:
            # Special handling for Phase 5 (two-step process)
            actual_output_dir = output_dir
            self.run_phase_5(actual_output_dir)
            logger.success("[SUCCESS] Phase 5 completed successfully!")
            self.show_phase_completion_stats(phase_num, actual_output_dir)
            self.phases[phase_num]["completed"] = True
            return actual_output_dir
        elif phase_num == 2:
            corpus_path = Path(actual_output_dir) / "corpus.txt"
            cmd = ["uv", "run", "python", "brain_approaches/semantic_folding/phrase_extractor.py",
                   "--corpus_path", str(corpus_path), "--output_dir", actual_output_dir]
        elif phase_num == 3:
            phrases_path = Path(actual_output_dir) / "phrases.txt"
            corpus_path = Path(actual_output_dir) / "corpus.txt"
            cmd = ["uv", "run", "python", "brain_approaches/semantic_folding/term_context.py",
                   "--phrases_path", str(phrases_path), "--corpus_path", str(corpus_path),
                   "--output_dir", actual_output_dir]

            if not self.config.get('normalize_matrix', True):
                cmd.append("--no_normalization")
        elif phase_num == 4:
            cmd = ["uv", "run", "python", "brain_approaches/semantic_folding/semantic_space.py",
                   "--corpus_path", str(Path(actual_output_dir) / "corpus.txt"),
                   "--matrix_path", str(Path(actual_output_dir) / "term_context_matrix.npz"),
                   "--output_dir", actual_output_dir,
                   "--grid_size", str(self.config['grid_size']),
                   "--max_edges", str(self.config['max_edges']),
                   "--edge_threshold", str(self.config['edge_threshold'])]
        elif phase_num == 5:
            # Phase 5 consists of two steps: phrase fingerprints and document fingerprints
            # First, generate phrase fingerprints
            cmd1 = ["uv", "run", "python", "phrase_fingerprints.py",
                    "--matrix_path", str(Path(actual_output_dir) / "term_context_matrix.npz"),
                    "--coordinates_path", str(Path(actual_output_dir) / "context_coordinates.csv"),
                    "--phrases_path", str(Path(actual_output_dir) / "phrases.txt"),
                    "--output_dir", actual_output_dir]

            # Then, generate document fingerprints
            cmd2 = ["uv", "run", "python", "doc_fingerprints.py",
                    "--corpus_path", str(Path(actual_output_dir) / "corpus.txt"),
                    "--phrases_path", str(Path(actual_output_dir) / "phrases.txt"),
                    "--fingerprints_dir", str(Path(actual_output_dir) / "fingerprints"),
                    "--output_dir", actual_output_dir,
                    "--top_percent", str(self.config.get('doc_top_percent', 0.05))]

            if self.config.get('doc_no_threshold', False):
                cmd2.append("--no_threshold")

        elif phase_num == 6:
            cmd = ["uv", "run", "python", "brain_approaches/semantic_folding/lance_storage.py",
                   "--corpus_path", str(Path(actual_output_dir) / "corpus.txt"),
                   "--fingerprints_dir", str(Path(actual_output_dir) / "fingerprints"),
                   "--doc_fingerprints_dir", str(Path(actual_output_dir) / "doc_fingerprints"),
                   "--output_dir", actual_output_dir]
        else:
            logger.info(f"Phase {phase_num} execution not yet implemented")
            sys.exit(1)

        logger.info(f"Command: {' '.join(cmd)}")
        logger.info("Starting execution...")

        # Start progress indicator
        progress_thread = self.show_progress_indicator(phase_name, duration_estimates.get(phase_num, 30))

        try:
            result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)

            # Stop progress indicator
            self.stop_progress_indicator(progress_thread)

            if result.returncode == 0:
                logger.success(f"[SUCCESS] Phase {phase_num} completed successfully!")

                # Show some statistics about what was created
                self.show_phase_completion_stats(phase_num, output_dir)

                self.phases[phase_num]["completed"] = True
            else:
                logger.error(f"[FAILED] Phase {phase_num} failed with exit code {result.returncode}")
                if result.stdout:
                    logger.error("STDOUT:")
                    logger.error(result.stdout[-1000:])  # Last 1000 chars
                if result.stderr:
                    logger.error("STDERR:")
                    logger.error(result.stderr[-1000:])  # Last 1000 chars
                sys.exit(result.returncode)
        except Exception as e:
            self.stop_progress_indicator(progress_thread)
            logger.error(f"[ERROR] Error running phase {phase_num}: {e}")
            sys.exit(1)

        return actual_output_dir

    def resume_pipeline(self, resume_state: Dict[str, Any]) -> None:
        """Resume pipeline from saved state"""
        last_output_dir = resume_state['last_output_dir']
        last_phase = resume_state['last_phase']

        logger.info("RESUMING SEMANTIC FOLDING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Output Directory: {last_output_dir}")
        logger.info(f"Resuming from: Phase {last_phase + 1}")
        logger.info(f"Last saved: {resume_state.get('timestamp', 'unknown')}")
        

        # Check if output directory exists
        output_path = Path(last_output_dir)
        if not output_path.exists():
            logger.info(f"ERROR: Output directory {last_output_dir} does not exist")
            sys.exit(1)

        # Update phase completion status based on directory contents
        self.check_phase_completion(output_path)

        # Count remaining phases
        remaining_phases = [p for p in range(last_phase + 1, 7) if not self.phases[p]["completed"]]
        total_remaining = len(remaining_phases)

        if total_remaining == 0:
            logger.info("All phases already completed!")
            self.clear_resume_state()
            return

        logger.info(f"Progress: {last_phase}/{6} phases completed, {total_remaining} remaining")
        

        # Run remaining phases
        completed_count = last_phase
        for i, phase_num in enumerate(remaining_phases, 1):
            logger.info(f"Phase {i}/{total_remaining}: {self.phases[phase_num]['name']}")
            try:
                self.run_specific_phase_non_interactive(phase_num, last_output_dir)
                # Save progress after each phase
                self.save_resume_state(last_output_dir, phase_num)
                completed_count = phase_num
            except SystemExit:
                logger.info(f"Pipeline interrupted at Phase {phase_num}. Progress saved for resume.")
                sys.exit(1)

        # Clear resume state on successful completion
        self.clear_resume_state()
        logger.success("\nPipeline completed successfully!")
        logger.success("All phases have been executed and results are ready.")

    def run_specific_phase(self) -> None:
        """Run a specific pipeline phase"""
        logger.success("\nSelect Phase to Run:")

        phase_choices = []
        for phase_num, phase_info in self.phases.items():
            status = "✅ COMPLETED" if phase_info["completed"] else "⏳ PENDING"
            phase_choices.append(f"{status}: Phase {phase_num} - {phase_info['name']}")

        choice = questionary.select("Select phase:", choices=phase_choices).ask()

        if not choice:
            return

        # Extract phase number
        parts = choice.split(": Phase ")
        if len(parts) < 2:
            return
        phase_num = int(parts[1].split()[0])

        # Get output directory
        last_run = self.check_last_run_status()
        if not last_run:
            logger.info("No output directory found. Please run Phase 1 first or specify output directory.")
            return

        # Run the phase with progress reporting
        self.run_specific_phase_non_interactive(phase_num, str(last_run))

    def view_output_files(self) -> None:
        """View and explore output files"""
        last_run = self.check_last_run_status()
        if not last_run:
            logger.info("No output directory found")
            return

        logger.info(f"\nOutput Directory: {last_run}")

        # Show directory structure
        logger.info("\nDirectory Structure:")
        for root, dirs, files in os.walk(last_run):
            level = root.replace(str(last_run), '').count(os.sep)
            indent = ' ' * 2 * level
            logger.info(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                logger.info(f"{subindent}{file}")
            if len(files) > 5:
                logger.info(f"{subindent}... and {len(files) - 5} more files")

        # Show key statistics
        logger.info("\nKey Statistics:")
        try:
            if (last_run / "corpus.txt").exists():
                with open(last_run / "corpus.txt", 'r', encoding='utf-8') as f:
                    corpus_lines = sum(1 for _ in f)
                logger.info(f"   Corpus passages: {corpus_lines}")

            if (last_run / "phrases.txt").exists():
                with open(last_run / "phrases.txt", 'r', encoding='utf-8') as f:
                    phrases_lines = sum(1 for _ in f)
                logger.info(f"   Phrases extracted: {phrases_lines}")

            fingerprints_dir = last_run / "fingerprints"
            if fingerprints_dir.exists():
                fp_count = len(list(fingerprints_dir.glob("*.txt")))
                logger.info(f"   Phrase fingerprints: {fp_count}")

            doc_fp_dir = last_run / "doc_fingerprints"
            if doc_fp_dir.exists():
                doc_fp_count = len(list(doc_fp_dir.glob("*_fingerlogger.info.txt")))
                logger.info(f"   Document fingerprints: {doc_fp_count}")

        except Exception as e:
            logger.info(f"   ERROR: Error reading statistics: {e}")

    def clean_outputs(self) -> None:
        """Clean old output directories"""
        if not QUESTIONARY_AVAILABLE:
            logger.info("questionary not available")
            return

        if not self.output_base.exists():
            logger.info("No output directory found")
            return

        # List output directories
        output_dirs = [d for d in self.output_base.iterdir() if d.is_dir() and d.name.startswith('musique_')]
        if not output_dirs:
            logger.info("No output directories found")
            return

        logger.info(f"\nFound {len(output_dirs)} output directories:")
        for i, d in enumerate(sorted(output_dirs, reverse=True)):
            size_mb = sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) / (1024 * 1024)
            logger.info(f"   {d.name}: {size_mb:.1f} MB")
        choices = ["All", "None"] + [d.name for d in sorted(output_dirs, reverse=True)]

        to_delete = questionary.checkbox(
            "Select directories to delete:",
            choices=[d.name for d in sorted(output_dirs, reverse=True)]
        ).ask()

        if to_delete:
            for dirname in to_delete:
                dir_path = self.output_base / dirname
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"Deleted: {dirname}")
                except Exception as e:
                    logger.info(f"Failed to delete {dirname}: {e}")
        else:
            logger.info("No directories selected for deletion")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Semantic Folding Pipeline TUI (Optimized Configuration)",
                                     epilog="Examples:\n"
                                            "  %(prog)s                                    # Interactive mode\n"
                                            "  %(prog)s --non-interactive                 # Show status only\n"
                                            "  %(prog)s --resume                          # Resume interrupted pipeline\n"
                                            "  %(prog)s --run-phase 2 --output-dir DIR    # Run specific phase\n"
                                            "\nOptimized Settings:\n"
                                            "  - Grid Size: 32x32 (better distribution)\n"
                                            "  - Max Edges: 200 (cleaner layouts)\n"
                                            "  - Edge Threshold: 0.05 (better connectivity)")
    parser.add_argument("--config", help="Configuration YAML file")
    parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode")
    parser.add_argument("--run-all", action="store_true", help="Run all phases in non-interactive mode")
    parser.add_argument("--run-phase", type=int, choices=[1, 2, 3, 4, 5, 6],
                       help="Run specific phase (1-6) in non-interactive mode")
    parser.add_argument("--output-dir", help="Output directory for phase execution")
    parser.add_argument("--resume", action="store_true", help="Resume from last saved state")

    args = parser.parse_args()

    # Check dependencies
    if not QUESTIONARY_AVAILABLE:
        logger.info("questionary not available. Install with: uv add questionary")
        logger.info("Falling back to basic configuration...")

        # Basic config mode
        config_file = args.config or "config/semantic_folding.yml"
        tui = SemanticFoldingTUI(config_file)
        tui.show_status()
        return

    # Create TUI
    tui = SemanticFoldingTUI(args.config)

    if args.resume:
        # Resume from saved state
        resume_state = tui.load_resume_state()
        if resume_state:
            logger.info(f"Resuming from: {resume_state['last_output_dir']}")
            logger.info(f"Last completed phase: {resume_state['last_phase']}")
            tui.resume_pipeline(resume_state)
        else:
            logger.info("No resume state found. Starting fresh...")
            tui.show_status()
    elif args.run_all:
        # Run all phases in non-interactive mode
        tui.run_all_phases()
    elif args.run_phase:
        # Run specific phase in non-interactive mode
        if not args.output_dir:
            logger.info("Error: --output-dir is required when using --run-phase")
            sys.exit(1)
        result_dir = tui.run_specific_phase_non_interactive(args.run_phase, args.output_dir)
        logger.info(f"\nOutput directory: {result_dir}")
    elif args.non_interactive:
        # Non-interactive mode - just show status
        tui.show_status()
    else:
        # Interactive mode
        tui.run_pipeline_menu()

def test() -> None :
    config_file = "config/semantic_folding.yml"
    tui = SemanticFoldingTUI(config_file)
    phase  = 5
    if phase ==1 : 
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{tui.output_base}/pipeline_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    else : 
        output_dir = "outputs\\pipeline_20260305_182700"
    tui.run_specific_phase_non_interactive(phase, str(output_dir))


if __name__ == "__main__":
    # main()
    test()
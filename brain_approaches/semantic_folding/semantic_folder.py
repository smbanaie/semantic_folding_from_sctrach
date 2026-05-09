#!/usr/bin/env python3
"""
Semantic Folding Pipeline Runner
Interactive TUI for executing the semantic folding pipeline with state management.

This script provides an interactive command-line interface for running a 6-step
semantic folding pipeline:
  1. Phrase Extraction
  2. Term-Context Matrix
  3. Semantic Space
  4. Phrase Fingerprints
  5. Document Fingerprints
  6. Query Processing

Features:
- State persistence across runs (config/exec_state.yml)
- Configuration defaults (config/semantic_folding.yml)
- Dynamic menu based on execution state
- Parameter collection with smart defaults from previous steps
- Run history management
- Extensible visualization system (phrase and document fingerprints)

Architecture:
- SemanticRunner: Main pipeline orchestrator
- VisualizationHandler: Abstract base for visualization operations
- PhraseVisualizationHandler: Phrase fingerprint visualization
- (Future) DocumentVisualizationHandler: Document fingerprint visualization
"""

import os, time
import shutil
import sys
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from lib import get_logger
logger = get_logger("semantic_folder")

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)
logger.add(
    "logs/semantic_runner.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)


# ============================================================================
# TERMINAL COLORS
# ============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER    = '\033[95m'
    BLUE      = '\033[94m'
    CYAN      = '\033[96m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'


# ============================================================================
# VISUALIZATION HANDLER BASE CLASS
# ============================================================================

class VisualizationHandler(ABC):
    """
    Abstract base class for visualization handlers.
    
    Each visualization type (phrase, document, etc.) should implement this
    interface to provide consistent parameter collection and execution.
    """
    
    def __init__(self, runner: 'SemanticRunner'):
        """
        Initialize visualization handler.
        
        Args:
            runner: Reference to parent SemanticRunner instance
        """
        self.runner = runner
    
    @abstractmethod
    def get_step_definition(self) -> Dict[str, Any]:
        """
        Return step definition dict for this visualization type.
        
        Returns:
            Step definition with id, name, script, and parameters
        """
        pass
    
    @abstractmethod
    def collect_parameters(self, run_id: str) -> Optional[Dict[str, str]]:
        """
        Collect parameters for this visualization type.
        
        Args:
            run_id: Current run ID for default path resolution
            
        Returns:
            Dict of parameter names to values, or None if cancelled
        """
        pass
    
    @abstractmethod
    def build_command(self, params: Dict[str, str]) -> List[str]:
        """
        Build CLI command from collected parameters.
        
        Args:
            params: Parameter dict from collect_parameters()
            
        Returns:
            Command as list of strings for subprocess
        """
        pass
    
    def execute(self, run_id: str) -> bool:
        """
        Execute the visualization with user interaction.
        
        Args:
            run_id: Current run ID
            
        Returns:
            True if execution succeeded, False otherwise
        """
        logger.info(f"Starting {self.__class__.__name__}")
        
        # Collect parameters
        params = self.collect_parameters(run_id)
        if params is None:
            self.runner.print_warning("Visualization cancelled")
            return False
        
        # Build command
        cmd = self.build_command(params)
        
        # Execute
        logger.info(f"Executing visualization command")
        logger.debug(f"Full command: {' '.join(cmd)}")
        
        print(f"\n{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Executing:{Colors.ENDC} {' '.join(cmd)}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}\n")
        
        try:
            subprocess.run(cmd, check=True, text=True)
            logger.info("Visualization completed successfully")
            self.runner.print_success("Visualization completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Visualization failed (code {e.returncode})")
            self.runner.print_error(f"Visualization failed (code {e.returncode})")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error during visualization: {e}")
            self.runner.print_error(f"Unexpected error: {e}")
            return False


# ============================================================================
# PHRASE VISUALIZATION HANDLER
# ============================================================================

class PhraseVisualizationHandler(VisualizationHandler):
    """
    Handler for phrase fingerprint visualization.
    
    Supports two modes:
    - Single: Visualize one phrase fingerprint
    - Compare: Side-by-side comparison of two phrase fingerprints
    
    Auto-calculates figure height based on mode and width:
    - Single mode: height = width / 3 (horizontal layout)
    - Compare mode: height constrained between 2/3 and 3/3 of width
    """
    
    def get_step_definition(self) -> Dict[str, Any]:
        """Return step definition for phrase visualization."""
        return {
            'id': 'viz',  # Special ID for visualization (not part of main pipeline)
            'name': 'Phrase Fingerprint Visualization',
            'script': 'brain_approaches/semantic_folding/phrase_visualizer.py',
            'required_params': ['fingerprints', 'output'],
            'optional_params': [
                'grid_size', 'threshold', 'no_morton', 'grid_borders',
                'border_color', 'border_width', 'max_shapes', 'width', 'height',
                'colorscale', 'generate_html', 'generate_png', 'save_metadata'
            ],
            'default_output': 'phrase_visualizations'
        }
    
    def collect_parameters(self, run_id: str) -> Optional[Dict[str, str]]:
        """
        Collect parameters for phrase visualization.
        
        Workflow:
        1. Collect fingerprints directory path
        2. Collect output directory path
        3. Select visualization mode (single/compare)
        4. Collect phrase(s) based on mode
        5. Collect optional styling parameters
        6. Auto-calculate height if width is provided
        
        Args:
            run_id: Current run ID for default path resolution
            
        Returns:
            Dict with all parameters including mode-specific phrases, or None if cancelled
        """
        logger.info("Collecting parameters for phrase visualization")
        params = {}
        
        print(f"\n{Colors.BOLD}Configure: Phrase Fingerprint Visualization{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        
        # ----------------------------------------------------------------
        # REQUIRED PARAMETERS
        # ----------------------------------------------------------------
        
        # 1. Fingerprints directory
        default_fp = self._get_step4_output() or f'outputs/{run_id}/phrase_fingerprints'
        fingerprints = self.runner.get_input(
            f"{Colors.BOLD}fingerprints{Colors.ENDC} (required)",
            default_fp
        )
        while not fingerprints:
            self.runner.print_error("'fingerprints' is required")
            fingerprints = self.runner.get_input(
                f"{Colors.BOLD}fingerprints{Colors.ENDC} (required)",
                default_fp
            )
        params['fingerprints'] = fingerprints
        logger.debug(f"Fingerprints directory: {fingerprints}")
        
        # 2. Output directory
        default_out = f'outputs/{run_id}/phrase_visualizations'
        output = self.runner.get_input(
            f"{Colors.BOLD}output{Colors.ENDC} (required)",
            default_out
        )
        while not output:
            self.runner.print_error("'output' is required")
            output = self.runner.get_input(
                f"{Colors.BOLD}output{Colors.ENDC} (required)",
                default_out
            )
        params['output'] = output
        logger.debug(f"Output directory: {output}")
        
        # 3. Visualization mode
        mode_choice = self.runner.get_choice(
            "Select visualization mode:",
            ['Visualize single phrase', 'Compare two phrases', 'Cancel']
        )
        
        if mode_choice == 3:
            return None
        
        mode = 'single' if mode_choice == 1 else 'compare'
        logger.debug(f"Visualization mode: {mode}")
        
        # 4. Collect phrase(s) based on mode
        if mode == 'single':
            phrase = self.runner.get_input(
                f"{Colors.BOLD}phrase{Colors.ENDC} (required)", None
            )
            while not phrase:
                self.runner.print_error("'phrase' is required for single mode")
                phrase = self.runner.get_input(
                    f"{Colors.BOLD}phrase{Colors.ENDC} (required)", None
                )
            params['phrase'] = phrase
            logger.debug(f"Single mode - phrase: {phrase}")
            
        elif mode == 'compare':
            phrase1 = self.runner.get_input(
                f"{Colors.BOLD}phrase1{Colors.ENDC} (required)", None
            )
            while not phrase1:
                self.runner.print_error("'phrase1' is required for compare mode")
                phrase1 = self.runner.get_input(
                    f"{Colors.BOLD}phrase1{Colors.ENDC} (required)", None
                )
            
            phrase2 = self.runner.get_input(
                f"{Colors.BOLD}phrase2{Colors.ENDC} (required)", None
            )
            while not phrase2:
                self.runner.print_error("'phrase2' is required for compare mode")
                phrase2 = self.runner.get_input(
                    f"{Colors.BOLD}phrase2{Colors.ENDC} (required)", None
                )
            
            params['phrase1'] = phrase1
            params['phrase2'] = phrase2
            logger.debug(f"Compare mode - phrase1: {phrase1}, phrase2: {phrase2}")
        
        # ----------------------------------------------------------------
        # OPTIONAL PARAMETERS
        # ----------------------------------------------------------------
        
        print(f"\n{Colors.CYAN}Optional parameters (Enter to skip):{Colors.ENDC}")
        
        # Collect width first (needed for height calculation)
        width_default = self.runner.get_default_value("width", "viz")
        width_val = self.runner.get_input(f"  width", width_default)
        if width_val:
            params['width'] = width_val
            width = int(width_val)
            
            # Auto-calculate height based on mode
            if mode == 'single':
                height = width // 3
                params['height'] = str(height)
                print(f"  {Colors.GREEN}✓ Auto-calculated height (1/3 width): {height}{Colors.ENDC}")
                logger.info(f"Auto-calculated height for single mode: {height}")
            else:
                # Compare mode: suggest width (square)
                min_height = int(width * 2 / 3)
                max_height = width
                height_default = str(width)
                
                while True:
                    height_val = self.runner.get_input(
                        f"  height (must be between {min_height} and {max_height})",
                        height_default
                    )
                    if not height_val:
                        break
                    
                    height = int(height_val)
                    if min_height <= height <= max_height:
                        params['height'] = str(height)
                        logger.info(f"Height set for compare mode: {height}")
                        break
                    else:
                        self.runner.print_error(
                            f"Height must be between {min_height} and {max_height}"
                        )
        
        # Other optional parameters
        optional_params = [
            ('grid_size', 'Grid size'),
            ('threshold', 'Activation threshold'),
            ('border_color', 'Border color'),
            ('border_width', 'Border width'),
            ('max_shapes', 'Maximum shapes to render'),
            ('colorscale', 'Plotly colorscale name'),
        ]
        
        for param_name, param_prompt in optional_params:
            default = self.runner.get_default_value(param_name, "viz")
            value = self.runner.get_input(f"  {param_prompt}", default)
            if value:
                params[param_name] = value
                logger.debug(f"Optional param - {param_name}: {value}")
        
        # Boolean flags
        bool_params = [
            ('no_morton', 'Use row-major encoding instead of Morton (true/false)'),
            ('grid_borders', 'Show grid borders (true/false)'),
            ('generate_html', 'Generate HTML output (true/false)'),
            ('generate_png', 'Generate PNG output (true/false)'),
            ('save_metadata', 'Save metadata JSON (true/false)')
        ]
        
        for param_name, param_prompt in bool_params:
            default = self.runner.get_default_value(param_name, "viz")
            value = self.runner.get_input(f"  {param_prompt}", default)
            if value:
                params[param_name] = value
                logger.debug(f"Boolean param - {param_name}: {value}")
        
        return params
    
    def _get_step4_output(self) -> Optional[str]:
        """Get Step 4 output directory from last run state."""
        run_id = self.runner.exec_state.get("last_run_id")
        if not run_id:
            return None
        
        run_data = self.runner.exec_state["runs"].get(run_id, {})
        step4_data = run_data.get("steps", {}).get(4)
        
        if step4_data and "output" in step4_data:
            output_path = step4_data["output"]
            if Path(output_path).exists():
                return output_path
        
        return None
    
    def build_command(self, params: Dict[str, str]) -> List[str]:
        """
        Build CLI command for phrase_visualizer.py.
        
        Args:
            params: Parameter dict from collect_parameters()
            
        Returns:
            Command as list of strings for subprocess
        """
        cmd = [
            "E:\\PHD\\GraphRag-Implementations\\YaALI\\"
            "knowledge-graph-builder\\.venv\\scripts\\python",
            "brain_approaches/semantic_folding/phrase_visualizer.py"
        ]
        
        # Add fingerprints and output
        cmd.extend(['--fingerprints', params['fingerprints']])
        cmd.extend(['--output', params['output']])
        
        # Add phrase(s)
        if 'phrase' in params:
            cmd.extend(['--phrase', params['phrase']])
        elif 'phrase1' in params and 'phrase2' in params:
            cmd.extend(['--phrase1', params['phrase1']])
            cmd.extend(['--phrase2', params['phrase2']])
        
        # Add other parameters
        for param, value in params.items():
            if param in ['fingerprints', 'output', 'phrase', 'phrase1', 'phrase2']:
                continue
            
            # Rename parameter if needed
            flag_name = self.runner.CLI_RENAME_MAP.get(param, param)
            flag = f"--{flag_name.replace('_', '-')}"
            
            # Handle boolean parameters with negation flags
            if param in self.runner.NEGATE_FLAG_MAP:
                if str(value).lower() in ("false", "no", "0"):
                    cmd.append(f"--{self.runner.NEGATE_FLAG_MAP[param]}")
                    logger.debug(f"Added negation flag: --{self.runner.NEGATE_FLAG_MAP[param]}")
                
            # Handle regular boolean flags
            elif str(value).lower() in ("true", "false"):
                if str(value).lower() == "true":
                    cmd.append(flag)
                    logger.debug(f"Added boolean flag: {flag}")
                    
            # Handle regular parameters with values
            else:
                cmd.extend([flag, value])
                logger.debug(f"Added param: {flag} {value}")
        
        logger.info(f"Built visualization command: {' '.join(cmd)}")
        return cmd


# ============================================================================
# MAIN PIPELINE RUNNER
# ============================================================================

class SemanticRunner:
    """
    Interactive runner for the semantic folding pipeline.
    
    Manages execution state, parameter collection, and step execution for a
    multi-stage semantic folding pipeline. Provides a TUI for starting new runs,
    continuing existing runs, rerunning steps, and visualizing results.
    
    Architecture:
    - Pipeline steps are defined in PIPELINE_STEPS
    - Visualization operations are handled by VisualizationHandler subclasses
    - State is persisted to config/exec_state.yml
    - Configuration is loaded from config/semantic_folding.yml
    """

    # ========================================================================
    # CONFIGURATION MAPPINGS
    # ========================================================================

    # Maps parameter names to their location in the YAML config file
    CONFIG_PATH_IN_YAML = {
        # Global parameters
        "grid_size":            ["grid_size"],
        "min_freq" :            ["min_freq"],
        "keep_verbs":           ["keep_verbs"],

        # Phase 1: Phrase Extraction
        "min_word_length":      ["phrase_extraction", "min_word_length"],
        "no_spacy":             ["phrase_extraction", "no_spacy"],
        "max_ngram":            ["phrase_extraction", "max_ngram"],
        "no_filter_generic":    ["phrase_extraction", "no_filter_generic"],
        "stats":                ["phrase_extraction", "stats"],

        # Phase 2: Term-Context Matrix
        "no_tfidf":            ["term_context_matrix", "no_tfidf"],

        # Phase 3: Semantic Space
        "method":               ["semantic_space", "method"],
        "visualize":            ["semantic_space", "visualize"],
        "show_density":         ["semantic_space", "show_density"],
        "enable_grid":          ["semantic_space", "enable_grid"],
        "grid_padding":         ["semantic_space", "grid_padding"],
        "collision_resolution": ["semantic_space", "collision_resolution"],
        "n_jobs":               ["semantic_space", "n_jobs"],
        "use_sparse":           ["semantic_space", "use_sparse"],

        # Phase 5: Document Fingerprints
        "top_percent":          ["document_fingerprints", "top_percent"],
        "no_normalize":         ["document_fingerprints", "no_normalize"],
        "normalize_method":     ["document_fingerprints", "normalize_method"],
        "compute_diversity":    ["document_fingerprints", "compute_diversity"],
        "diversity_sample":     ["document_fingerprints", "diversity_sample"],

        # Phase 6: Query Processing
        "weighting":            ["query_processing", "weighting"],
        "idf_weights":            ["query_processing", "idf_weights"],
        "top_k":                ["query_processing", "top_k"],
        "spreading_steps":      ["query_processing", "spreading_steps"],

        # Phrase Visualization
        "threshold":            ["phrase_visualization", "threshold"],
        "no_morton":            ["phrase_visualization", "encoding"],
        "grid_borders":         ["phrase_visualization", "grid_borders"],
        "border_color":         ["phrase_visualization", "border_color"],
        "border_width":         ["phrase_visualization", "border_width"],
        "max_shapes":           ["phrase_visualization", "max_shapes"],
        "width":                ["phrase_visualization", "figure_width"],
        "height":               ["phrase_visualization", "figure_height"],
        "colorscale":           ["phrase_visualization", "colorscale"],
        "generate_html":        ["phrase_visualization", "generate_html"],
        "generate_png":         ["phrase_visualization", "generate_png"],
        "save_metadata":        ["phrase_visualization", "save_metadata"],
    }

    # ========================================================================
    # PIPELINE STEP DEFINITIONS
    # ========================================================================

    PIPELINE_STEPS = [
        {
            "id": 1,
            "name": "Phrase Extraction",
            "script": "brain_approaches/semantic_folding/phrase_extractor.py",
            "required_params": ["corpus", "output"],
            "optional_params": [
                "min_freq", "min_word_length", "no_spacy",
                "no_filter_generic", "keep_verbs", "stats"
            ],
            "default_output": "extracted_phrases",
            "extra_outputs": {
                "vocab":   lambda output: str(Path(output) / "vocabulary.csv"),
                "mapping": lambda output: str(Path(output) / "phrase_to_contexts.json"),
            },
            "depends_on": []
        },
        {
            "id": 2,
            "name": "Term-Context Matrix",
            "script": "brain_approaches/semantic_folding/term_context.py",
            "required_params": ["corpus", "vocab", "mapping", "output"],
            "optional_params": ["no_tfidf"],
            "default_output": "term_context_matrix",
            "extra_outputs": {
                "matrix_npz" : lambda output: str(Path(output) / "term_context_matrix.npz"),
                "metadata"   : lambda output: str(Path(output) / "term_context_matrix.json"),
                "idf_weights": lambda output: str(Path(output) / "idf_weights.json"),
            },
            "depends_on": [1]
        },
        {
            "id": 3,
            "name": "Semantic Space",
            "script": "brain_approaches/semantic_folding/semantic_space.py",
            "required_params": ["matrix_npz", "metadata", "output"],
            "optional_params": [
                "method", "grid_size", "visualize", "show_density"
            ],
            "default_output": "semantic_space",
            "extra_outputs": {
                "coordinates": lambda output: str(Path(output) / "context_coordinates.json"),
            },
            "depends_on": [2]
        },
        {
            "id": 4,
            "name": "Phrase Fingerprints",
            "script": "brain_approaches/semantic_folding/phrase_fingerprints.py",
            "required_params": ["coordinates", "metadata", "output"],
            "optional_params": ["grid_size"],
            "default_output": "phrase_fingerprints",
            "depends_on": [3]
        },
        {
            "id": 5,
            "name": "Document Fingerprints",
            "script": "brain_approaches/semantic_folding/doc_fingerprints.py",
            "required_params": ["corpus", "fingerprints", "output"],
            "optional_params": [
                "top_percent", "no_normalize", "normalize_method",
                "compute_diversity", "diversity_sample"
            ],
            "default_output": "document_fingerprints",
            "depends_on": [4]
        },
        {
            "id": 6,
            "name": "Query Processing",
            "script": "brain_approaches/semantic_folding/query_processor.py",
            "required_params": [
                "query", "fingerprints", "doc_fingerprints",
                "output"
            ],
            "optional_params": [
                "weighting", "idf_weights", "top_k", "spreading_steps", "grid_size"
            ],
            "default_output": "query_results",
            "depends_on": [2, 4, 5]
        }
    ]

    # ========================================================================
    # CLI PARAMETER MAPPINGS
    # ========================================================================

    # Maps internal parameter names to CLI flag names
    CLI_RENAME_MAP = {
        "matrix_npz":       "matrix",
        "metadata":         "metadata",
        "coordinates":      "coordinates",
        "fingerprints":     "fingerprints",
        "doc_fingerprints": "doc-fingerprints",
        "idf_weights":      "idf-weights",
        "max_ngram":        "max-ngram",
        "min_word_length":  "min-word-length",
        "no_spacy":         "no-spacy",
        "no_filter_generic": "no-filter-generic",
        "keep_verbs":       "keep-verbs",
        "use_tfidf":        "use-tfidf",
        "no_tfidf":         "no-tfidf",
        "show_density":     "show-density",
        "enable_grid":      "enable-grid",
        "grid_padding":     "grid-padding",
        "collision_resolution": "collision-resolution",
        "n_jobs":           "n-jobs",
        "use_sparse":       "use-sparse",
        "top_percent":      "top-percent",
        "normalize_method": "normalize-method",
        "no_normalize":     "no-normalize",
        "compute_diversity": "compute-diversity",
        "diversity_sample": "diversity-sample",
        "top_k":            "top-k",
        "spreading_steps":  "spreading-steps",
        "no_morton":        "no-morton",
        "grid_borders":     "grid-borders",
        "border_color":     "border-color",
        "border_width":     "border-width",
        "max_shapes":       "max-shapes",
        "generate_html":    "generate-html",
        "generate_png":     "generate-png",
        "save_metadata":    "save-metadata",
    }

    # Maps boolean parameters to their negation flags
    NEGATE_FLAG_MAP = {
        "no_morton": "morton",  # If no_morton=false, use --morton
    }

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(self):
        """Initialize the semantic runner with state and config management."""
        self.state_file = Path("config/exec_state.yml")
        self.config_file = Path("config/semantic_folding.yml")
        self.exec_state = self.load_state()
        self.config = self.load_config()
        
        # Initialize visualization handlers
        self.viz_handlers = {
            'phrase': PhraseVisualizationHandler(self)
        }
        
        logger.info("SemanticRunner initialized")
        logger.debug(f"State file: {self.state_file}")
        logger.debug(f"Config file: {self.config_file}")

    def view_manage_runs(self):
        """View and manage historical runs with deletion options."""
        while True:
            self.print_header("Run Management")
            
            # Load current state
            self.exec_state = self.load_state()
            runs = self.exec_state.get("runs", {})
            
            if not runs:
                print(f"{Colors.YELLOW}No runs found in history.{Colors.ENDC}\n")
                input(f"{Colors.GREEN}Press Enter to return to main menu...{Colors.ENDC}")
                return
            
            # Display runs
            print(f"{Colors.GREEN}Historical Runs:{Colors.ENDC}\n")
            for idx, (run_id, run_data) in enumerate(sorted(runs.items()), 1):
                corpus = run_data.get("corpus", "N/A")
                timestamp = run_data.get("timestamp", "N/A")
                completed = run_data.get("completed_steps", [])
                print(f"  {idx}. {Colors.CYAN}{run_id}{Colors.ENDC}")
                print(f"     Corpus: {corpus}")
                print(f"     Created: {timestamp}")
                print(f"     Completed steps: {completed}")
                print()
            
            # Check for orphaned directories
            orphaned = self._find_orphaned_directories(runs)
            if orphaned:
                print(f"{Colors.YELLOW}Orphaned Directories (not in run history):{Colors.ENDC}\n")
                for idx, orphan_path in enumerate(orphaned, 1):
                    size = self._get_directory_size(orphan_path)
                    print(f"  {idx}. {orphan_path} ({size})")
                print()
            
            # Build menu options
            options = ["Delete specific run", "Delete all runs"]
            if orphaned:
                options.extend(["Delete specific orphaned directory", "Delete all orphaned directories"])
            options.append("Back to main menu")
            
            choice = self.get_choice("Run Management:", options)
            selected = options[choice - 1]
            
            if selected == "Back to main menu":
                return
            elif selected == "Delete specific run":
                self._delete_specific_run(runs)
            elif selected == "Delete all runs":
                self._delete_all_runs(runs)
            elif selected == "Delete specific orphaned directory":
                self._delete_specific_orphaned(orphaned)
            elif selected == "Delete all orphaned directories":
                self._delete_all_orphaned(orphaned)

    def _find_orphaned_directories(self, runs: dict) -> list:
        """Find output directories that exist but are not in run history."""
        output_base = Path("output")
        if not output_base.exists():
            return []
        
        # Get all run IDs from history
        known_runs = set(runs.keys())
        
        # Get all directories in output/
        existing_dirs = [d for d in output_base.iterdir() if d.is_dir()]
        
        # Find orphaned directories
        orphaned = []
        for dir_path in existing_dirs:
            # Check if this directory belongs to any known run
            is_orphaned = True
            for run_id in known_runs:
                if dir_path.name.startswith(run_id) or run_id in dir_path.name:
                    is_orphaned = False
                    break
            
            if is_orphaned:
                orphaned.append(dir_path)
        
        return sorted(orphaned)

    def _get_directory_size(self, path: Path) -> str:
        """Calculate and format directory size."""
        try:
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            # Format size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if total_size < 1024.0:
                    return f"{total_size:.1f} {unit}"
                total_size /= 1024.0
            return f"{total_size:.1f} TB"
        except Exception as e:
            logger.warning(f"Could not calculate size for {path}: {e}")
            return "unknown size"

    def _delete_specific_run(self, runs: dict):
        """Delete a specific run and its associated directories."""
        run_list = sorted(runs.keys())
        options = [f"{run_id} ({runs[run_id].get('corpus', 'N/A')})" for run_id in run_list]
        options.append("Cancel")
        
        choice = self.get_choice("Select run to delete:", options)
        
        if options[choice - 1] == "Cancel":
            return
        
        run_id = run_list[choice - 1]
        
        # Confirm deletion
        print(f"\n{Colors.RED}Are you sure you want to delete run '{run_id}'?{Colors.ENDC}")
        print(f"{Colors.RED}This will remove all associated output directories.{Colors.ENDC}")
        confirm = input(f"{Colors.YELLOW}Type 'yes' to confirm: {Colors.ENDC}").strip().lower()
        
        if confirm != "yes":
            print(f"{Colors.YELLOW}Deletion cancelled.{Colors.ENDC}")
            time.sleep(1)
            return
        
        # Delete directories associated with this run
        deleted_dirs = self._delete_run_directories(run_id)
        
        # Remove from state
        del self.exec_state["runs"][run_id]
        
        # Update last_run_id if it was the deleted one
        if self.exec_state.get("last_run_id") == run_id:
            self.exec_state["last_run_id"] = None
            self.exec_state["last_step"] = None
        
        self.save_state()
        
        print(f"\n{Colors.GREEN}Run '{run_id}' deleted successfully.{Colors.ENDC}")
        if deleted_dirs:
            print(f"{Colors.GREEN}Deleted directories:{Colors.ENDC}")
            for dir_path in deleted_dirs:
                print(f"  - {dir_path}")
        time.sleep(2)

    def _delete_all_runs(self, runs: dict):
        """Delete all runs and their associated directories."""
        print(f"\n{Colors.RED}WARNING: This will delete ALL runs and their output directories!{Colors.ENDC}")
        confirm = input(f"{Colors.YELLOW}Type 'DELETE ALL' to confirm: {Colors.ENDC}").strip()
        
        if confirm != "DELETE ALL":
            print(f"{Colors.YELLOW}Deletion cancelled.{Colors.ENDC}")
            time.sleep(1)
            return
        
        # Delete all run directories
        all_deleted = []
        for run_id in runs.keys():
            deleted_dirs = self._delete_run_directories(run_id)
            all_deleted.extend(deleted_dirs)
        
        # Clear state
        self.exec_state["runs"] = {}
        self.exec_state["last_run_id"] = None
        self.exec_state["last_step"] = None
        self.save_state()
        
        print(f"\n{Colors.GREEN}All runs deleted successfully.{Colors.ENDC}")
        if all_deleted:
            print(f"{Colors.GREEN}Deleted {len(all_deleted)} directories.{Colors.ENDC}")
        time.sleep(2)

    def _delete_run_directories(self, run_id: str) -> list:
        """Delete all directories associated with a run ID."""
        output_base = Path("output")
        if not output_base.exists():
            return []
        
        deleted = []
        for dir_path in output_base.iterdir():
            if dir_path.is_dir() and (dir_path.name.startswith(run_id) or run_id in dir_path.name):
                try:
                    shutil.rmtree(dir_path)
                    deleted.append(dir_path)
                    logger.info(f"Deleted directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Failed to delete {dir_path}: {e}")
                    print(f"{Colors.RED}Failed to delete {dir_path}: {e}{Colors.ENDC}")
        
        return deleted

    def _delete_specific_orphaned(self, orphaned: list):
        """Delete a specific orphaned directory."""
        options = [str(path) for path in orphaned]
        options.append("Cancel")
        
        choice = self.get_choice("Select orphaned directory to delete:", options)
        
        if options[choice - 1] == "Cancel":
            return
        
        dir_path = orphaned[choice - 1]
        
        # Confirm deletion
        print(f"\n{Colors.RED}Are you sure you want to delete '{dir_path}'?{Colors.ENDC}")
        confirm = input(f"{Colors.YELLOW}Type 'yes' to confirm: {Colors.ENDC}").strip().lower()
        
        if confirm != "yes":
            print(f"{Colors.YELLOW}Deletion cancelled.{Colors.ENDC}")
            time.sleep(1)
            return
        
        try:
            shutil.rmtree(dir_path)
            print(f"\n{Colors.GREEN}Directory '{dir_path}' deleted successfully.{Colors.ENDC}")
            logger.info(f"Deleted orphaned directory: {dir_path}")
        except Exception as e:
            print(f"{Colors.RED}Failed to delete directory: {e}{Colors.ENDC}")
            logger.error(f"Failed to delete {dir_path}: {e}")
        
        time.sleep(2)

    def _delete_all_orphaned(self, orphaned: list):
        """Delete all orphaned directories."""
        print(f"\n{Colors.RED}This will delete {len(orphaned)} orphaned directories.{Colors.ENDC}")
        confirm = input(f"{Colors.YELLOW}Type 'yes' to confirm: {Colors.ENDC}").strip().lower()
        
        if confirm != "yes":
            print(f"{Colors.YELLOW}Deletion cancelled.{Colors.ENDC}")
            time.sleep(1)
            return
        
        deleted_count = 0
        failed_count = 0
        
        for dir_path in orphaned:
            try:
                shutil.rmtree(dir_path)
                deleted_count += 1
                logger.info(f"Deleted orphaned directory: {dir_path}")
            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to delete {dir_path}: {e}")
                print(f"{Colors.RED}Failed to delete {dir_path}: {e}{Colors.ENDC}")
        
        print(f"\n{Colors.GREEN}Deleted {deleted_count} orphaned directories.{Colors.ENDC}")
        if failed_count > 0:
            print(f"{Colors.YELLOW}Failed to delete {failed_count} directories.{Colors.ENDC}")
        
        time.sleep(2)

    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================

    def load_state(self) -> Dict[str, Any]:
        """Load execution state from YAML file."""
        if not self.state_file.exists():
            logger.info(f"State file not found, creating new one at {self.state_file}")
            # Create empty state structure
            empty_state = {
                "last_run_id": None,
                "last_step": None,
                "runs": {}
            }
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                yaml.dump(empty_state, f, default_flow_style=False)
            return empty_state
        
        try:
            with open(self.state_file, 'r') as f:
                state = yaml.safe_load(f) or {}
            
            # Ensure required keys exist
            if "last_run_id" not in state:
                state["last_run_id"] = None
            if "last_step" not in state:
                state["last_step"] = None
            if "runs" not in state:
                state["runs"] = {}
            
            return state
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return {
                "last_run_id": None,
                "last_step": None,
                "runs": {}
            }

    def save_state(self):
        """Save execution state to config/exec_state.yml."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.exec_state, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Saved state to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            self.print_error(f"Failed to save state: {e}")

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from config/semantic_folding.yml.
        
        Returns:
            Config dict with default parameters for all pipeline steps
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                logger.info(f"Loaded config from {self.config_file}")
                return config
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                self.print_error(f"Failed to load config: {e}")
                return {}
        else:
            logger.warning(f"Config file not found: {self.config_file}")
            return {}

    def get_default_value(self, param_name: str, step_id: Any) -> Optional[str]:
        """
        Get default value for a parameter from config file.
        
        Args:
            param_name: Parameter name (e.g., "grid_size", "min_freq")
            step_id: Step ID (1-6) or "viz" for visualization
            
        Returns:
            Default value as string, or None if not found
        """
        if param_name not in self.CONFIG_PATH_IN_YAML:
            return None
        
        path = self.CONFIG_PATH_IN_YAML[param_name]
        value = self.config
        
        try:
            for key in path:
                value = value[key]
            
            # Convert to string for consistency
            if isinstance(value, bool):
                return "true" if value else "false"
            return str(value)
        except (KeyError, TypeError):
            return None

    # ========================================================================
    # RUN MANAGEMENT
    # ========================================================================

    def create_new_run(self, corpus: str) -> str:
        """
        Create a new run with the given corpus path.
        
        Args:
            corpus: Path to corpus file (already validated)
            
        Returns:
            New run ID
        """
        # Generate run ID
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize run data
        self.exec_state["runs"][run_id] = {
            "corpus": corpus,
            "created_at": datetime.now().isoformat(),
            "steps": {}
        }
        
        # Update last_run_id
        self.exec_state["last_run_id"] = run_id
        
        # Save state
        self.save_state()
        
        logger.debug(f"Created new run: {run_id}")
        
        return run_id

    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """Get run information from state."""
        return self.exec_state["runs"].get(run_id, {})

    def update_step_completion(self, run_id: str, step_id: int, output: str, params: Dict[str, str]):
        """
        Update state after step completion.
        
        Args:
            run_id: Current run ID
            step_id: Completed step ID
            output: Output directory path
            params: Parameters used for this step
        """
        if run_id not in self.exec_state["runs"]:
            logger.error(f"Run {run_id} not found in state")
            return
        
        self.exec_state["runs"][run_id]["steps"][step_id] = {
            "completed_at": datetime.now().isoformat(),
            "output": output,
            "params": params
        }
        
        # Evaluate and save extra_outputs if they exist
        step_config = self.PIPELINE_STEPS[step_id - 1]
        if "extra_outputs" in step_config:
            for key, path_func in step_config["extra_outputs"].items():
                file_path = path_func(output)
                if Path(file_path).exists():
                    self.exec_state["runs"][run_id]["steps"][step_id][key] = file_path
                else:
                    logger.warning(f"Expected output file not found: {file_path}")
        
        self.exec_state["last_run_id"] = run_id
        self.exec_state["last_step"] = step_id
        
        self.save_state()
        logger.info(f"Updated completion for run {run_id}, step {step_id}")

    # ========================================================================
    # PARAMETER RESOLUTION
    # ========================================================================

    def resolve_parameter_from_previous_step(
        self, 
        run_id: str, 
        param_name: str, 
        step_id: int
    ) -> Optional[str]:
        run_data = self.get_run_info(run_id)
        if not run_data:
            logger.debug(f"[resolve] No run_data found for run_id={run_id}")
            return None
        
        completed_steps = run_data.get("steps", {})
        logger.debug(f"[resolve] Resolving '{param_name}' for step {step_id}. Completed steps: {list(completed_steps.keys())}")
        
        for prev_step_id in range(step_id - 1, 0, -1):
            if prev_step_id not in completed_steps:
                logger.debug(f"[resolve]   Step {prev_step_id}: not in completed_steps, skipping")
                continue
            
            prev_step_data = completed_steps[prev_step_id]
            prev_step_def = self.get_step_definition(prev_step_id)
            
            logger.debug(f"[resolve]   Step {prev_step_id}: keys in step_data = {list(prev_step_data.keys())}")
            
            if not prev_step_def:
                logger.debug(f"[resolve]   Step {prev_step_id}: no step_def found, skipping")
                continue
            
            # Don't auto-resolve output parameter
            if param_name == "output" and "output" in prev_step_data:
                logger.debug(f"[resolve]   Step {prev_step_id}: skipping 'output' param")
                continue
            
            # Direct lookup in saved step state
            if param_name in prev_step_data:
                resolved = prev_step_data[param_name]
                logger.debug(f"[resolve]   Step {prev_step_id}: found '{param_name}' directly = '{resolved}'")
                if resolved and isinstance(resolved, str) and Path(resolved).exists():
                    logger.debug(f"[resolve]   Step {prev_step_id}: path exists, returning '{resolved}'")
                    return resolved
                else:
                    logger.debug(f"[resolve]   Step {prev_step_id}: value invalid or path does not exist: '{resolved}'")
            else:
                logger.debug(f"[resolve]   Step {prev_step_id}: '{param_name}' NOT in step_data")
            
            # Check extra_outputs
            extra_outputs = prev_step_def.get("extra_outputs", {})
            logger.debug(f"[resolve]   Step {prev_step_id}: extra_outputs keys = {list(extra_outputs.keys())}")
            if param_name in extra_outputs:
                output_path = prev_step_data.get("output")
                logger.debug(f"[resolve]   Step {prev_step_id}: found in extra_outputs, output_path = '{output_path}'")
                if output_path:
                    resolved = extra_outputs[param_name](output_path)
                    logger.debug(f"[resolve]   Step {prev_step_id}: extra_outputs resolved = '{resolved}', exists = {Path(resolved).exists()}")
                    if Path(resolved).exists():
                        return resolved
            
            # Match common patterns
            if "output" in prev_step_data:
                output_path = prev_step_data["output"]
                if param_name == "corpus" and prev_step_id == 1:
                    corpus = run_data.get("corpus")
                    logger.debug(f"[resolve]   Step {prev_step_id}: corpus pattern matched, corpus = '{corpus}'")
                    if corpus:
                        return corpus
                elif param_name == "fingerprints" and prev_step_id == 4:
                    logger.debug(f"[resolve]   Step {prev_step_id}: fingerprints pattern matched, returning '{output_path}'")
                    return output_path
                elif param_name == "doc_fingerprints" and prev_step_id == 5:
                    logger.debug(f"[resolve]   Step {prev_step_id}: doc_fingerprints pattern matched, returning '{output_path}'")
                    return output_path
        
        logger.debug(f"[resolve] '{param_name}' could not be resolved from any previous step")
        return None

    def get_step_definition(self, step_id: int) -> Optional[Dict[str, Any]]:
        """Get step definition by ID."""
        for step in self.PIPELINE_STEPS:
            if step["id"] == step_id:
                return step
        return None

    # ========================================================================
    # PARAMETER COLLECTION
    # ========================================================================

    def collect_step_parameters(self, step: Dict[str, Any], run_id: str) -> Optional[Dict[str, str]]:
        """
        Collect parameters for a pipeline step.
        
        Resolution order:
        1. Previous step outputs (for dependencies)
        2. User input (required parameters)
        3. Config file defaults (optional parameters)
        4. User input (optional parameters, if user wants to override)
        
        Args:
            step: Step definition dict
            run_id: Current run ID
            
        Returns:
            Dict of parameter names to values, or None if cancelled
        """
        logger.info(f"Collecting parameters for step {step['id']}: {step['name']}")
        params = {}
        
        print(f"\n{Colors.BOLD}Configure: {step['name']}{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        
        # ----------------------------------------------------------------
        # REQUIRED PARAMETERS
        # ----------------------------------------------------------------
        
        for param in step["required_params"]:
            # Try to resolve from previous steps
            resolved = self.resolve_parameter_from_previous_step(run_id, param, step["id"])
            
            if param == "output":
                # Special handling for output parameter
                default = f"outputs/{run_id}/{step['default_output']}"
            elif param == "corpus":
                # Use corpus from run info
                run_data = self.get_run_info(run_id)
                default = run_data.get("corpus") or resolved
            else:
                default = resolved
            
            value = self.get_input(f"{Colors.BOLD}{param}{Colors.ENDC} (required)", default)
            
            while not value:
                self.print_error(f"'{param}' is required")
                value = self.get_input(f"{Colors.BOLD}{param}{Colors.ENDC} (required)", default)
            
            params[param] = value
            logger.debug(f"Required param - {param}: {value}")
        
        # ----------------------------------------------------------------
        # OPTIONAL PARAMETERS
        # ----------------------------------------------------------------
        
        # OPTIONAL PARAMETERS
        if step["optional_params"]:
            print(f"\n{Colors.CYAN}Optional parameters (Enter to skip):{Colors.ENDC}")
            
            for param in step["optional_params"]:
                # Try to resolve from previous steps first
                resolved = self.resolve_parameter_from_previous_step(run_id, param, step["id"])
                
                # Fall back to config default if not resolved
                config_default = self.get_default_value(param, step["id"])
                
                # Use resolved value if available, otherwise config default
                default = resolved if resolved else config_default
                
                value = self.get_input(f"  {param}", default)
                if value:
                    params[param] = value
                    logger.debug(f"Optional param - {param}: {value}")
        
        return params

    # ========================================================================
    # COMMAND EXECUTION
    # ========================================================================

    def build_command(self, step: Dict[str, Any], params: Dict[str, str]) -> List[str]:
        """
        Build CLI command from step definition and parameters.
        
        Args:
            step: Step definition dict
            params: Parameter dict from collect_step_parameters()
            
        Returns:
            Command as list of strings for subprocess
        """
        cmd = [
            "E:\\PHD\\GraphRag-Implementations\\YaALI\\"
            "knowledge-graph-builder\\.venv\\scripts\\python",
            step["script"]
        ]
        
        for param, value in params.items():
            # Rename parameter if needed
            flag_name = self.CLI_RENAME_MAP.get(param, param)
            flag = f"--{flag_name.replace('_', '-')}"
            
            # Handle boolean parameters with negation flags
            if param in self.NEGATE_FLAG_MAP:
                if str(value).lower() in ("false", "no", "0"):
                    cmd.append(f"--{self.NEGATE_FLAG_MAP[param]}")
                    logger.debug(f"Added negation flag: --{self.NEGATE_FLAG_MAP[param]}")
            
            # Handle regular boolean flags
            elif str(value).lower() in ("true", "false"):
                if str(value).lower() == "true":
                    cmd.append(flag)
                    logger.debug(f"Added boolean flag: {flag}")
            
            # Handle regular parameters with values
            else:
                cmd.extend([flag, value])
                logger.debug(f"Added param: {flag} {value}")
        
        logger.info(f"Built command: {' '.join(cmd)}")
        return cmd

    def execute_step(self, step: Dict[str, Any], run_id: str) -> bool:
        """
        Execute a pipeline step.
        
        Args:
            step: Step definition dict
            run_id: Current run ID
            
        Returns:
            True if execution succeeded, False otherwise
        """
        logger.info(f"Executing step {step['id']}: {step['name']}")
        
        # Collect parameters
        params = self.collect_step_parameters(step, run_id)
        if params is None:
            self.print_warning("Step cancelled")
            return False
        
        # Build command
        cmd = self.build_command(step, params)
        
        # Execute
        logger.info(f"Running command: {' '.join(cmd)}")
        print(f"\n{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Executing:{Colors.ENDC} {' '.join(cmd)}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}\n")
        
        try:
            subprocess.run(cmd, check=True, text=True)
            logger.info(f"Step {step['id']} completed successfully")
            
            # Update state
            output = params.get("output", "")
            self.update_step_completion(run_id, step["id"], output, params)
            
            self.print_success(f"Step {step['id']} completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Step {step['id']} failed (code {e.returncode})")
            self.print_error(f"Step {step['id']} failed (code {e.returncode})")
            return False
            
        except Exception as e:
            logger.exception(f"Unexpected error during step {step['id']}: {e}")
            self.print_error(f"Unexpected error: {e}")
            return False


    # ========================================================================
    # USER INTERFACE
    # ========================================================================

    def print_header(self, text: str):
        """Print formatted header."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")

    def print_success(self, text: str):
        """Print success message."""
        print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

    def print_error(self, text: str):
        """Print error message."""
        print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

    def print_warning(self, text: str):
        """Print warning message."""
        print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

    def get_input(self, prompt: str, default: Any = None) -> str:
        """
        Get user input with optional default value.
        
        Args:
            prompt: Input prompt text
            default: Default value (shown in yellow if provided)
            
        Returns:
            User input or default value
        """
        if default is not None:
            # Show default in green
            full_prompt = f"{prompt} [{Colors.YELLOW}{default}{Colors.ENDC}]: "
        else:
            full_prompt = f"{prompt}: "
        
        value = input(full_prompt).strip()
        return value if value else (str(default) if default is not None else "")
        
    def get_choice(self, prompt: str, options: List[str]) -> int:
        """
        Get user choice from a list of options.
        
        Args:
            prompt: Choice prompt text
            options: List of option strings
            
        Returns:
            Selected option number (1-indexed)
        """
        print(f"\n{Colors.BOLD}{prompt}{Colors.ENDC}")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        
        while True:
            try:
                choice = int(input(f"\n{Colors.BOLD}Enter choice (1-{len(options)}): {Colors.ENDC}"))
                if 1 <= choice <= len(options):
                    return choice
                else:
                    self.print_error(f"Please enter a number between 1 and {len(options)}")
            except ValueError:
                self.print_error("Please enter a valid number")

    # ========================================================================
    # MAIN MENU
    # ========================================================================

    def show_main_menu(self):
        """Display main menu and handle user choice."""
        self.print_header("Semantic Folding Pipeline Runner")
        
        # Build menu options based on state
        options = []
        
        # Option 1: Start new run
        options.append("Start new run")
        
        # Option 2: Select/Change current run/step (if runs exist)
        if self.exec_state["runs"]:
            options.append("Change current run/step")
        
        # Option 3: Continue from last step (if last_run_id exists)
        if self.exec_state.get("last_run_id"):
            last_run = self.exec_state["last_run_id"]
            last_step = 5 if self.exec_state.get("last_step") >= 6 else self.exec_state.get("last_step")
            
            if last_step:
                options.append(f"Continue from step {last_step + 1} (run: {last_run})")
            else:
                options.append(f"Start step 1 (run: {last_run})")
        
        # Option 4: Manage runs (if runs exist)
        if self.exec_state["runs"]:
            options.append("Manage runs")
        
        # Option 5: Visualize results
        options.append("Visualize results")
        
        # Option 6: Exit
        options.append("Exit")
        
        choice = self.get_choice("Main Menu:", options)
        
        # Handle choice
        if options[choice - 1] == "Start new run":
            self.start_new_run()
        elif options[choice - 1] == "Change current run/step":
            self.select_run()
        elif "Continue from step" in options[choice - 1] or "Start step 1" in options[choice - 1]:
            self.continue_run()
        elif options[choice - 1] == "Manage runs":
            self.view_manage_runs()
        elif options[choice - 1] == "Visualize results":
            self.visualize_menu()
        elif options[choice - 1] == "Exit":
            print(f"\n{Colors.GREEN}Goodbye!{Colors.ENDC}\n")
            sys.exit(0)

    def start_new_run(self):
        """Start a new pipeline run."""
        self.print_header("Start New Run")
        
        # Get default corpus path from config
        default_corpus = self.config.get("paths", {}).get("corpus_path", None)
        
        logger.debug(f"self.config keys: {list(self.config.keys())}")
        logger.debug(f"paths section: {self.config.get('paths', {})}")
        logger.debug(f"default_corpus: {default_corpus}")
        
        corpus = self.get_input(f"{Colors.BOLD}Corpus path{Colors.ENDC} (required)", default_corpus)
        while not corpus or not Path(corpus).exists():
            if not corpus:
                self.print_error("Corpus path is required")
            else:
                self.print_error(f"File not found: {corpus}")
            corpus = self.get_input(f"{Colors.BOLD}Corpus path{Colors.ENDC} (required)", default_corpus)
        
        run_id = self.create_new_run(corpus)
        self.print_success(f"Created run: {run_id}")
        
        # Start from step 1
        self.run_pipeline(run_id, start_step=1)

    def select_run(self):
        """Select an existing run to work with."""
        self.print_header("Select Run")
        
        runs = list(self.exec_state["runs"].keys())
        if not runs:
            self.print_warning("No existing runs found")
            return
        
        # Display runs with details
        print(f"\n{Colors.BOLD}Available runs:{Colors.ENDC}")
        for i, run_id in enumerate(runs, 1):
            run_data = self.get_run_info(run_id)
            corpus = run_data.get("corpus", "N/A")
            created = run_data.get("created_at", "N/A")
            completed_steps = len(run_data.get("steps", {}))
            print(f"  {i}. {run_id}")
            print(f"     Corpus: {corpus}")
            print(f"     Created: {created}")
            print(f"     Completed steps: {completed_steps}/6")
        
        choice = self.get_choice("Select run:", [f"{run_id}" for run_id in runs])
        selected_run = runs[choice - 1]
        
        # Update last_run_id
        self.exec_state["last_run_id"] = selected_run
        self.save_state()
        
        self.print_success(f"Selected run: {selected_run}")
        
        # Ask what to do next
        run_data = self.get_run_info(selected_run)
        completed_steps = run_data.get("steps", {})
        
        if completed_steps:
            last_completed = max(completed_steps.keys())
            next_step = last_completed + 1
            
            options = []
            if next_step <= 6:
                options.append(f"Continue from step {next_step}")
            options.append("Re-run a specific step")
            options.append("Back to main menu")
            
            choice = self.get_choice("What would you like to do?", options)
            
            if options[choice - 1].startswith("Continue"):
                self.run_pipeline(selected_run, start_step=next_step)
            elif options[choice - 1] == "Re-run a specific step":
                self.select_step(selected_run)
            else:
                return
        else:
            # No steps completed, start from step 1
            self.run_pipeline(selected_run, start_step=1)

    def continue_run(self):
        """Continue from the last executed step."""
        run_id = self.exec_state.get("last_run_id")
        if not run_id:
            self.print_error("No run to continue")
            return
        
        last_step = self.exec_state.get("last_step")
        if last_step is None:
            start_step = 1
        elif last_step >= 6:  # All steps completed - restart from step 6
            start_step = 6
        else:
            start_step = last_step + 1
        
        if start_step > 6:
            self.print_warning("All steps completed")
            return
        
        self.run_pipeline(run_id, start_step=start_step)

    def select_step(self, run_id: str):
        """Select a specific step to run."""
        self.print_header("Select Step")
        
        options = [f"Step {step['id']}: {step['name']}" for step in self.PIPELINE_STEPS]
        choice = self.get_choice("Select step to run:", options)
        
        step = self.PIPELINE_STEPS[choice - 1]
        
        # Check dependencies
        if not self.check_dependencies(run_id, step):
            self.print_error("Dependencies not met. Please complete previous steps first.")
            return
        
        success = self.execute_step(step, run_id)
        
        if success:
            # Ask if user wants to continue
            if step["id"] < 6:
                cont = self.get_input(f"\n{Colors.BOLD}Continue to next step?{Colors.ENDC} (y/n)", "y")
                if cont.lower() == "y":
                    self.run_pipeline(run_id, start_step=step["id"] + 1)

    def check_dependencies(self, run_id: str, step: Dict[str, Any]) -> bool:
        """
        Check if all dependencies for a step are met.
        
        Args:
            run_id: Current run ID
            step: Step definition dict
            
        Returns:
            True if all dependencies are met, False otherwise
        """
        run_data = self.get_run_info(run_id)
        completed_steps = set(run_data.get("steps", {}).keys())
        
        depends_on = step.get("depends_on", [])
        for dep_step_id in depends_on:
            if dep_step_id not in completed_steps:
                logger.warning(f"Step {step['id']} depends on step {dep_step_id} which is not completed")
                return False
        
        return True

    def run_pipeline(self, run_id: str, start_step: int = 1):
        """
        Run pipeline steps sequentially.
        
        Args:
            run_id: Current run ID
            start_step: Step ID to start from (1-6)
        """
        self.print_header(f"Running Pipeline (Run: {run_id})")
        
        for step in self.PIPELINE_STEPS:
            if step["id"] < start_step:
                continue
            
            # Check dependencies
            if not self.check_dependencies(run_id, step):
                self.print_error(f"Cannot run step {step['id']}: dependencies not met")
                break
            
            # Execute step
            success = self.execute_step(step, run_id)
            
            if not success:
                self.print_error(f"Pipeline stopped at step {step['id']}")
                break
            
            # Ask if user wants to continue (except for last step)
            if step["id"] < 6:
                cont = self.get_input(f"\n{Colors.BOLD}Continue to next step?{Colors.ENDC} (y/n)", "y")
                if cont.lower() != "y":
                    self.print_warning("Pipeline paused")
                    break
        else:
            # All steps completed
            self.print_success("Pipeline completed successfully!")

    # ========================================================================
    # VISUALIZATION MENU
    # ========================================================================

    def visualize_menu(self):
        """Display visualization menu."""
        self.print_header("Visualization")
        
        options = [
            "Phrase Extraction Visualization",
            "Back to main menu"
        ]
        
        choice = self.get_choice("Select visualization type:", options)
        
        if choice == 1:
            self.viz_handlers['phrase'].handle()
        elif choice == 2:
            return

    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================

    def run(self):
        """Main entry point for the runner."""
        try:
            while True:
                self.show_main_menu()
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.ENDC}")
            sys.exit(0)
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            self.print_error(f"Unexpected error: {e}")
            sys.exit(1)


# ============================================================================
# VISUALIZATION HANDLERS
# ============================================================================

class PhraseVisualizationHandler:
    """Handler for phrase extraction visualization."""
    
    def __init__(self, runner: SemanticRunner):
        self.runner = runner
    
    def handle(self):
        """Handle phrase visualization workflow."""
        self.runner.print_header("Phrase Extraction Visualization")
        
        # Get parameters
        params = self.collect_parameters()
        if not params:
            return
        
        # Build and execute command
        cmd = self.build_command(params)
        
        print(f"\n{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Executing:{Colors.ENDC} {' '.join(cmd)}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}\n")
        
        try:
            subprocess.run(cmd, check=True, text=True)
            self.runner.print_success("Visualization completed successfully")
        except subprocess.CalledProcessError as e:
            self.runner.print_error(f"Visualization failed (code {e.returncode})")
        except Exception as e:
            logger.exception(f"Unexpected error during visualization: {e}")
            self.runner.print_error(f"Unexpected error: {e}")
    
    def collect_parameters(self) -> Optional[Dict[str, str]]:
        """Collect parameters for phrase visualization."""
        params = {}
        
        print(f"\n{Colors.BOLD}Configure: Phrase Visualization{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        
        # Required parameters
        phrases_file = self.runner.get_input(
            f"{Colors.BOLD}phrases_file{Colors.ENDC} (required)", 
            None
        )
        while not phrases_file or not Path(phrases_file).exists():
            if not phrases_file:
                self.runner.print_error("phrases_file is required")
            else:
                self.runner.print_error(f"File not found: {phrases_file}")
            phrases_file = self.runner.get_input(
                f"{Colors.BOLD}phrases_file{Colors.ENDC} (required)", 
                None
            )
        params["phrases_file"] = phrases_file
        
        output = self.runner.get_input(
            f"{Colors.BOLD}output{Colors.ENDC} (required)", 
            "outputs/phrase_viz"
        )
        while not output:
            self.runner.print_error("output is required")
            output = self.runner.get_input(
                f"{Colors.BOLD}output{Colors.ENDC} (required)", 
                "outputs/phrase_viz"
            )
        params["output"] = output
        
        # Optional parameters with config defaults
        print(f"\n{Colors.CYAN}Optional parameters (Enter to skip):{Colors.ENDC}")
        
        optional_params = {
            "no_morton": "viz.phrase.no_morton",
            "grid_borders": "viz.phrase.grid_borders",
            "border_color": "viz.phrase.border_color",
            "border_width": "viz.phrase.border_width",
            "max_shapes": "viz.phrase.max_shapes",
            "generate_html": "viz.phrase.generate_html",
            "generate_png": "viz.phrase.generate_png",
            "save_metadata": "viz.phrase.save_metadata"
        }
        
        for param, config_path in optional_params.items():
            default = self.runner.get_default_value(param, "viz")
            value = self.runner.get_input(f"  {param}", default)
            if value:
                params[param] = value
        
        return params
    
    def build_command(self, params: Dict[str, str]) -> List[str]:
        """Build visualization command."""
        cmd = [
            "E:\\PHD\\GraphRag-Implementations\\YaALI\\"
            "knowledge-graph-builder\\.venv\\scripts\\python",
            "brain_approaches/semantic_folding/visualize_phrases.py"
        ]
        
        for param, value in params.items():
            flag_name = self.runner.CLI_RENAME_MAP.get(param, param)
            flag = f"--{flag_name.replace('_', '-')}"
            
            # Handle boolean parameters
            if str(value).lower() in ("true", "false"):
                if str(value).lower() == "true":
                    cmd.append(flag)
            else:
                cmd.extend([flag, value])
        
        return cmd


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    runner = SemanticRunner()
    runner.run()


if __name__ == "__main__":
    main()

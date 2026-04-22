#!/usr/bin/env python3
"""
Semantic Folding Pipeline Runner
Interactive TUI for executing the semantic folding pipeline with state management.
"""

import os
import shutil
import sys
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger


# Configure loguru
logger.remove()
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


class Colors:
    HEADER    = '\033[95m'
    BLUE      = '\033[94m'
    CYAN      = '\033[96m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'


class SemanticRunner:
    """Interactive runner for the semantic folding pipeline."""

    # ------------------------------------------------------------------
    # CONFIG_PATH_IN_YAML
    # ------------------------------------------------------------------
    CONFIG_PATH_IN_YAML = {
        # Global
        "grid_size":            ["grid_size"],
        "min_freq" :            ["min_freq"],
        "keep_verbs":           ["keep_verbs"],

        # Phase 1
        "min_word_length":      ["phrase_extraction", "min_word_length"],
        "use_spacy":            ["phrase_extraction", "use_spacy"],
        "max_ngram":            ["phrase_extraction", "max_ngram"],
        "filter_generic":       ["phrase_extraction", "filter_generic"],
        "stats":                ["phrase_extraction", "stats"],

        # Phase 2
        "use_tfidf":            ["term_context_matrix", "use_tfidf"],

        # Phase 3
        "method":               ["semantic_space", "method"],
        "visualize":            ["semantic_space", "visualize"],
        "show_density":         ["semantic_space", "show_density"],
        "enable_grid":          ["semantic_space", "enable_grid"],
        "grid_padding":         ["semantic_space", "grid_padding"],
        "collision_resolution": ["semantic_space", "collision_resolution"],
        "n_jobs":               ["semantic_space", "n_jobs"],
        "use_sparse":           ["semantic_space", "use_sparse"],

        # Phase 5
        "top_percent":          ["document_fingerprints", "top_percent"],
        "normalize":            ["document_fingerprints", "normalize"],
        "normalize_method":     ["document_fingerprints", "normalize_method"],
        "compute_diversity":    ["document_fingerprints", "compute_diversity"],
        "diversity_sample":     ["document_fingerprints", "diversity_sample"],

        # Phase 6
        "weighting":            ["query_processing", "weighting"],
        "idf":                  ["query_processing", "idf"],
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

    # ------------------------------------------------------------------
    # VISUALIZATION_STEP — sidecar, not part of PIPELINE_STEPS
    # ------------------------------------------------------------------
    VISUALIZATION_STEP = {
        "id":     "viz",
        "name":   "Phrase Fingerprint Visualization",
        "script": "brain_approaches/semantic_folding/phrase_visualizer.py",
    }

    # ------------------------------------------------------------------
    # PIPELINE_STEPS
    # ------------------------------------------------------------------
    PIPELINE_STEPS = [
        {
            "id": 1,
            "name": "Phrase Extraction",
            "script": "brain_approaches/semantic_folding/phrase_extractor.py",
            "required_params": ["corpus", "output"],
            "optional_params": [
                "min_freq", "min_word_length", "use_spacy",
                "filter_generic", "keep_verbs", "stats"
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
            "optional_params": ["use_tfidf"],
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
            "required_params": ["matrix", "metadata", "output"],
            "optional_params": [
                "method", "grid_size", "visualize", "show_density"
            ],
            "default_output": "semantic_space",
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
                "idf_weights", "grid_size", "top_percent", "normalize",
                "normalize_method", "use_spacy", "keep_verbs", "filter_generic",
                "min_word_length"
            ],
            "default_output": "doc_fingerprints",
            "depends_on": [3, 4]
        },
        {
            "id": 6,
            "name": "Query Processing",
            "script": "brain_approaches/semantic_folding/query_processing.py",
            "required_params": [
                "query",
                "phrase_fp_dir",
                "doc_fp_dir",
                "weighting",
            ],
            "optional_params": [
                "idf",
                "top_k",
                "spreading_steps",
                "keep_verbs",
                "output",
                "grid_size"
            ],
            "default_output": "query_results.json",
            "depends_on": [5]
        },
    ]

    # ------------------------------------------------------------------
    # CLI flag renaming
    # ------------------------------------------------------------------
    CLI_RENAME_MAP = {
        "phrase_fp_dir":   "phrase-fp-dir",
        "doc_fp_dir":      "doc-fp-dir",
        "spreading_steps": "spreading-steps",
        "min_phrase_freq": "min-freq"
    }

    NEGATE_FLAG_MAP = {
        "use_spacy":           "no-spacy",
        "filter_generic":      "no-filter-generic",
        "use_tfidf":           "no-tfidf",
        "use_word_boundaries": "no-word-boundaries",
        "enable_grid":         "no-grid",
        "keep_verbs":          "no-verbs",
        "normalize":           "no-normalize",
        # visualization negations
        "generate_html":       "no-html",
        "generate_png":        "no-png",
        "save_metadata":       "no-metadata",
        "grid_borders":        "no-grid-borders",
    }

    def __init__(self):
        self.config_dir       = Path("config")
        self.exec_state_path  = self.config_dir / "exec_state.yml"
        self.config_path      = self.config_dir / "semantic_folding.yml"

        self.config_dir.mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        logger.info("Initializing SemanticRunner")
        self.config     = self.load_config()
        self.exec_state = self.load_exec_state()
        logger.debug(f"Config loaded from: {self.config_path}")
        logger.debug(f"Exec state loaded from: {self.exec_state_path}")

    # ------------------------------------------------------------------
    # Config / state I/O
    # ------------------------------------------------------------------

    def load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}, using empty config")
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded config with {len(config)} top-level keys")
        return config

    def load_exec_state(self) -> Dict[str, Any]:
        if not self.exec_state_path.exists():
            logger.info("No existing exec state found, starting fresh")
            return {"last_run_id": None, "last_step": None, "runs": {}}
        with open(self.exec_state_path, "r", encoding="utf-8") as f:
            state = yaml.safe_load(f) or {
                "last_run_id": None, "last_step": None, "runs": {}
            }
        logger.info(f"Loaded exec state: {len(state.get('runs', {}))} previous run(s)")
        return state

    def save_exec_state(self):
        with open(self.exec_state_path, "w", encoding="utf-8") as f:
            yaml.dump(self.exec_state, f, default_flow_style=False,
                    sort_keys=False, allow_unicode=True)
        logger.debug(f"Exec state saved to {self.exec_state_path}")

    # ------------------------------------------------------------------
    # TUI helpers
    # ------------------------------------------------------------------

    def print_header(self, text: str):
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.ENDC}\n")

    def print_success(self, text: str):
        print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

    def print_error(self, text: str):
        print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

    def print_warning(self, text: str):
        print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

    def get_input(self, prompt: str, default: Optional[str] = None) -> str:
        if default is not None:
            prompt = f"{prompt} [{Colors.YELLOW}{default}{Colors.ENDC}]: "
        else:
            prompt = f"{prompt}: "
        value = input(prompt).strip()
        return value if value else (default if default is not None else "")

    def get_choice(self, prompt: str, options: List[str]) -> int:
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"  {Colors.BOLD}{i}.{Colors.ENDC} {option}")
        while True:
            try:
                choice = int(
                    input(f"\n{Colors.BOLD}Enter choice (1-{len(options)}): {Colors.ENDC}")
                )
                if 1 <= choice <= len(options):
                    return choice
                print(f"{Colors.RED}Invalid choice.{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.RED}Please enter a number.{Colors.ENDC}")

    def generate_run_id(self) -> str:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.debug(f"Generated run ID: {run_id}")
        return run_id

    def get_step_by_id(self, step_id: int) -> Optional[Dict[str, Any]]:
        for step in self.PIPELINE_STEPS:
            if step["id"] == step_id:
                return step
        logger.warning(f"Step ID {step_id} not found")
        return None

    # ------------------------------------------------------------------
    # Default value resolution
    # ------------------------------------------------------------------

    def get_default_value(self, param: str, step_id) -> Optional[str]:
        _MISSING = object()

        if self.exec_state.get("last_run_id"):
            run_id   = self.exec_state["last_run_id"]
            run_data = self.exec_state["runs"].get(run_id)

            if run_data:
                param_mapping = {
                    "vocab":       "vocabulary.csv",
                    "mapping":     "phrase_to_contexts.json",
                    "phrases":     "vocabulary.csv",
                    "matrix":      "term_context_matrix.npz",
                    "metadata":    "term_context_matrix.json",
                    "coordinates": "context_coordinates.json",
                    "idf":         "idf_weights.json",
                    "idf_weights": "idf_weights.json",
                    "som":         "som_model.pkl"
                }

                # Only iterate numeric step IDs
                if isinstance(step_id, int):
                    for prev_step_id in range(step_id - 1, 0, -1):
                        step_data = run_data["steps"].get(prev_step_id)
                        if not step_data:
                            continue
                        extra = step_data.get("extra_outputs", {})
                        if param in extra:
                            logger.debug(f"Default '{param}' from step {prev_step_id} extra_outputs")
                            return extra[param]
                        if "output" in step_data and param in param_mapping:
                            output_path = step_data["output"]
                            if Path(output_path).is_file() and param_mapping[param] in output_path:
                                return output_path
                            candidate = str(Path(output_path) / param_mapping[param])
                            if Path(candidate).exists():
                                logger.debug(f"Default '{param}' from step {prev_step_id} output dir")
                                return candidate

                # Same-step parameters
                step_params = run_data["steps"].get(step_id, {}).get("parameters", {})
                if param in step_params:
                    logger.debug(f"Default '{param}' from previous run step {step_id} params")
                    return str(step_params[param])

                if param == "coordinates":
                    step3_out = run_data["steps"].get(3, {}).get("output", "")
                    if step3_out:
                        candidate = str(Path(step3_out) / "context_coordinates.json")
                        if Path(candidate).exists():
                            return candidate

                if param in ("fingerprints", "phrase_fp_dir", "phrase-fp-dir"):
                    step4_out = run_data["steps"].get(4, {}).get("output", "")
                    if step4_out and Path(step4_out).exists():
                        logger.debug(f"Default '{param}' from Step 4 output: {step4_out}")
                        return step4_out

                if param in ("doc_fp_dir", "doc-fp-dir"):
                    step5_out = run_data["steps"].get(5, {}).get("output", "")
                    if step5_out and Path(step5_out).exists():
                        logger.debug(f"Default 'doc_fp_dir' from Step 5 output: {step5_out}")
                        return step5_out

                if param in ("idf", "idf_weights"):
                    step2_out = run_data["steps"].get(2, {}).get("output", "")
                    if step2_out:
                        candidate = str(Path(step2_out) / "idf_weights.json")
                        if Path(candidate).exists():
                            logger.debug(f"Default '{param}' resolved from Step 2 dir: {candidate}")
                            return candidate

        # YAML config fallback
        if param not in self.CONFIG_PATH_IN_YAML or not self.config:
            return None

        path  = self.CONFIG_PATH_IN_YAML[param]
        value = self.config

        for key in path:
            if not isinstance(value, dict):
                return None
            value = value.get(key, _MISSING)
            if value is _MISSING:
                return None

        if value is None:
            return None

        # Special: encoding is stored as "Morton"/"Row-major", map to bool string
        if param == "no_morton":
            return "false" if str(value).lower() == "morton" else "true"

        resolved = str(value)
        logger.debug(f"Default '{param}' from YAML config: {resolved}")
        return resolved

    # ------------------------------------------------------------------
    # Output path helper
    # ------------------------------------------------------------------

    def get_output_path(self, step: Dict[str, Any], run_id: str) -> str:
        output_base = self.config.get("paths", {}).get("output_base", "outputs")
        output_dir  = Path(output_base) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / step["default_output"])
        logger.debug(f"Output path for step {step['id']}: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Parameter collection
    # ------------------------------------------------------------------

    def collect_parameters(
        self, step: Dict[str, Any], run_id: str
    ) -> Dict[str, str]:
        logger.info(f"Collecting parameters for step {step['id']}: {step['name']}")
        params = {}

        print(f"\n{Colors.BOLD}Configure: {step['name']}{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")

        for param in step["required_params"]:
            if param == "output":
                default = self.get_output_path(step, run_id)
            elif param == "corpus":
                default = (
                    self.get_default_value(param, step["id"])
                    or self.config.get("paths", {}).get("corpus_path", "data/corpus.txt")
                )
            elif param == "query":
                default = None
            else:
                default = self.get_default_value(param, step["id"])

            value = self.get_input(
                f"{Colors.BOLD}{param}{Colors.ENDC} (required)", default
            )
            while not value:
                self.print_error(f"'{param}' is required")
                value = self.get_input(
                    f"{Colors.BOLD}{param}{Colors.ENDC} (required)", default
                )

            params[param] = value
            logger.debug(f"Required param — {param}: {value}")

        print(f"\n{Colors.CYAN}Optional parameters (Enter to skip):{Colors.ENDC}")

        for param in step["optional_params"]:
            if param == "output":
                default = self.get_output_path(step, run_id)
            else:
                default = self.get_default_value(param, step["id"])

            value = self.get_input(f"  {param}", default)

            if value:
                if isinstance(value, str) and value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                params[param] = value
                logger.debug(f"Optional param — {param}: {value}")
            elif default is not None and value == default:
                if isinstance(default, str) and default.lower() in ("true", "false"):
                    params[param] = default.lower() == "true"
                else:
                    params[param] = default
                logger.debug(f"Optional param (default kept) — {param}: {params[param]}")

        logger.info(
            f"Parameter collection done for step {step['id']}: {len(params)} params"
        )
        return params

    # ------------------------------------------------------------------
    # Command builder
    # ------------------------------------------------------------------

    def build_command(self, step: Dict, params: Dict) -> List[str]:
        cmd = [
            "E:\\PHD\\GraphRag-Implementations\\YaALI\\"
            "knowledge-graph-builder\\.venv\\scripts\\python",
            step["script"],
        ]

        for key, value in params.items():
            if isinstance(value, bool):
                if key in self.NEGATE_FLAG_MAP:
                    if not value:
                        cmd.append(f"--{self.NEGATE_FLAG_MAP[key]}")
                else:
                    if value:
                        cli_key = self.CLI_RENAME_MAP.get(key, key.replace("_", "-"))
                        cmd.append(f"--{cli_key}")
            else:
                if key == "output" and step["id"] in [1, 2, 5]:
                    cmd.extend(["--output-dir", str(value)])
                else:
                    cli_key = self.CLI_RENAME_MAP.get(key, key.replace("_", "-"))
                    cmd.extend([f"--{cli_key}", str(value)])

        logger.debug(f"Built command: {' '.join(cmd)}")
        return cmd

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def execute_step(
        self, step: Dict[str, Any], params: Dict[str, str], run_id: str
    ) -> bool:
        cmd = self.build_command(step, params)

        logger.info(f"Executing step {step['id']}: {step['name']} | run_id={run_id}")
        logger.debug(f"Full command: {' '.join(cmd)}")

        print(f"\n{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Executing:{Colors.ENDC} {' '.join(cmd)}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}\n")

        try:
            subprocess.run(cmd, check=True, text=True)

            if run_id not in self.exec_state["runs"]:
                self.exec_state["runs"][run_id] = {
                    "created_at": datetime.now().isoformat(),
                    "steps": {},
                }

            output_path = params.get("output", self.get_output_path(step, run_id))

            extra_outputs = {}
            for label, path_fn in step.get("extra_outputs", {}).items():
                extra_outputs[label] = path_fn(output_path)
                logger.debug(f"Extra output — {label}: {extra_outputs[label]}")

            step_record = {
                "name":         step["name"],
                "completed_at": datetime.now().isoformat(),
                "parameters":   params,
                "output":       output_path,
            }
            if extra_outputs:
                step_record["extra_outputs"] = extra_outputs

            self.exec_state["runs"][run_id]["steps"][step["id"]] = step_record
            self.exec_state["last_run_id"] = run_id
            self.exec_state["last_step"]   = step["id"]
            self.save_exec_state()

            logger.info(f"Step {step['id']} completed — output: {output_path}")
            self.print_success(
                f"Step {step['id']}: {step['name']} completed successfully"
            )
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Step {step['id']} failed (code {e.returncode})")
            self.print_error(
                f"Step {step['id']}: {step['name']} failed (code {e.returncode})"
            )
            return False
        except Exception as e:
            logger.exception(f"Unexpected error in step {step['id']}: {e}")
            self.print_error(f"Unexpected error: {e}")
            return False

    # ------------------------------------------------------------------
    # Phrase visualization
    # ------------------------------------------------------------------

    def _step4_output_available(self) -> Optional[str]:
        """Return Step 4 output dir if it exists in the last run, else None."""
        run_id = self.exec_state.get("last_run_id")
        if not run_id:
            return None
        step4 = self.exec_state["runs"].get(run_id, {}).get("steps", {}).get(4)
        if step4 and Path(step4["output"]).exists():
            return step4["output"]
        return None

    def _viz_output_dir(self, run_id: str) -> str:
        output_base = self.config.get("paths", {}).get("output_base", "outputs")
        viz_dir = Path(output_base) / run_id / "phrase_visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        return str(viz_dir)

    def _get_viz_default(self, param: str) -> Optional[str]:
        """Pull defaults from phrase_visualization config section."""
        return self.get_default_value(param, "viz")

    def _collect_viz_params(self, run_id: str, mode: str) -> Optional[Dict[str, Any]]:
        """
        Collect parameters for phrase_visualizer.py.
        mode: "single" | "compare"
        Returns a dict ready for build_command, or None if user cancels.
        """
        step4_out = self._step4_output_available()
        viz_cfg   = self.config.get("phrase_visualization", {})

        params: Dict[str, Any] = {}

        print(f"\n{Colors.BOLD}Configure: Phrase Fingerprint Visualization{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")

        # --fingerprints
        fp_default = step4_out or ""
        fp = self.get_input(f"{Colors.BOLD}fingerprints{Colors.ENDC} (required)", fp_default)
        if not fp:
            self.print_error("'fingerprints' is required")
            return None
        params["fingerprints"] = fp

        # --output
        out_default = self._viz_output_dir(run_id)
        params["output"] = self.get_input(
            f"{Colors.BOLD}output{Colors.ENDC} (required)", out_default
        ) or out_default

        # phrase(s)
        if mode == "single":
            phrase = self.get_input(f"{Colors.BOLD}phrase{Colors.ENDC} (required)")
            while not phrase:
                self.print_error("'phrase' is required")
                phrase = self.get_input(f"{Colors.BOLD}phrase{Colors.ENDC} (required)")
            params["phrase"] = phrase
        else:
            phrase1 = self.get_input(f"{Colors.BOLD}phrase1{Colors.ENDC} (required)")
            while not phrase1:
                self.print_error("'phrase1' is required")
                phrase1 = self.get_input(f"{Colors.BOLD}phrase1{Colors.ENDC} (required)")
            phrase2 = self.get_input(f"{Colors.BOLD}phrase2{Colors.ENDC} (required)")
            while not phrase2:
                self.print_error("'phrase2' is required")
                phrase2 = self.get_input(f"{Colors.BOLD}phrase2{Colors.ENDC} (required)")
            params["phrase1"] = phrase1
            params["phrase2"] = phrase2

        # Optional params with config defaults
        print(f"\n{Colors.CYAN}Optional parameters (Enter to skip):{Colors.ENDC}")

        # Collect width first
        width_default = str(viz_cfg.get("figure_width", 1800))
        width_val = self.get_input(f"  width", width_default)
        width = int(width_val) if width_val else int(width_default)
        params["width"] = width

        # Handle height based on mode
        if mode == "single":
            # Single phrase: height = width // 3
            height = width // 3
            params["height"] = height
            print(f"  height (auto-calculated): {height}")
        else:
            # Compare mode: height must be between 2/3 and 3/3 of width
            min_height = int(width * 2 / 3)
            max_height = width
            height_default = str(viz_cfg.get("figure_height", max_height))
            
            while True:
                height_val = self.get_input(
                    f"  height (must be between {min_height} and {max_height})", 
                    height_default
                )
                height = int(height_val) if height_val else int(height_default)
                
                if min_height <= height <= max_height:
                    params["height"] = height
                    break
                else:
                    self.print_error(
                        f"Height must be between {min_height} and {max_height} "
                        f"(2/3 to 3/3 of width {width})"
                    )

        optional_viz = [
            ("grid_size",     str(viz_cfg.get("grid_size",     self.config.get("grid_size", 128)))),
            ("threshold",     str(viz_cfg.get("threshold",     0.0))),
            ("border_color",  str(viz_cfg.get("border_color",  "lightgray"))),
            ("border_width",  str(viz_cfg.get("border_width",  1.0))),
            ("max_shapes",    str(viz_cfg.get("max_shapes",    300))),
            ("colorscale",    str(viz_cfg.get("colorscale",    "Blues"))),
        ]
        for param, default in optional_viz:
            val = self.get_input(f"  {param}", default)
            if val:
                params[param] = val

        # Boolean flags — show current default, let user toggle
        bool_flags = [
            ("no_morton",     viz_cfg.get("encoding", "Morton").lower() != "morton"),
            ("grid_borders",  viz_cfg.get("grid_borders",  True)),
            ("generate_html", viz_cfg.get("generate_html", True)),
            ("generate_png",  viz_cfg.get("generate_png",  True)),
            ("save_metadata", viz_cfg.get("save_metadata", True)),
        ]
        for flag, default_val in bool_flags:
            default_str = "true" if default_val else "false"
            val = self.get_input(f"  {flag} (true/false)", default_str)
            if val:
                params[flag] = val.lower() == "true"
            else:
                params[flag] = default_val

        logger.info(f"Visualization params collected: {len(params)} params")
        return params

    def run_phrase_visualization(self):
        """Interactive menu for running phrase_visualizer.py after Step 4."""
        step4_out = self._step4_output_available()
        if not step4_out:
            self.print_error("Step 4 (Phrase Fingerprints) must be completed first")
            return

        run_id = self.exec_state.get("last_run_id")
        if not run_id:
            self.print_error("No active run found")
            return

        self.print_header("Phrase Fingerprint Visualization")

        choice = self.get_choice(
            "Select visualization mode:",
            ["Visualize single phrase", "Compare two phrases", "Cancel"]
        )

        if choice == 3:
            return

        mode = "single" if choice == 1 else "compare"
        params = self._collect_viz_params(run_id, mode)

        if not params:
            self.print_warning("Visualization cancelled")
            return

        # Build command for phrase_visualizer.py
        viz_step = self.VISUALIZATION_STEP.copy()
        cmd = [
            "E:\\PHD\\GraphRag-Implementations\\YaALI\\"
            "knowledge-graph-builder\\.venv\\scripts\\python",
            viz_step["script"],
        ]

        # Add required args
        cmd.extend(["--fingerprints", params["fingerprints"]])
        cmd.extend(["--output", params["output"]])

        if mode == "single":
            cmd.extend(["--phrase", params["phrase"]])
        else:
            cmd.extend(["--phrase1", params["phrase1"]])
            cmd.extend(["--phrase2", params["phrase2"]])

        # Add optional args
        for key in ["grid_size", "threshold", "border_color", "border_width",
                    "max_shapes", "width", "height", "colorscale"]:
            if key in params:
                cli_key = key.replace("_", "-")
                cmd.extend([f"--{cli_key}", str(params[key])])

        # Boolean flags
        if params.get("no_morton", False):
            cmd.append("--no-morton")
        if not params.get("grid_borders", True):
            cmd.append("--no-grid-borders")
        if not params.get("generate_html", True):
            cmd.append("--no-html")
        if not params.get("generate_png", True):
            cmd.append("--no-png")
        if not params.get("save_metadata", True):
            cmd.append("--no-metadata")

        logger.info(f"Executing phrase visualization | mode={mode}")
        logger.debug(f"Full command: {' '.join(cmd)}")

        print(f"\n{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Executing:{Colors.ENDC} {' '.join(cmd)}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}\n")

        # Don't modify exec_state - visualization is a side operation
        try:
            subprocess.run(cmd, check=True, text=True)
            logger.info("Phrase visualization completed successfully")
            self.print_success("Phrase visualization completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Visualization failed (code {e.returncode})")
            self.print_error(f"Visualization failed (code {e.returncode})")
        except Exception as e:
            logger.exception(f"Unexpected error during visualization: {e}")
            self.print_error(f"Unexpected error: {e}")


    def run(self):
        self.print_header("Semantic Folding Pipeline Runner")

        while True:
            print(f"\n{Colors.BOLD}Main Menu{Colors.ENDC}")
            print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")

            # Show current run status if exists
            if self.exec_state.get("last_run_id"):
                run_id = self.exec_state["last_run_id"]
                last_step = self.exec_state.get("last_step")
                
                if last_step is not None:
                    step = self.get_step_by_id(last_step)
                    step_name = step["name"] if step else f"Step {last_step}"
                    print(
                        f"{Colors.YELLOW}Current run:{Colors.ENDC} {run_id} "
                        f"(last step: {last_step} - {step_name})"
                    )
                else:
                    print(f"{Colors.YELLOW}Current run:{Colors.ENDC} {run_id} (no steps completed)")

            # Build dynamic menu based on state
            options = []
            
            # Always show "Start new run"
            options.append("Start new run")
            
            has_active_run = self.exec_state.get("last_run_id") is not None
            last_step = self.exec_state.get("last_step")
            
            # Only show "Continue to next step" if there's an active run
            if has_active_run:
                if last_step is None:
                    options.append("Continue to next step (Step 1: Corpus Preparation)")
                elif last_step < len(self.PIPELINE_STEPS):
                    next_step = self.get_step_by_id(last_step + 1)
                    next_name = next_step["name"] if next_step else f"Step {last_step + 1}"
                    options.append(f"Continue to next step (Step {last_step + 1}: {next_name})")
            
            # Only show "Rerun last step" if there's a completed step
            if has_active_run and last_step is not None:
                step = self.get_step_by_id(last_step)
                step_name = step["name"] if step else f"Step {last_step}"
                options.append(f"Rerun last step (Step {last_step}: {step_name})")
            
            # Show visualization option if Step 4 is complete
            if self._step4_output_available():
                options.append("Visualize phrase fingerprints")
            
            # Show history options if runs exist
            if self.exec_state.get("runs"):
                options.append("View run history")
                options.append("Clear history")
            
            # Always show Exit
            options.append("Exit")

            choice = self.get_choice("Select an option:", options)
            selected = options[choice - 1]

            if selected == "Exit":
                logger.info("User exited")
                print(f"\n{Colors.GREEN}Goodbye!{Colors.ENDC}\n")
                break

            elif selected == "Start new run":
                run_id = self.generate_run_id()
                logger.info(f"Starting new run: {run_id}")
                self.exec_state["last_run_id"] = run_id
                self.exec_state["last_step"] = None
                self.exec_state["runs"][run_id] = {
                    "created_at": datetime.now().isoformat(),
                    "steps": {},
                }
                self.save_exec_state()
                self.print_success(f"New run started: {run_id}")

                step = self.PIPELINE_STEPS[0]
                params = self.collect_parameters(step, run_id)
                self.execute_step(step, params, run_id)

            elif selected.startswith("Continue to next step"):
                run_id = self.exec_state["last_run_id"]
                last_step = self.exec_state.get("last_step")

                if last_step is None:
                    next_step_id = 1
                else:
                    next_step_id = last_step + 1

                step = self.get_step_by_id(next_step_id)
                if not step:
                    self.print_error(f"Step {next_step_id} not found")
                    continue

                params = self.collect_parameters(step, run_id)
                self.execute_step(step, params, run_id)

            elif selected.startswith("Rerun last step"):
                run_id = self.exec_state["last_run_id"]
                last_step = self.exec_state.get("last_step")

                step = self.get_step_by_id(last_step)
                if not step:
                    self.print_error(f"Step {last_step} not found")
                    continue

                params = self.collect_parameters(step, run_id)
                self.execute_step(step, params, run_id)

            elif selected == "View run history":
                self.view_run_history()

            elif selected == "Visualize phrase fingerprints":
                self.run_phrase_visualization()

            elif selected == "Clear history":
                self.clear_history()

        # ------------------------------------------------------------------
        # Run history
        # ------------------------------------------------------------------

        def view_run_history(self):
            if not self.exec_state["runs"]:
                self.print_warning("No run history available")
                return

            self.print_header("Run History")

            for run_id, run_data in self.exec_state["runs"].items():
                print(f"\n{Colors.BOLD}Run ID:{Colors.ENDC} {run_id}")
                print(f"{Colors.CYAN}Created:{Colors.ENDC} {run_data.get('created_at', 'N/A')}")
                print(f"{Colors.CYAN}Steps completed:{Colors.ENDC}")

                for step_id, step_data in sorted(run_data.get("steps", {}).items()):
                    print(
                        f"  {Colors.GREEN}✓{Colors.ENDC} Step {step_id}: "
                        f"{step_data['name']} ({step_data.get('completed_at', 'N/A')})"
                    )
                    print(f"    Output: {step_data.get('output', 'N/A')}")

            input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")

    def clear_history(self):
        """Clear execution history and optionally remove output folders."""
        self.print_header("Clear History")
        
        if not self.exec_state.get("runs"):
            self.print_warning("No run history to clear.")
            return
        
        # Show what will be cleared
        print(f"\n{Colors.YELLOW}This will clear:{Colors.ENDC}")
        print(f"  • Execution state file: {self.exec_state_path}")
        print(f"  • Run history: {len(self.exec_state['runs'])} run(s)")
        
        # Collect all output folders from runs
        tracked_folders = set()
        for run_id, run_data in self.exec_state["runs"].items():
            for step_id, step_data in run_data.get("steps", {}).items():
                if "output" in step_data:
                    output_path = Path(step_data["output"])
                    if output_path.exists():
                        tracked_folders.add(output_path)
        
        if tracked_folders:
            print(f"  • {len(tracked_folders)} tracked output folder(s)")
        
        # Confirm
        confirm = input(f"\n{Colors.RED}Are you sure? (yes/no): {Colors.ENDC}").strip().lower()
        if confirm not in ["yes", "y"]:
            print(f"{Colors.YELLOW}Cancelled.{Colors.ENDC}")
            return
        
        # Clear state file
        self.exec_state = {"runs": {}, "last_run_id": None, "last_step": None}
        self.save_exec_state()
        logger.info("Execution state cleared")
        self.print_success("Execution state cleared")
        
        # Remove tracked output folders
        removed_count = 0
        for folder in tracked_folders:
            try:
                if folder.exists():
                    shutil.rmtree(folder)
                    removed_count += 1
                    logger.info(f"Removed output folder: {folder}")
            except Exception as e:
                self.print_error(f"Failed to remove {folder}: {e}")
        
        if removed_count > 0:
            self.print_success(f"Removed {removed_count} tracked output folder(s)")
        
        # Check for remaining folders in outputs directory
        outputs_dir = Path(self.config.get("paths", {}).get("outputs", "outputs"))
        if outputs_dir.exists():
            remaining_folders = [
                d for d in outputs_dir.iterdir() 
                if d.is_dir() and d not in tracked_folders
            ]
            
            if remaining_folders:
                print(f"\n{Colors.YELLOW}Found {len(remaining_folders)} untracked folder(s) in {outputs_dir}:{Colors.ENDC}")
                for folder in remaining_folders:
                    print(f"  • {folder.name}")
                
                remove_all = input(f"\n{Colors.YELLOW}Remove these folders too? (yes/no): {Colors.ENDC}").strip().lower()
                if remove_all in ["yes", "y"]:
                    removed_untracked = 0
                    for folder in remaining_folders:
                        try:
                            shutil.rmtree(folder)
                            removed_untracked += 1
                            logger.info(f"Removed untracked folder: {folder}")
                        except Exception as e:
                            self.print_error(f"Failed to remove {folder}: {e}")
                    
                    if removed_untracked > 0:
                        self.print_success(f"Removed {removed_untracked} untracked folder(s)")
                else:
                    print(f"{Colors.YELLOW}Kept untracked folders.{Colors.ENDC}")
        
        print(f"\n{Colors.GREEN}History cleared successfully!{Colors.ENDC}")



def main():
    try:
        runner = SemanticRunner()
        runner.run()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.ENDC}")
        logger.warning("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}Fatal error: {e}{Colors.ENDC}")
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

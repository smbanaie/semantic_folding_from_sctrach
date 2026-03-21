#!/usr/bin/env python3
"""
Semantic Folding Pipeline Runner
Interactive TUI for executing the semantic folding pipeline with state management.
"""

import os
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
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class SemanticRunner:
    """Interactive runner for semantic folding pipeline"""
    # Fall back to config values for non-path parameters
    CONFIG_PATH_IN_YAML = {
            # Global
            "grid_size":            ["grid_size"],                          # ← top-level, single key

            # Phase 1
            "min_freq":             ["phrase_extraction", "min_freq"],
            "min_word_length":      ["phrase_extraction", "min_word_length"],
            "use_spacy":            ["phrase_extraction", "use_spacy"],
            "max_ngram":            ["phrase_extraction", "max_ngram"],
            "filter_generic":       ["phrase_extraction", "filter_generic"],
            "stats":                ["phrase_extraction", "stats"],

            # Phase 2
            "use_tfidf":            ["term_context_matrix", "use_tfidf"],
            "min_phrase_freq":      ["term_context_matrix", "min_phrase_freq"],
            "use_word_boundaries":  ["term_context_matrix", "use_word_boundaries"],
            "keep_verbs":           ["term_context_matrix", "keep_verbs"],

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
            "min_freq":             ["document_fingerprints", "min_freq"],
            "normalize":            ["document_fingerprints", "normalize"],
            "normalize_method":     ["document_fingerprints", "normalize_method"],

            # Query
            "weighting":            ["query_processing", "weighting"],
            "spreading_radius":     ["query_processing", "spreading", "radius"],
            "spreading_decay":      ["query_processing", "spreading", "decay"],
            "top_k":                ["query_processing", "top_k"],
        }

    PIPELINE_STEPS = [
    {
        "id": 1,
        "name": "phrase_extraction",
        "script": "brain_approaches/semantic_folding/phrase_extractor.py",
        "required_params": ["corpus", "output"],
        "optional_params": ["min_freq", "min_word_length", "use_spacy", "filter_generic", "stats"],
        "default_output": "phrases.txt",
        "depends_on": []
    },
        {
            "id": 2,
            "name": "Term-Context Matrix",
            "script": "brain_approaches/semantic_folding/term_context.py",
            "required_params": ["corpus", "phrases", "output"],
            "optional_params": ["use_tfidf", "min_phrase_freq", "use_word_boundaries", "keep_verbs"],
            "default_output": "term_context_matrix.npz",
            "extra_outputs": {
                  "metadata": lambda output: str(Path(output).with_suffix('.json'))
            },
            "depends_on": [1]
        },
        {
            "id": 3,
            "name": "Semantic Space",
            "script": "brain_approaches/semantic_folding/semantic_space.py",
            "required_params": ["matrix", "metadata", "output"],
            "optional_params": ["method", "grid_size", "visualize", "show_density"],
            "default_output": "semantic_space",
            "depends_on": [2]
        },
        {
            "id": 4,
            "name": "Phrase Fingerprints",
            "script": "brain_approaches/semantic_folding/phrase_fingerprints.py",
            "required_params": ["coordinates","metadata", "output"],
            "optional_params": ["grid_size"],
            "default_output": "phrase_fingerprints",
            "depends_on": [3]
        },
        {
            "id": 5,
            "name": "Document Fingerprints",
            "script": "brain_approaches/semantic_folding/doc_fingerprints.py",
            "required_params": ["corpus", "phrases", "fingerprints", "output"],
            "optional_params": ["top_percent", "min_freq", "normalize", "normalize_method"],
            "default_output": "doc_fingerprints",
            "depends_on": [4]
        },
        {
            "id": 6,
            "name": "Query Processing",
            "script": "brain_approaches/semantic_folding/query_processing.py",
            "required_params": ["query", "phrases", "fingerprints", "doc_fps"],
            "optional_params": ["corpus", "weighting", "spreading_radius", "spreading_decay", "top_k", "output_json"],
            "default_output": "query_results.json",
            "depends_on": [5]
        }
    ]

    def __init__(self):
        self.config_dir = Path("config")
        self.exec_state_path = self.config_dir / "exec_state.yml"
        self.config_path = self.config_dir / "semantic_folding.yml"

        self.config_dir.mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        logger.info("Initializing SemanticRunner")
        self.config = self.load_config()
        self.exec_state = self.load_exec_state()
        logger.debug(f"Config loaded from: {self.config_path}")
        logger.debug(f"Exec state loaded from: {self.exec_state_path}")

    def load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}, using empty config")
            return {}

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded config with {len(config)} top-level keys")
        return config

    def load_exec_state(self) -> Dict[str, Any]:
        if not self.exec_state_path.exists():
            logger.info("No existing exec state found, starting fresh")
            return {"last_run_id": None, "last_step": None, "runs": {}}

        with open(self.exec_state_path, 'r') as f:
            state = yaml.safe_load(f) or {"last_run_id": None, "last_step": None, "runs": {}}

        run_count = len(state.get("runs", {}))
        logger.info(f"Loaded exec state: {run_count} previous run(s) found")
        return state

    def save_exec_state(self):
        with open(self.exec_state_path, 'w') as f:
            yaml.dump(self.exec_state, f, default_flow_style=False, sort_keys=False)
        logger.debug(f"Exec state saved to {self.exec_state_path}")

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
        if default is not None:                          # ← was: if default:
            prompt = f"{prompt} [{Colors.YELLOW}{default}{Colors.ENDC}]: "
        else:
            prompt = f"{prompt}: "
        value = input(prompt).strip()
        return value if value else (default if default is not None else "")  # ← was: (default or "")


    def get_choice(self, prompt: str, options: List[str]) -> int:
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"  {Colors.BOLD}{i}.{Colors.ENDC} {option}")

        while True:
            try:
                choice = int(input(f"\n{Colors.BOLD}Enter choice (1-{len(options)}): {Colors.ENDC}"))
                if 1 <= choice <= len(options):
                    return choice
                print(f"{Colors.RED}Invalid choice. Please enter a number between 1 and {len(options)}.{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.RED}Invalid input. Please enter a number.{Colors.ENDC}")

    def generate_run_id(self) -> str:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.debug(f"Generated new run ID: {run_id}")
        return run_id

    def get_step_by_id(self, step_id: int) -> Optional[Dict[str, Any]]:
        for step in self.PIPELINE_STEPS:
            if step["id"] == step_id:
                return step
        logger.warning(f"Step ID {step_id} not found in pipeline definition")
        return None

    def get_default_value(self, param: str, step_id: int) -> Optional[str]:
        """
        Resolve the default value for a pipeline parameter.

        Resolution priority (first match wins):
        1. Output path from a *previous* step in the last run
            - Checks `step_data["output"]` against known filename patterns
            - Checks `step_data["extra_outputs"]` by label name
        2. Parameter value stored from the *current* step in the last run
            - Looks up `run_data["steps"][step_id]["parameters"][param]`
        3. Special case: reconstruct `coordinates` path from step 3 output dir
            - Builds candidate path and verifies it exists on disk
        4. YAML config fallback via CLASS-LEVEL `CONFIG_PATH_IN_YAML` mapping
            - Traverses nested keys defined in the mapping
            - Returns `str(value)` for prompt display
        5. Returns `None` if no default could be resolved

        Args:
            param:   The parameter name to resolve (e.g. "show_density", "grid_size").
            step_id: The current step being configured (used to scope previous-step lookup).

        Returns:
            The resolved default as a string (for prompt display), or None if unresolved.

        Notes:
            - Boolean YAML values (false/true) are returned as "False"/"True" strings.
            - A sentinel object is used internally to distinguish a missing key from
            a key whose value is legitimately False, 0, or an empty string.
            - All resolution attempts are logged at DEBUG level for traceability.
        """
        _MISSING = object()  # Sentinel: distinguishes "key absent" from False / 0 / "" / None
        # Temporary diagnostic - remove after fix
        # ------------------------------------------------------------------ #
        # Priority 1 & 2 & 3 — Previous run state                            #
        # ------------------------------------------------------------------ #
        if self.exec_state.get("last_run_id"):
            run_id   = self.exec_state["last_run_id"]
            run_data = self.exec_state["runs"].get(run_id)

            if run_data:
                param_mapping = {
                    "phrases":    "phrases.txt",
                    "matrix":     "term_context_matrix.npz",
                    "coordinates":"context_coordinates.json"
                }

                # --- Priority 1: Output / extra_outputs from earlier steps --- #
                for prev_step_id in range(1, step_id):
                    step_data = run_data["steps"].get(prev_step_id)
                    if step_data is None:
                        continue

                    # 1a. Primary output path — match by expected filename
                    if "output" in step_data and param in param_mapping:
                        output_path = step_data["output"]
                        if param_mapping[param] in output_path:
                            logger.debug(
                                f"Default for '{param}' resolved from step {prev_step_id} "
                                f"output: {output_path}"
                            )
                            return output_path

                    # 1b. Named extra outputs — match by label == param
                    extra = step_data.get("extra_outputs", {})
                    if param in extra:
                        logger.debug(
                            f"Default for '{param}' resolved from step {prev_step_id} "
                            f"extra_outputs['{param}']: {extra[param]}"
                        )
                        return extra[param]

                # --- Priority 2: Same-step parameters from the last run --- #
                step_params = (
                    run_data["steps"]
                    .get(step_id, {})
                    .get("parameters", {})
                )
                if param in step_params:
                    logger.debug(
                        f"Default for '{param}' resolved from previous run "
                        f"step {step_id} parameters: {step_params[param]}"
                    )
                    return step_params[param]

                # --- Priority 3: Reconstruct coordinates path from step 3 --- #
                if param == "coordinates":
                    step3_data = run_data["steps"].get(3, {})
                    step3_output = step3_data.get("output", "")
                    if step3_output:
                        candidate = str(Path(step3_output) / "context_coordinates.json")
                        if Path(candidate).exists():
                            logger.debug(
                                f"Default for 'coordinates' reconstructed from "
                                f"step 3 output dir: {candidate}"
                            )
                            return candidate
                        else:
                            logger.debug(
                                f"Default for 'coordinates': candidate path does not exist "
                                f"— {candidate}"
                            )

                # --- Priority 3b: fingerprints directory is Step 4's output dir --- #
                if param == "fingerprints":
                    step4_data   = run_data["steps"].get(4, {})
                    step4_output = step4_data.get("output", "")
                    if step4_output and Path(step4_output).exists():
                        logger.debug(
                            f"Default for 'fingerprints' resolved from step 4 "
                            f"output dir: {step4_output}"
                        )
                        return step4_output
                    else:
                        logger.debug(
                            f"Default for 'fingerprints': step 4 output missing or "
                            f"does not exist on disk — {step4_output!r}"
                        )            

                # --- Priority 3c: doc_fps = Step 5 output directory --- #
                if param == "doc_fingerprints":
                    step5_data   = run_data["steps"].get(5, {})
                    step5_output = step5_data.get("output", "")
                    if step5_output and Path(step5_output).exists():
                        logger.debug(
                            f"Default for 'doc_fingerprints' resolved from step 5 "
                            f"output dir: {step5_output}"
                        )
                        return step5_output
                    else:
                        logger.debug(
                            f"Default for 'doc_fingerprints': step 5 output missing or "
                            f"does not exist on disk — {step5_output!r}"
                        )            
        # ------------------------------------------------------------------ #
        # Priority 4 — YAML config fallback via CONFIG_PATH_IN_YAML           #
        # ------------------------------------------------------------------ #
        if param not in self.CONFIG_PATH_IN_YAML:
            logger.debug(
                f"Default for '{param}': not in CONFIG_PATH_IN_YAML and "
                f"no prior run match — returning None"
            )
            return None

        if not self.config:
            logger.debug(
                f"Default for '{param}': CONFIG_PATH_IN_YAML match found but "
                f"self.config is not loaded — returning None"
            )
            return None

        path  = self.CONFIG_PATH_IN_YAML[param]
        value = self.config  # Root of the loaded YAML dict

        for key in path:
            if not isinstance(value, dict):
                logger.debug(
                    f"Default for '{param}': YAML traversal failed at key '{key}' — "
                    f"current node is {type(value).__name__}, not a dict"
                )
                return None

            value = value.get(key, _MISSING)

            if value is _MISSING:
                logger.debug(
                    f"Default for '{param}': key '{key}' not found in YAML "
                    f"at path {path}"
                )
                return None

        # value is now the resolved leaf — could be False, 0, "", etc.
        if value is None:
            logger.debug(
                f"Default for '{param}': YAML path {path} resolved to explicit None"
            )
            return None

        resolved = str(value)
        logger.debug(f"Default for '{param}' resolved from YAML config: {resolved}")
        return resolved

    def get_output_path(self, step: Dict[str, Any], run_id: str) -> str:
        """Generate output path within run_id directory"""
        output_base = self.config.get("paths", {}).get("output_base", "outputs")
        output_dir = Path(output_base) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / step["default_output"])
        logger.debug(f"Output path for step {step['id']}: {output_path}")
        return output_path

    def collect_parameters(self, step: Dict[str, Any], run_id: str) -> Dict[str, str]:
        logger.info(f"Collecting parameters for step {step['id']}: {step['name']}")
        params = {}

        print(f"\n{Colors.BOLD}Configure parameters for: {step['name']}{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")

        for param in step["required_params"]:
            default = self.get_default_value(param, step["id"])
            if param == "corpus":
                default = default or self.config.get("paths", {}).get("corpus_path", "data/corpus.txt")
            elif param == "output":
                default = self.get_output_path(step, run_id)

            value = self.get_input(f"{Colors.BOLD}{param}{Colors.ENDC} (required)", default)
            while not value:
                self.print_error(f"Parameter '{param}' is required")
                value = self.get_input(f"{Colors.BOLD}{param}{Colors.ENDC} (required)", default)

            params[param] = value
            logger.debug(f"Required param set — {param}: {value}")

        print(f"\n{Colors.CYAN}Optional parameters (press Enter to skip):{Colors.ENDC}")
        for param in step["optional_params"]:
            if param == "output":
                default = self.get_output_path(step, run_id)
            else:
                default = self.get_default_value(param, step["id"])

            value = self.get_input(f"  {param}", default)

            # Only store if user entered something OR a default exists
            if value:
                if isinstance(value, str) and value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                params[param] = value
                logger.debug(f"Optional param set — {param}: {value}")
            elif default is not None and value == default:      # ← user pressed Enter on a default
                if isinstance(default, str) and default.lower() in ('true', 'false'):
                    params[param] = default.lower() == 'true'
                else:
                    params[param] = default
                logger.debug(f"Optional param kept from default — {param}: {params[param]}")

        logger.info(f"Parameter collection complete for step {step['id']}: {len(params)} params collected")
        return params

    def build_command(self, step: Dict, params: Dict) -> List[str]:
        cmd = ["E:\\PHD\\GraphRag-Implementations\\YaALI\\knowledge-graph-builder\\.venv\\scripts\\python", step["script"]]
        
        # Map positive config names to negative CLI flags
        negate_renamed_no_values_map = {
            'use_spacy': 'no-spacy',
            'filter_generic': 'no-filter-generic',
            'use_tfidf': 'no-tfidf',
            'use_word_boundaries': 'no-word-boundaries',
            'enable_grid' : 'no-grid'
        }
        renamed_has_value_parameters_map = {
            'min_phrase_freq':'min_freq'
        }
        
        for key, value in params.items():
            if isinstance(value, bool):
                # Check if this is a positive flag that maps to a negative CLI flag
                if key in negate_renamed_no_values_map:
                    # Add the negative flag when value is False
                    if not value:
                        cmd.append(f"--{negate_renamed_no_values_map[key]}")
                else:
                    # Normal flag: add when True
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
            else:
                if key in renamed_has_value_parameters_map :
                    cmd.extend([f"--{renamed_has_value_parameters_map[key].replace('_', '-')}", str(value)])
                else:
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        logger.debug(f"Built command: {' '.join(cmd)}")
        return cmd

    def execute_step(self, step: Dict[str, Any], params: Dict[str, str], run_id: str) -> bool:
        cmd = self.build_command(step, params)

        logger.info(f"Executing step {step['id']}: {step['name']} | run_id={run_id}")
        logger.debug(f"Full command: {' '.join(cmd)}")

        print(f"\n{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Executing:{Colors.ENDC} {' '.join(cmd)}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}\n")

        try:
            result = subprocess.run(cmd, check=True, text=True)

            if run_id not in self.exec_state["runs"]:
                self.exec_state["runs"][run_id] = {
                    "created_at": datetime.now().isoformat(),
                    "steps": {}
                }

            output_path = params.get("output", self.get_output_path(step, run_id))

            # Build extra outputs map if defined
            extra_outputs = {}
            for label, path_fn in step.get("extra_outputs", {}).items():
                extra_outputs[label] = path_fn(output_path)
                logger.debug(f"Extra output tracked — {label}: {extra_outputs[label]}")

            step_record = {
                "name": step["name"],
                "completed_at": datetime.now().isoformat(),
                "parameters": params,
                "output": output_path,
            }

            if extra_outputs:
                step_record["extra_outputs"] = extra_outputs

            self.exec_state["runs"][run_id]["steps"][step["id"]] = step_record

            self.exec_state["last_run_id"] = run_id
            self.exec_state["last_step"] = step["id"]
            self.save_exec_state()

            logger.info(f"Step {step['id']} completed successfully — output: {output_path}")
            if extra_outputs:
                for label, path in extra_outputs.items():
                    logger.info(f"  Extra output — {label}: {path}")

            self.print_success(f"Step {step['id']}: {step['name']} completed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Step {step['id']} failed with return code {e.returncode}")
            self.print_error(f"Step {step['id']}: {step['name']} failed with error code {e.returncode}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error during step {step['id']}: {e}")
            self.print_error(f"Unexpected error: {str(e)}")
            return False


    def show_last_run_info(self):
        if not self.exec_state["last_run_id"]:
            return

        run_id = self.exec_state["last_run_id"]
        last_step_id = self.exec_state["last_step"]

        if run_id not in self.exec_state["runs"]:
            return

        run_data = self.exec_state["runs"][run_id]
        last_step = self.get_step_by_id(last_step_id)

        if not last_step:
            return

        logger.debug(f"Displaying last run info: run_id={run_id}, last_step={last_step_id}")

        print(f"\n{Colors.BOLD}Last Run Information:{Colors.ENDC}")
        print(f"  Run ID: {Colors.YELLOW}{run_id}{Colors.ENDC}")
        print(f"  Last Step: {Colors.YELLOW}{last_step['name']} (Step {last_step_id}){Colors.ENDC}")

        if last_step_id in run_data["steps"]:
            step_data = run_data["steps"][last_step_id]
            print(f"  Completed: {Colors.GREEN}{step_data['completed_at']}{Colors.ENDC}")
            print(f"  Output: {Colors.CYAN}{step_data['output']}{Colors.ENDC}")

    def rerun_last_step(self) -> None:
        """Rerun the last completed step with option to modify parameters"""
        if not self.exec_state["last_run_id"]:
            self.print_error("No previous run found")
            return

        run_id = self.exec_state["last_run_id"]
        run_data = self.exec_state["runs"][run_id]

        if not run_data["steps"]:
            self.print_error("No completed steps found in last run")
            return

        last_step_id = max(run_data["steps"].keys())
        step_data = run_data["steps"][last_step_id]
        step_name = step_data["name"]

        # Use PIPELINE_STEPS, not self.config
        step_def = self.get_step_by_id(last_step_id)

        if step_def is None:
            self.print_error(f"Step definition for '{step_name}' not found in pipeline")
            logger.error(f"Could not find step definition for step_id={last_step_id}, name={step_name}")
            return

        print(f"\n{Colors.HEADER}Rerunning Step {last_step_id}: {step_name}{Colors.ENDC}")
        print(f"Run ID: {run_id}\n")

        prev_params = step_data.get("parameters", {})
        print(f"{Colors.CYAN}Previous parameters:{Colors.ENDC}")
        for key, value in prev_params.items():
            print(f"  {key}: {value}")

        modify = self.get_input("\nModify parameters? (y/n)", "n").lower()

        if modify == 'y':
            logger.info("User chose to modify parameters for rerun")
            params = self.collect_parameters(step_def, run_id)
        else:
            logger.info("User chose to keep previous parameters for rerun")
            params = prev_params

        print(f"\n{Colors.CYAN}Parameters for rerun:{Colors.ENDC}")
        for key, value in params.items():
            print(f"  {key}: {value}")

        confirm = self.get_input(f"\nProceed with rerun? (y/n)", "y").lower()

        if confirm != 'y':
            logger.info("Rerun cancelled by user")
            print(f"{Colors.YELLOW}Rerun cancelled{Colors.ENDC}")
            return

        logger.info(f"Executing rerun of step {last_step_id} with run_id {run_id}")
        success = self.execute_step(step_def, params, run_id)

        if not success:
            print(f"\n{Colors.RED}✗ Step {last_step_id}: {step_name} rerun failed{Colors.ENDC}\n")
            logger.error(f"Step {last_step_id} rerun failed")

    def continue_run(self):
        logger.info("User selected: continue pipeline")

        if not self.exec_state["last_run_id"] or not self.exec_state["last_step"]:
            logger.warning("Continue requested but no previous run found, redirecting to new run")
            self.print_error("No previous run found. Starting new run instead.")
            return self.start_new_run()

        run_id = self.exec_state["last_run_id"]
        last_step_id = self.exec_state["last_step"]

        if last_step_id >= len(self.PIPELINE_STEPS):
            logger.info("All pipeline steps completed")
            self.print_success("Pipeline completed! All steps have been executed.")
            return True

        next_step = self.PIPELINE_STEPS[last_step_id]
        logger.info(f"Continuing run {run_id} with step {next_step['id']}: {next_step['name']}")

        print(f"\n{Colors.BOLD}Continuing run: {Colors.YELLOW}{run_id}{Colors.ENDC}")
        print(f"Next step: {Colors.YELLOW}Step {next_step['id']}: {next_step['name']}{Colors.ENDC}")

        params = self.collect_parameters(next_step, run_id)
        return self.execute_step(next_step, params, run_id)

    def show_run_history(self):
        logger.info("Displaying run history")

        if not self.exec_state["runs"]:
            logger.info("No run history available")
            print(f"\n{Colors.YELLOW}No run history available{Colors.ENDC}")
            return

        print(f"\n{Colors.BOLD}Run History:{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")

        for run_id, run_data in sorted(self.exec_state["runs"].items(), reverse=True):
            print(f"\n{Colors.BOLD}Run ID: {Colors.YELLOW}{run_id}{Colors.ENDC}")
            print(f"Created: {run_data['created_at']}")
            print(f"Steps completed: {len(run_data['steps'])}")

            for step_id in sorted(run_data["steps"].keys()):
                step_data = run_data["steps"][step_id]
                print(f"  {Colors.GREEN}✓{Colors.ENDC} Step {step_id}: {step_data['name']}")
                print(f"    Output: {Colors.CYAN}{step_data['output']}{Colors.ENDC}")

        print(f"\n{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")

    def start_new_run(self) -> bool:
        run_id = self.generate_run_id()
        logger.info(f"Starting new run: {run_id}")

        self.print_header(f"New Run: {run_id}")

        # Let user choose starting step
        print(f"{Colors.BOLD}Select starting step:{Colors.ENDC}")
        step_options = [f"Step {s['id']}: {s['name']}" for s in self.PIPELINE_STEPS]
        start_choice = self.get_choice("Start from which step?", step_options)
        start_step = self.PIPELINE_STEPS[start_choice - 1]

        logger.info(f"User selected starting step {start_step['id']}: {start_step['name']}")

        # Initialize run in exec_state
        self.exec_state["runs"][run_id] = {
            "created_at": datetime.now().isoformat(),
            "steps": {}
        }
        self.save_exec_state()

        # Execute from selected step onward
        current_index = start_choice - 1
        while current_index < len(self.PIPELINE_STEPS):
            step = self.PIPELINE_STEPS[current_index]

            print(f"\n{Colors.BOLD}{Colors.HEADER}Step {step['id']}: {step['name']}{Colors.ENDC}")

            params = self.collect_parameters(step, run_id)

            print(f"\n{Colors.CYAN}Parameters summary:{Colors.ENDC}")
            for key, value in params.items():
                print(f"  {key}: {value}")

            confirm = self.get_input(f"\nExecute step {step['id']}? (y/n)", "y").lower()
            if confirm != 'y':
                logger.info(f"User skipped step {step['id']}")
                self.print_warning(f"Step {step['id']} skipped")
                break

            success = self.execute_step(step, params, run_id)

            if not success:
                logger.error(f"Step {step['id']} failed, stopping run {run_id}")
                self.print_error(f"Run stopped at step {step['id']}")
                return False

            current_index += 1

            # If there are more steps, ask to continue
            if current_index < len(self.PIPELINE_STEPS):
                next_step = self.PIPELINE_STEPS[current_index]
                cont = self.get_input(
                    f"\nContinue to Step {next_step['id']}: {next_step['name']}? (y/n)",
                    "y"
                ).lower()
                if cont != 'y':
                    logger.info(f"User chose to stop after step {step['id']}")
                    self.print_warning("Run paused. You can continue later from the main menu.")
                    return True
            else:
                self.print_success("All pipeline steps completed successfully!")
                logger.info(f"Run {run_id} completed all steps")

        return True

    def run(self):
        logger.info("SemanticRunner started")
        self.print_header("Semantic Folding Pipeline Runner")

        while True:
            self.show_last_run_info()

            options = []
            if self.exec_state["last_run_id"]:
                last_step = self.get_step_by_id(self.exec_state["last_step"])
                options.append(f"Rerun last step (Step {self.exec_state['last_step']}: {last_step['name']})")
                if self.exec_state["last_step"] < len(self.PIPELINE_STEPS):
                    options.append("Continue to next step")

            options.extend(["Start new run", "View run history", "Exit"])

            choice = self.get_choice("What would you like to do?", options)
            logger.debug(f"User menu choice: {choice}")

            offset = 0
            if self.exec_state["last_run_id"]:
                if choice == 1:
                    self.rerun_last_step()
                    continue
                if self.exec_state["last_step"] < len(self.PIPELINE_STEPS):
                    if choice == 2:
                        self.continue_run()
                        continue
                    offset = 1

            adjusted_choice = choice - offset - (1 if self.exec_state["last_run_id"] else 0)

            if adjusted_choice == 1:
                self.start_new_run()
            elif adjusted_choice == 2:
                self.show_run_history()
            elif adjusted_choice == 3:
                logger.info("User exited the runner")
                print(f"\n{Colors.GREEN}Goodbye!{Colors.ENDC}\n")
                break


def main():
    try:
        runner = SemanticRunner()
        runner.run()
    except KeyboardInterrupt:
        logger.warning("Runner interrupted by user (KeyboardInterrupt)")
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.ENDC}\n")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal error in SemanticRunner: {e}")
        print(f"\n{Colors.RED}Fatal error: {str(e)}{Colors.ENDC}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

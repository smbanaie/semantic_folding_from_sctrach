"""I/O utilities for saving agent outputs for debugging and inspection.

Writes JSON files to `src/agents/output/` with a timestamped filename.
"""
from __future__ import annotations

import json
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any


def _make_serializable(obj: Any) -> Any:
    """Recursively convert common objects to JSON-serializable types."""
    # Pydantic BaseModel has .dict()
    try:
        from pydantic import BaseModel

        if isinstance(obj, BaseModel):
            return _make_serializable(obj.dict())
    except Exception:
        pass

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_make_serializable(v) for v in obj)
    # Fallbacks for common objects
    if hasattr(obj, "__dict__"):
        try:
            return _make_serializable(vars(obj))
        except Exception:
            pass
    # If it's already JSON serializable (str, int, float, bool, None)
    return obj


_RUN_ID: str | None = None


def set_run_id(run_id: str | None = None) -> str:
    """Set a run id to group agent outputs. Returns the run id."""
    global _RUN_ID
    if run_id is None:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    _RUN_ID = run_id
    return _RUN_ID


def get_run_id() -> str:
    """Get current run id, auto-generate if unset."""
    global _RUN_ID
    if _RUN_ID is None:
        set_run_id()
    return _RUN_ID  # type: ignore[return-value]


def save_agent_output(agent_name: str, data: Any, *, filename: str | None = None) -> Path:
    """Save `data` to `src/agents/output/{run_id}/{agent_name}_{timestamp}.json`.

    If `set_run_id` wasn't called, a run id will be auto-generated.
    Returns the path written.
    """
    base = Path(__file__).resolve().parents[1]  # src/
    run_id = get_run_id()
    out_dir = base / "agents" / "output" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = filename or f"{agent_name}_{ts}.json"
    path = out_dir / fname

    serializable = _make_serializable(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    return path

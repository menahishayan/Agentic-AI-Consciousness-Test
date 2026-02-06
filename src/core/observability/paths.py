from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.observability.config import LoggingConfig


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


@dataclass
class RunPaths:
    run_dir: Path
    run_json: Path
    events_jsonl: Path
    llm_jsonl: Path
    memory_jsonl: Path
    state_jsonl: Path
    tracebacks_log: Path
    tracebacks_jsonl: Path


def create_run_dir(config: LoggingConfig) -> RunPaths:
    root = config.log_root
    root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_{config.run_id}"
    if config.run_name:
        base_name = f"{base_name}_{_sanitize(config.run_name)}"

    run_dir = root / base_name
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{base_name}_{suffix}"

    run_dir.mkdir(parents=True, exist_ok=True)

    run_json = run_dir / "run.json"
    events_jsonl = run_dir / "events.jsonl"
    llm_jsonl = run_dir / "llm.jsonl"
    memory_jsonl = run_dir / "memory.jsonl"
    state_jsonl = run_dir / "state.jsonl"
    tracebacks_log = run_dir / "tracebacks.log"
    tracebacks_jsonl = run_dir / "tracebacks.jsonl"

    for path in (
        events_jsonl,
        llm_jsonl,
        memory_jsonl,
        state_jsonl,
        tracebacks_log,
        tracebacks_jsonl,
    ):
        path.touch(exist_ok=True)

    return RunPaths(
        run_dir=run_dir,
        run_json=run_json,
        events_jsonl=events_jsonl,
        llm_jsonl=llm_jsonl,
        memory_jsonl=memory_jsonl,
        state_jsonl=state_jsonl,
        tracebacks_log=tracebacks_log,
        tracebacks_jsonl=tracebacks_jsonl,
    )

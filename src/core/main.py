"""
Entry point for the brain-inspired multi-agent system.

Usage:
    cd capstone/
    source venv/bin/activate
    python -m src.core.main

Configuration:
    config.json   — all tunable parameters
    .env          — API keys (ANTHROPIC_API_KEY, etc.)

To swap environments: change "adapter_folder" in config.json.
To swap LLM providers: change "llm.provider" in config.json.
The brain code is untouched in both cases.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import coloredlogs

# Ensure src/ is on the Python path when running as a module
_SRC = Path(__file__).resolve().parent.parent.parent
_SRC_CORE = _SRC / "src"
for _p in [str(_SRC_CORE), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: Path) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "adapter_folder": "animalai",
        "adapter_config": {},
        "llm": {"enabled": True, "provider": "anthropic", "model": "claude-sonnet-4-6"},
        "memory": {"working_memory_capacity": 100, "faiss_k_default": 5},
        "observability": {"log_root": "src/logs/runs", "log_state": True, "log_prompts": False, "log_memory": True},
    }
    if not path.exists():
        return defaults
    user = json.loads(path.read_text())
    return _deep_merge(defaults, user)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent

    # Load environment variables
    _load_env(project_root / ".env")

    # Configure logging
    coloredlogs.install(
        level=os.getenv("LOG_LEVEL", "INFO"),
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("main")

    # Load config
    config = _load_config(project_root / "config.json")
    max_steps = int(os.getenv("MAX_STEPS", "500"))

    obs_cfg = config.get("observability", {})
    log_state = _env_bool("LOG_STATE", bool(obs_cfg.get("log_state", True)))
    log_prompts = _env_bool("LOG_PROMPTS", bool(obs_cfg.get("log_prompts", False)))
    log_memory = _env_bool("LOG_MEMORY", bool(obs_cfg.get("log_memory", True)))
    log_root = obs_cfg.get("log_root", "src/logs/runs")

    logger.info("Starting agent (adapter=%s, max_steps=%d)", config.get("adapter_folder"), max_steps)

    # Build run directory and structured logger
    from core.observability.paths import make_run_dir
    run_dir = make_run_dir(log_root)
    logger.info("Log dir: %s", run_dir)

    from core.observability.logger import RunLogger
    from core.observability.exceptions import install_exception_hooks

    run_logger = RunLogger(
        run_dir=run_dir,
        config=config,
        log_state=log_state,
        log_prompts=log_prompts,
        log_memory=log_memory,
    )

    install_exception_hooks(on_exception=lambda et, ev, tb: run_logger.traceback(ev, "unhandled"))

    # Build adapter
    from core.adapters.loader import build_adapter
    adapter = build_adapter(config["adapter_folder"], config.get("adapter_config", {}))

    # Build LLM client
    from core.llm.factory import build_llm_client
    llm_client = build_llm_client(config.get("llm"), logger=run_logger)
    if llm_client:
        logger.info("LLM provider: %s", llm_client.provider_name)
    else:
        logger.info("LLM disabled")

    # Build memory manager
    from core.memory.manager import MemoryManager
    memory = MemoryManager(config)

    # Build and run agent loop
    from core.runtime.loop import AgentLoop
    loop = AgentLoop(
        adapter=adapter,
        memory=memory,
        llm_client=llm_client,
        logger=run_logger,
        config=config,
    )

    try:
        stats = loop.run(max_steps=max_steps)
        logger.info("Run complete: %s", stats)
    finally:
        adapter.close()
        run_logger.close()
        logger.info("Shutdown complete. Logs at: %s", run_dir)


if __name__ == "__main__":
    main()

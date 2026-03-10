from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

from core.adapters.loader import build_adapter
from core.llm.factory import build_llm_client
from core.memory.manager import MemoryManager
from core.observability import LoggingConfig, RunLogger, install_exception_hooks
from core.runtime.loop import AgentLoop


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> Dict[str, Any]:
    default: Dict[str, Any] = {
        "adapter_folder": "minedojo",
        "llm": {
            "enabled": False,
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0.1,
            "max_tokens": 500,
            "timeout_s": 1.0,
        },
        "adapter_config": {
            "task_id": "harvest_milk",
            "image_size": [160, 256],
        },
        "policy_generator": {
            "weights": {
                "goal_coherence": 0.6,
                "prediction_error": 0.4,
                "allostatic_survival_fit": 0.2,
                "allostatic_urgency_alignment": 0.2,
            },
            "fallback_scores": {
                "goal_coherence": 0.5,
                "prediction_error": 0.5,
                "allostatic_survival_fit": 0.5,
                "allostatic_urgency_alignment": 0.5,
            },
            "discovery": {
                "reserved_methods": [
                    "reset",
                    "step",
                    "close",
                    "sample_action",
                    "get_available_vitals",
                    "get_available_policies",
                    "get_action_space_actions",
                    "get_raw_observation",
                ]
            },
            "long_term_memory": {
                "path": "data/long_term_memory/policies.json",
                "max_score_history": 200,
                "max_outcome_history": 200,
            },
            "max_expected_error": 1.0,
            "prediction_error_window": 20,
            "allostatic_controller": {
                "enabled": True,
                "llm_interval_steps": 5,
                "llm_timeout_s": 1.0,
                "max_voxel_chars": 2000,
                "life_drop_threshold": 2.0,
                "food_drop_threshold": 2.0,
                "air_drop_threshold": 20.0,
                "high_risk_trigger": 0.7,
                "default_goal_horizon_steps": 160,
                "heuristic_confidence": 0.65,
                "temperature": 0.1,
                "max_tokens": 500,
            },
        },
    }
    if not path.exists():
        return default

    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("config.json must contain a top-level JSON object.")
    return _deep_merge(default, parsed)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    load_env_file(root / ".env")
    app_config = load_config(root / "config.json")

    max_steps = int(os.getenv("MAX_STEPS", "50"))
    include_inventory = _env_flag("INCLUDE_INVENTORY", True)
    include_voxels = _env_flag("INCLUDE_VOXELS", True)
    adapter_folder = str(app_config.get("adapter_folder", "minedojo"))
    llm_cfg = app_config.get("llm", {})
    adapter_cfg = app_config.get("adapter_config", {})
    policy_cfg = app_config.get("policy_generator", {})
    if not isinstance(llm_cfg, dict):
        raise ValueError("'llm' in config.json must be an object.")
    if not isinstance(adapter_cfg, dict):
        raise ValueError("'adapter_config' in config.json must be an object.")
    if not isinstance(policy_cfg, dict):
        raise ValueError("'policy_generator' in config.json must be an object.")
    ltm_cfg = policy_cfg.get("long_term_memory", {})
    if not isinstance(ltm_cfg, dict):
        raise ValueError("'policy_generator.long_term_memory' must be an object.")

    config = LoggingConfig.from_env()
    with RunLogger(config) as logger:
        install_exception_hooks(logger)

        adapter, observation_mapper = build_adapter(adapter_folder, adapter_cfg)
        llm_client = build_llm_client(llm_cfg, logger=logger)

        memory_manager = MemoryManager(
            logger=logger,
            long_term_memory_config=ltm_cfg,
        )

        logger.event(
            "run.config",
            {
                "adapter_folder": adapter_folder,
                "adapter_config": adapter_cfg,
                "max_steps": max_steps,
                "include_inventory": include_inventory,
                "include_voxels": include_voxels,
                "llm": {
                    "enabled": bool(llm_cfg.get("enabled", False)),
                    "provider": llm_cfg.get("provider", "openai"),
                    "model": llm_cfg.get("model"),
                    "timeout_s": llm_cfg.get("timeout_s"),
                },
                "policy_generator": policy_cfg,
            },
        )

        loop = AgentLoop(
            adapter=adapter,
            observation_mapper=observation_mapper,
            memory_manager=memory_manager,
            adapter_folder=adapter_folder,
            policy_config=policy_cfg,
            llm_client=llm_client,
            logger=logger,
            include_inventory=include_inventory,
            include_voxels=include_voxels,
        )

        try:
            loop.run(max_steps)
        finally:
            adapter.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

from core.adapters.loader import build_adapter
from core.llm.factory import build_llm_client
from core.memory import MemoryConfig, MemoryManager
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
            "long_term_memory": {
                "path": "data/long_term_memory",
                "max_score_history": 200,
                "max_outcome_history": 200,
            },
            "max_expected_error": 1.0,
            "prediction_error_window": 20,
            "llm_conflict_threshold": 0.8,
            "skill_gap_urgency_threshold": 0.8,
            "pe_high_threshold": 0.7,
            "pe_streak_threshold": 5,
            "llm_reeval_interval": 10,
            "allostatic_controller": {
                "planning_horizon": 50,
                "history_window": 20,
                "irreversibility_bonus": 0.3,
                "recovery_weight_factor": 0.2,
                "urgency_tie_epsilon": 0.05,
                "threat_prior_weight": 0.3,
                "min_confidence": 0.5,
            },
            "perceptual_prediction_error": {
                "alpha": 0.1,
                "epsilon": 0.01,
                "sigma_clip": 3.0,
                "default_precision": 0.5,
                "min_precision": 0.3,
                "world_model_alpha": 0.1,
                "action_confidence_threshold": 20,
            },
            "memory": {
                "working_memory_capacity": 100,
                "pe_min_observations": 5,
                "pe_ema_alpha": 0.1,
                "faiss_k_default": 5,
                "faiss_epsilon": 1e-6,
                "episode_length": 1000,
            },
            "arousal_valence": {
                "w_health": 0.35,
                "w_hunger": 0.25,
                "w_threat": 0.30,
                "w_pred_err": 0.10,
                "v_health": 0.30,
                "v_hunger": 0.30,
                "v_resources": 0.20,
                "v_oxygen": 0.20,
                "decay_rate": 0.95,
                "resting_arousal": 0.10,
                "urgency_broadcast_threshold": 0.65,
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
    adapter_cfg = app_config.get("adapter_config", {})
    policy_cfg = app_config.get("policy_generator", {})
    llm_cfg = app_config.get("llm", {})
    if not isinstance(adapter_cfg, dict):
        raise ValueError("'adapter_config' in config.json must be an object.")
    if not isinstance(policy_cfg, dict):
        raise ValueError("'policy_generator' in config.json must be an object.")
    if not isinstance(llm_cfg, dict):
        raise ValueError("'llm' in config.json must be an object.")
    ltm_cfg = policy_cfg.get("long_term_memory", {})
    memory_cfg = policy_cfg.get("memory", {})
    if not isinstance(ltm_cfg, dict):
        raise ValueError("'policy_generator.long_term_memory' must be an object.")
    if not isinstance(memory_cfg, dict):
        raise ValueError("'policy_generator.memory' must be an object.")
    resolved_ltm_cfg = dict(ltm_cfg)
    ltm_path = Path(str(resolved_ltm_cfg.get("path", "data/long_term_memory")))
    if not ltm_path.is_absolute():
        ltm_path = (root / ltm_path).resolve()
    resolved_ltm_cfg["path"] = str(ltm_path)

    config = LoggingConfig.from_env()
    if not config.log_root.is_absolute():
        config.log_root = (root / config.log_root).resolve()
    with RunLogger(config) as logger:
        install_exception_hooks(logger)

        adapter, observation_mapper = build_adapter(adapter_folder, adapter_cfg)

        memory_manager = MemoryManager(
            config=MemoryConfig(
                working_memory_capacity=int(memory_cfg.get("working_memory_capacity", 100)),
                pe_min_observations=int(memory_cfg.get("pe_min_observations", 5)),
                pe_ema_alpha=float(memory_cfg.get("pe_ema_alpha", 0.1)),
                faiss_k_default=int(memory_cfg.get("faiss_k_default", 5)),
                faiss_epsilon=float(memory_cfg.get("faiss_epsilon", 1e-6)),
                episode_length=int(memory_cfg.get("episode_length", 1000)),
            ),
            logger=logger,
            long_term_memory_config=resolved_ltm_cfg,
        )
        llm_client = build_llm_client(llm_cfg, logger=logger)

        logger.event(
            "run.config",
            {
                "adapter_folder": adapter_folder,
                "adapter_config": adapter_cfg,
                "max_steps": max_steps,
                "include_inventory": include_inventory,
                "include_voxels": include_voxels,
                "policy_generator": policy_cfg,
            },
        )

        loop = AgentLoop(
            adapter=adapter,
            observation_mapper=observation_mapper,
            memory_manager=memory_manager,
            adapter_folder=adapter_folder,
            policy_config=policy_cfg,
            llm_config=llm_cfg,
            logger=logger,
            include_inventory=include_inventory,
            include_voxels=include_voxels,
            llm_client=llm_client,
        )

        try:
            loop.run(max_steps)
        finally:
            adapter.close()


if __name__ == "__main__":
    main()

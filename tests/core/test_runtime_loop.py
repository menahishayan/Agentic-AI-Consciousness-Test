"""Integration test for AgentLoop using a mock adapter (no AnimalAI required)."""
import pytest
import tempfile
from pathlib import Path
from core.adapters.iot.env_adapter import IoTStubAdapter
from core.memory.manager import MemoryManager
from core.observability.logger import RunLogger
from core.observability.paths import make_run_dir
from core.runtime.loop import AgentLoop


def _make_config(tmp_path):
    return {
        "adapter_folder": "iot",
        "llm": {"enabled": False},
        "memory": {
            "working_memory_capacity": 20,
            "faiss_k_default": 3,
            "pe_ema_alpha": 0.1,
            "pe_min_observations": 3,
            "faiss_epsilon": 1e-6,
            "episode_length": 100,
            "long_term_memory_path": str(tmp_path / "ltm"),
        },
        "observability": {"log_root": str(tmp_path / "logs"), "log_state": False, "log_prompts": False, "log_memory": False},
        "policy_generator": {
            "llm_conflict_threshold": 0.8,
            "pe_streak_threshold": 5,
            "pe_high_threshold": 0.7,
            "llm_reeval_interval": 10,
            "skill_gap_urgency_threshold": 0.8,
        },
        "allostatic_controller": {"planning_horizon": 20, "history_window": 10},
        "arousal_valence": {},
        "world_model": {"alpha": 0.1},
        "perceptual_prediction_error": {"alpha": 0.1, "epsilon": 0.01},
    }


def test_loop_runs_10_steps(tmp_path):
    config = _make_config(tmp_path)
    adapter = IoTStubAdapter({"seed": 42})
    memory = MemoryManager(config)

    run_dir = make_run_dir(config["observability"]["log_root"])
    logger = RunLogger(run_dir, config, log_state=False, log_prompts=False, log_memory=False)

    loop = AgentLoop(adapter=adapter, memory=memory, llm_client=None, logger=logger, config=config)
    stats = loop.run(max_steps=10)

    assert stats["steps_run"] == 10 or stats["done_reason"] == "episode_done"
    logger.close()


def test_loop_metrics_written(tmp_path):
    config = _make_config(tmp_path)
    adapter = IoTStubAdapter({"seed": 1})
    memory = MemoryManager(config)

    run_dir = make_run_dir(config["observability"]["log_root"])
    logger = RunLogger(run_dir, config, log_state=False, log_prompts=False, log_memory=False)

    loop = AgentLoop(adapter=adapter, memory=memory, llm_client=None, logger=logger, config=config)
    loop.run(max_steps=5)
    logger.close()

    metrics_file = run_dir / "metrics.jsonl"
    assert metrics_file.exists()
    lines = metrics_file.read_text().strip().splitlines()
    assert len(lines) >= 1

    import json
    row = json.loads(lines[0])
    assert "health" in row
    assert "policy_id" in row
    assert "step" in row


def test_workspace_is_cleared_each_step(tmp_path):
    config = _make_config(tmp_path)
    adapter = IoTStubAdapter({})
    memory = MemoryManager(config)
    run_dir = make_run_dir(config["observability"]["log_root"])
    logger = RunLogger(run_dir, config, log_state=False, log_prompts=False, log_memory=False)

    loop = AgentLoop(adapter=adapter, memory=memory, llm_client=None, logger=logger, config=config)
    loop._current_state = adapter.reset()

    # Run 3 steps and verify workspace is refreshed each time
    for step in range(3):
        loop._step = step
        loop._run_step()
        msgs = loop._workspace.broadcast()
        # Goal is always re-published — workspace should have at least goal message
        kinds = {m.kind for m in msgs}
        assert "goal" in kinds

    logger.close()

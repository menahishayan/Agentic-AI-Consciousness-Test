from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional

from core.layers.interoceptive import VitalStateMonitor
from core.memory.manager import MemoryManager
from core.memory.self_state import SelfStateMemory
from core.models.state import AgentState
from core.runtime.loop import AgentLoop


class _DummyAdapter:
    def __init__(self) -> None:
        self._steps = 0

    def reset(self) -> Any:
        return None, {"life": 20, "food": 18}

    def step(self, action: Any) -> Any:
        _ = action
        self._steps += 1
        return None, 1.0, True, {"life": 19, "food": 17}

    def close(self) -> None:
        return None

    def sample_action(self) -> str:
        return "noop"

    def get_available_vitals(self) -> List[str]:
        return ["life", "food"]


def _map_obs(
    raw_obs: Any,
    info: Any,
    vital_state_monitor: Optional[VitalStateMonitor] = None,
) -> AgentState:
    _ = raw_obs
    payload = info if isinstance(info, Mapping) else {}
    monitor = vital_state_monitor or VitalStateMonitor()
    monitor.update(payload)
    return AgentState.from_info(dict(payload))


def test_self_state_memory_records_and_filters() -> None:
    memory = SelfStateMemory(max_records=3)
    memory.record({"step": 0, "phase": "pre_action", "vital_state": {"state": {"life": 20}}})
    memory.record({"step": 0, "phase": "post_step", "vital_state": {"state": {"life": 19}}})
    memory.record({"step": 1, "phase": "pre_action", "vital_state": {"state": {"life": 18}}})
    memory.record({"step": 1, "phase": "post_step", "vital_state": {"state": {"life": 17}}})

    all_records = memory.query({})
    assert len(all_records) == 3
    assert all_records[0]["step"] == 0
    assert all_records[-1]["phase"] == "post_step"

    post_step_records = memory.query({"phase": "post_step"})
    assert len(post_step_records) == 2
    assert all(item["phase"] == "post_step" for item in post_step_records)

    step_records = memory.query({"step": 1, "limit": 1})
    assert len(step_records) == 1
    assert step_records[0]["step"] == 1


def test_agent_loop_persists_vital_state_snapshots(tmp_path: Path) -> None:
    memory_manager = MemoryManager(
        long_term_memory_config={
            "path": str(tmp_path / "policies.json"),
            "max_score_history": 20,
            "max_outcome_history": 20,
        }
    )
    loop = AgentLoop(
        adapter=_DummyAdapter(),
        observation_mapper=_map_obs,
        memory_manager=memory_manager,
        adapter_folder="dummy",
        policy_config={},
    )

    loop.run_step(0)

    snapshots = memory_manager.query({"target": "self_state"})
    assert isinstance(snapshots, list)
    assert len(snapshots) >= 3

    pre_action = [item for item in snapshots if item.get("phase") == "pre_action"]
    allostatic = [item for item in snapshots if item.get("phase") == "allostatic_pre_action"]
    post_step = [item for item in snapshots if item.get("phase") == "post_step"]

    assert pre_action
    assert allostatic
    assert post_step

    assert pre_action[-1]["vital_state"]["state"]["life"] == 20
    assert post_step[-1]["vital_state"]["state"]["life"] == 19
    assert "allostatic_assessment" in allostatic[-1]
    assert allostatic[-1]["allostatic_assessment"]["source"] == "drive_model"

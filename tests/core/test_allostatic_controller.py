from __future__ import annotations

from typing import Any, Dict

from core.layers.interoceptive import AllostaticController
from core.llm.types import LLMResponse
from core.models.state import AgentState


class _StubLLMClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0
        self.last_request = None

    def generate(self, request: Any) -> LLMResponse:
        self.calls += 1
        self.last_request = request
        return LLMResponse(text=self.text)


def _state(info: Dict[str, Any]) -> AgentState:
    return AgentState.from_info(info)


def _vitals(**values: Any) -> Dict[str, Any]:
    state = {
        "life": values.get("life", 20),
        "food": values.get("food", 20),
        "air": values.get("air", 300),
        "is_alive": values.get("is_alive", True),
        "is_dead": values.get("is_dead", False),
    }
    return {
        "expected_vitals": list(state.keys()),
        "state": state,
        "missing": [],
    }


def test_allostatic_controller_uses_llm_structured_output() -> None:
    llm = _StubLLMClient(
        text="""
{
  "survival_horizon_steps": 220,
  "risk_level": 0.35,
  "confidence": 0.88,
  "rationale_summary": "Food is sufficient but oxygen safety should be maintained.",
  "needs": [
    {
      "need_id": "restore_oxygen",
      "urgency": 0.45,
      "time_to_critical_steps": 190,
      "actions": ["ascend", "move"],
      "resources": ["air_pocket"],
      "evidence": ["air=190"]
    }
  ],
  "policy_bias_tags": ["survival", "oxygen", "ascend"]
}
""".strip()
    )
    controller = AllostaticController(
        llm_client=llm,
        config={"enabled": True, "llm_interval_steps": 5, "max_voxel_chars": 300},
    )

    assessment = controller.assess(
        step=0,
        state=_state({"xpos": 1, "ypos": 70, "zpos": -3, "voxels": [[[1, 2], [3, 4]]]}),
        goals=[{"description": "build shelter", "priority": 1.0}],
        vital_state=_vitals(life=18, food=17, air=190),
        obs={"voxels": [[[1, 2], [3, 4]]]},
        info={},
    )

    assert assessment["source"] == "llm"
    assert assessment["survival_horizon_steps"] == 220
    assert assessment["risk_level"] == 0.35
    assert assessment["confidence"] == 0.88
    assert assessment["needs"][0]["need_id"] == "restore_oxygen"
    assert "oxygen" in assessment["policy_bias_tags"]
    assert llm.calls == 1
    assert llm.last_request is not None
    assert "voxels_truncated" in llm.last_request.messages[1].content


def test_allostatic_controller_falls_back_to_heuristic_on_invalid_json() -> None:
    llm = _StubLLMClient(text="not-json")
    controller = AllostaticController(
        llm_client=llm,
        config={"enabled": True, "llm_interval_steps": 5},
    )

    assessment = controller.assess(
        step=0,
        state=_state({"xpos": 0, "ypos": 65, "zpos": 0}),
        goals=[{"description": "find food"}],
        vital_state=_vitals(life=12, food=8, air=300),
    )

    assert assessment["source"] == "heuristic"
    assert assessment["needs"]
    assert assessment["risk_level"] >= 0.4
    assert llm.calls == 1


def test_allostatic_controller_reuses_cache_when_not_triggered() -> None:
    llm = _StubLLMClient(
        text='{"survival_horizon_steps":180,"risk_level":0.2,"confidence":0.9,"rationale_summary":"Stable.","needs":[],"policy_bias_tags":["survival"]}'
    )
    controller = AllostaticController(
        llm_client=llm,
        config={"enabled": True, "llm_interval_steps": 5},
    )

    first = controller.assess(
        step=0,
        state=_state({"xpos": 2, "ypos": 64, "zpos": 2}),
        goals=[],
        vital_state=_vitals(life=20, food=20, air=300),
    )
    second = controller.assess(
        step=1,
        state=_state({"xpos": 2, "ypos": 64, "zpos": 2}),
        goals=[],
        vital_state=_vitals(life=20, food=20, air=300),
    )

    assert first["source"] == "llm"
    assert second["source"] == "cache"
    assert second["survival_horizon_steps"] == first["survival_horizon_steps"]
    assert llm.calls == 1


def test_allostatic_controller_refreshes_on_life_drop_trigger() -> None:
    llm = _StubLLMClient(
        text='{"survival_horizon_steps":160,"risk_level":0.5,"confidence":0.8,"rationale_summary":"Monitor vitals.","needs":[{"need_id":"preserve_health","urgency":0.6,"time_to_critical_steps":80,"actions":["retreat"],"resources":["cover"],"evidence":["life_low"]}],"policy_bias_tags":["health","retreat"]}'
    )
    controller = AllostaticController(
        llm_client=llm,
        config={
            "enabled": True,
            "llm_interval_steps": 5,
            "life_drop_threshold": 2.0,
        },
    )

    controller.assess(
        step=0,
        state=_state({"xpos": 0, "ypos": 64, "zpos": 0}),
        goals=[],
        vital_state=_vitals(life=20, food=20, air=300),
    )
    refreshed = controller.assess(
        step=1,
        state=_state({"xpos": 0, "ypos": 64, "zpos": 0}),
        goals=[],
        vital_state=_vitals(life=17, food=20, air=300),
    )

    assert refreshed["source"] == "llm"
    assert llm.calls == 2

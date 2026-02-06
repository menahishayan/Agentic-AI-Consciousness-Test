from __future__ import annotations

from typing import Any

from core.models.state import AgentState


def map_obs(raw_obs: Any, info: Any) -> AgentState:
    raise NotImplementedError("Observation mapping not implemented.")

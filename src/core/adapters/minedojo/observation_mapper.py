from __future__ import annotations

from typing import Any

from core.models.state import AgentState


def map_obs(raw_obs: Any, info: Any) -> AgentState:
    _ = raw_obs
    return AgentState.from_info(info or {})

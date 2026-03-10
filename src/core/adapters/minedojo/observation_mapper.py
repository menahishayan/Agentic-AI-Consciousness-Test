from __future__ import annotations

from typing import Any, Mapping, Optional

from core.layers.interoceptive import VitalStateMonitor
from core.models.state import AgentState

_DEFAULT_VITAL_STATE_MONITOR = VitalStateMonitor()


def map_obs(
    raw_obs: Any,
    info: Any,
    vital_state_monitor: Optional[VitalStateMonitor] = None,
) -> AgentState:
    _ = raw_obs
    state = AgentState.from_info(info or {})
    monitor = vital_state_monitor or _DEFAULT_VITAL_STATE_MONITOR
    payload = info if isinstance(info, Mapping) else {}
    monitor.update(payload)
    return state

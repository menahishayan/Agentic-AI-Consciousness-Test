"""
VitalStateMonitor — Layer 1, Interoceptive Foundation

Reads AgentState.homeostasis, normalizes values, and publishes vital_state
messages to GlobalWorkspace. This is the first step in each cognitive cycle.

No game-specific logic. No adapter imports.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.coordination.messages import AgentMessage
from core.coordination.workspace import GlobalWorkspace
from core.models.state import AgentState


class VitalStateMonitor:
    """
    Monitors the agent's homeostatic variables and publishes normalized
    vital state snapshots to the GlobalWorkspace each step.

    Implements Seth's "interoceptive foundation" — the beast machine's
    awareness of its own body state.
    """

    def __init__(self, available_vitals: List[str]) -> None:
        """
        Args:
            available_vitals: Channel names reported by the adapter
                              (e.g. ["health", "saturation", "energy"])
        """
        self._available_vitals = available_vitals

    def update(
        self,
        state: AgentState,
        workspace: GlobalWorkspace,
        step: int,
    ) -> Dict[str, Optional[float]]:
        """
        Read homeostatic values from state, publish to workspace, return snapshot.

        Returns:
            Dict mapping channel_id → normalized float [0,1] or None
        """
        vitals = self._extract_vitals(state)

        workspace.publish(AgentMessage(
            sender="VitalStateMonitor",
            kind="vital_state",
            payload=vitals,
            step=step,
        ))

        return vitals

    def _extract_vitals(self, state: AgentState) -> Dict[str, Optional[float]]:
        h = state.homeostasis
        raw = {
            "health":     h.health,
            "saturation": h.saturation,
            "energy":     h.energy,
            "oxygen":     h.oxygen,
            "is_alive":   float(h.is_alive) if h.is_alive is not None else 1.0,
            "resource_level":   state.resources.resource_level,
            "threat_proximity": state.resources.threat_proximity,
        }
        # Only return channels that the adapter declared available
        return {k: v for k, v in raw.items() if k in self._available_vitals or k in ("is_alive", "resource_level", "threat_proximity")}

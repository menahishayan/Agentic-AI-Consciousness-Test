from __future__ import annotations

from typing import Any, List, Union

from core.models.signals import Goal, ThreatSignal


class HomeostaticAgent:
    def __init__(self) -> None:
        pass

    def step(
        self, agent_state: Any, memory_manager: Any, workspace: Any
    ) -> List[Union[Goal, ThreatSignal]]:
        raise NotImplementedError("Homeostatic agent step not implemented.")

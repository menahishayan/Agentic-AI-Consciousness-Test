from __future__ import annotations

from typing import Any

from core.models.signals import ActionProposal


class MotorAgent:
    def __init__(self) -> None:
        pass

    def step(self, goals: Any, predictions: Any, memory_manager: Any) -> ActionProposal:
        raise NotImplementedError("Motor agent step not implemented.")

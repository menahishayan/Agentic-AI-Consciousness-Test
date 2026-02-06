from __future__ import annotations

from typing import Any


class PolicyGenerator:
    def propose_actions(self, context: Any) -> Any:
        raise NotImplementedError("Policy generation not implemented.")


class FreeEnergyMinimizer:
    def select_action(self, proposals: Any) -> Any:
        raise NotImplementedError("Free energy minimization not implemented.")


class MotorControlInterface:
    def translate(self, action: Any) -> Any:
        raise NotImplementedError("Motor control translation not implemented.")

from __future__ import annotations

from typing import Any


class MotorControlInterface:
    def translate(self, action: Any) -> Any:
        return action

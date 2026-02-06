from __future__ import annotations

from typing import Any

from core.models.signals import WorkspaceBroadcast


class MetacognitiveAgent:
    def __init__(self) -> None:
        pass

    def step(self, signals: Any, memory_manager: Any, workspace: Any) -> WorkspaceBroadcast:
        raise NotImplementedError("Metacognitive agent step not implemented.")

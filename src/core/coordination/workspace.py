from __future__ import annotations

from typing import List, Optional

from core.coordination.messages import AgentMessage


class GlobalWorkspace:
    """
    Central message bus for all inter-layer communication.

    Design contract:
    - Layers publish messages via publish()
    - Layers read the full message set via broadcast()
    - AgentLoop calls clear() at the start of each step (except persistent goals)
    - Persistent messages (kind="goal") are re-published each step by AgentLoop

    This is the ONLY communication channel. No layer may import or reference
    another layer directly.
    """

    def __init__(self) -> None:
        self._messages: List[AgentMessage] = []

    def publish(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def broadcast(self) -> List[AgentMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def get_by_kind(self, kind: str) -> List[AgentMessage]:
        return [m for m in self._messages if m.kind == kind]

    def get_latest(self, kind: str) -> Optional[AgentMessage]:
        matches = self.get_by_kind(kind)
        return matches[-1] if matches else None

    def __len__(self) -> int:
        return len(self._messages)

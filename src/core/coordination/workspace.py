from __future__ import annotations

from typing import List

from core.coordination.messages import AgentMessage


class GlobalWorkspace:
    def __init__(self) -> None:
        self._messages: List[AgentMessage] = []

    def publish(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def broadcast(self) -> List[AgentMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

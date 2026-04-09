"""
WorkingMemoryBuffer — rolling buffer of recent AgentState snapshots.

Provides temporal context for the agent's current situation.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

from core.models.state import AgentState


class WorkingMemoryBuffer:
    def __init__(self, capacity: int = 100) -> None:
        self._buffer: deque = deque(maxlen=capacity)
        self.capacity = capacity

    def record(self, state: AgentState) -> None:
        self._buffer.append(state)

    def get_recent(self, n: int = 10) -> List[AgentState]:
        items = list(self._buffer)
        return items[-n:] if len(items) >= n else items

    def get_all(self) -> List[AgentState]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

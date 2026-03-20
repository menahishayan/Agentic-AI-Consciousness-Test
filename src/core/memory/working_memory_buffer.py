from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class WorkingMemoryEntry:
    tick: int
    entry_type: str
    payload: Dict[str, Any]
    priority: float

    def __post_init__(self) -> None:
        self.priority = float(max(0.0, min(1.0, self.priority)))


class WorkingMemoryBuffer:
    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, int(capacity))
        self._buffer: deque[WorkingMemoryEntry] = deque(maxlen=self._capacity)

    def record(self, entry: WorkingMemoryEntry) -> None:
        if not isinstance(entry, WorkingMemoryEntry):
            raise TypeError("WorkingMemoryBuffer.record expects WorkingMemoryEntry.")

        dropped: Optional[WorkingMemoryEntry] = None
        if len(self._buffer) == self._capacity:
            dropped = self._buffer[0]
        self._buffer.append(entry)

        if (
            dropped is not None
            and dropped.entry_type == "goal"
            and float(dropped.priority) > 0.8
        ):
            logger.warning(
                "High-priority goal evicted from working memory (tick=%s, priority=%.3f).",
                dropped.tick,
                dropped.priority,
            )

    def get_recent(self, n: int, entry_type: Optional[str] = None) -> List[WorkingMemoryEntry]:
        if n <= 0:
            return []

        if entry_type is None:
            selected = list(self._buffer)
        else:
            selected = [entry for entry in self._buffer if entry.entry_type == entry_type]
        return list(reversed(selected))[: int(n)]

    def get_active_goals(self) -> List[WorkingMemoryEntry]:
        return [entry for entry in reversed(self._buffer) if entry.entry_type == "goal"]

    def clear(self) -> None:
        self._buffer.clear()

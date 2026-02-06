from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AgentMessage:
    sender: Optional[str] = None
    kind: Optional[str] = None
    payload: Optional[Any] = None

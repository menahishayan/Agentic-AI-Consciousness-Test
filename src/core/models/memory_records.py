from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SelfStateSnapshot:
    timestamp: Optional[str] = None
    state: Optional[Any] = None


@dataclass
class PredictionErrorRecord:
    timestamp: Optional[str] = None
    error: Optional[Any] = None


@dataclass
class PolicyTraceRecord:
    timestamp: Optional[str] = None
    action: Optional[Any] = None
    outcome: Optional[Any] = None

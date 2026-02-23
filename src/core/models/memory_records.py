from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


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


@dataclass
class PolicyMemoryRecord:
    policy_id: Optional[str] = None
    adapter_folder: Optional[str] = None
    callable_name: Optional[str] = None
    source: Optional[str] = None
    signature: Optional[str] = None
    tags: Optional[List[str]] = None
    discovered_at: Optional[str] = None
    last_seen_at: Optional[str] = None
    selected_count: Optional[int] = None
    success_count: Optional[int] = None
    last_selected_at: Optional[str] = None
    last_score: Optional[float] = None
    score_history: Optional[List[Any]] = None
    outcome_history: Optional[List[Any]] = None

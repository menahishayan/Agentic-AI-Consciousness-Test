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


@dataclass
class EpisodicMemoryRecord:
    timestamp: Optional[str] = None
    summary: Optional[str] = None
    details: Optional[Any] = None


@dataclass
class SemanticMemoryEntry:
    key: Optional[str] = None
    value: Optional[Any] = None


@dataclass
class ProceduralSkill:
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Any] = None

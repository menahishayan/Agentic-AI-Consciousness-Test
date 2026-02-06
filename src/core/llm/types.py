from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMRequest:
    messages: List[LLMMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class LLMResponse:
    text: str
    raw: Any = None
    usage: Optional[LLMUsage] = None
    latency_ms: Optional[float] = None
    provider: Optional[str] = None
    model: Optional[str] = None

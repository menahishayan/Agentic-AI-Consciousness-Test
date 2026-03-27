from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMMessage:
    role: str    # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMRequest:
    messages: List[LLMMessage]
    model: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.1
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    raw: Optional[Any] = None

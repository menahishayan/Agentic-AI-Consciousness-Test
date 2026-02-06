from __future__ import annotations

from typing import Protocol

from core.llm.types import LLMRequest, LLMResponse


class LLMClient(Protocol):
    def generate(self, request: LLMRequest) -> LLMResponse:
        ...

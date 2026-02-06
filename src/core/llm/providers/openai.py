from __future__ import annotations

from core.llm.client import LLMClient
from core.llm.types import LLMRequest, LLMResponse


class OpenAIClientStub(LLMClient):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("OpenAI client adapter not implemented.")

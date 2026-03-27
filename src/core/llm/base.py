from __future__ import annotations

from abc import ABC, abstractmethod

from core.llm.types import LLMRequest, LLMResponse


class AbstractLLMClient(ABC):
    """
    Provider-agnostic LLM interface.

    Brain layers import only this class. To switch provider, change
    config.json "provider" field and set the corresponding API key in .env.
    The PolicyGenerator never imports a concrete provider.
    """

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request and return the response."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. 'anthropic', 'openai')."""

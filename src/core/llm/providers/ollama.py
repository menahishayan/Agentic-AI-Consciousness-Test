"""
Ollama provider — runs local models via Ollama's OpenAI-compatible API.

Requires Ollama running locally (default: http://localhost:11434).
No API key needed. Set base_url in config["llm"]["base_url"] to override.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from core.llm.base import AbstractLLMClient
from core.llm.types import LLMRequest, LLMResponse

log = logging.getLogger(__name__)


class OllamaClient(AbstractLLMClient):
    def __init__(
        self,
        default_model: Optional[str] = None,
        base_url: str = "http://localhost:11434/v1",
        default_max_tokens: int = 500,
        default_temperature: float = 0.1,
        timeout: Optional[float] = 15.0,
        logger: Optional[Any] = None,
    ) -> None:
        try:
            import openai as _openai
            self._client = _openai.OpenAI(
                api_key="ollama",  # Ollama ignores the key; openai SDK requires a non-empty value
                base_url=base_url,
                timeout=timeout,
            )
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Run: pip install openai>=1.0.0"
            ) from exc

        self._default_model = default_model or "gemma3:1b"
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature
        self._run_logger = logger

    @property
    def provider_name(self) -> str:
        return "ollama"

    def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self._default_model
        temperature = (
            request.temperature if request.temperature is not None
            else self._default_temperature
        )
        max_tokens = request.max_tokens or self._default_max_tokens

        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        t0 = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            log.error("Ollama API error: %s", exc)
            raise

        latency_ms = (time.monotonic() - t0) * 1000.0
        content = response.choices[0].message.content or "" if response.choices else ""

        usage = response.usage
        llm_response = LLMResponse(
            content=content,
            model=model,
            provider="ollama",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            raw=response,
        )

        # Note: LLM logging is handled by PolicyGenerator's _llm_log_cb,
        # which has full context (step, trigger_reason, selected action).
        # Do not log here to avoid duplicate entries in llm.jsonl.

        return llm_response

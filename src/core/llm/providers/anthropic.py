"""
Anthropic Claude provider — primary LLM implementation.

Uses the `anthropic` SDK. Model defaults to claude-sonnet-4-6.
All configuration (model, temperature, max_tokens, timeout) comes from config.json.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from core.llm.base import AbstractLLMClient
from core.llm.types import LLMRequest, LLMResponse

log = logging.getLogger(__name__)


class AnthropicClient(AbstractLLMClient):
    def __init__(
        self,
        api_key: str,
        default_model: Optional[str] = None,
        default_max_tokens: int = 500,
        default_temperature: float = 0.1,
        timeout: Optional[float] = 15.0,
        logger: Optional[Any] = None,
    ) -> None:
        try:
            import anthropic as _anthropic
            self._client = _anthropic.Anthropic(
                api_key=api_key,
                timeout=timeout,
            )
        except ImportError as exc:
            raise ImportError("anthropic package not installed. Run: pip install anthropic>=0.25.0") from exc

        self._default_model = default_model or "claude-sonnet-4-6"
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature
        self._run_logger = logger

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self._default_model
        temperature = request.temperature if request.temperature is not None else self._default_temperature
        max_tokens = request.max_tokens or self._default_max_tokens

        # Separate system message if present
        system_content = None
        messages = []
        for msg in request.messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                messages.append({"role": msg.role, "content": msg.content})

        if not messages:
            messages = [{"role": "user", "content": ""}]

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system_content:
            kwargs["system"] = system_content

        t0 = time.monotonic()
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            log.error("Anthropic API error: %s", exc)
            raise

        latency_ms = (time.monotonic() - t0) * 1000.0
        content = response.content[0].text if response.content else ""

        llm_response = LLMResponse(
            content=content,
            model=response.model,
            provider="anthropic",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            raw=response,
        )

        if self._run_logger is not None:
            try:
                self._run_logger.llm(
                    prompt=messages[-1]["content"] if messages else "",
                    response=content,
                    model=model,
                    latency_ms=latency_ms,
                    input_tokens=llm_response.input_tokens,
                    output_tokens=llm_response.output_tokens,
                )
            except Exception:
                pass

        return llm_response

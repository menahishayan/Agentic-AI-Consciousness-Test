"""
OpenAI provider — full implementation.

Uses the `openai` SDK. Set provider="openai" in config.json and
OPENAI_API_KEY in .env to use this provider.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from core.llm.base import AbstractLLMClient
from core.llm.types import LLMRequest, LLMResponse

log = logging.getLogger(__name__)


class OpenAIClient(AbstractLLMClient):
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
            import openai as _openai
            self._client = _openai.OpenAI(api_key=api_key, timeout=timeout)
        except ImportError as exc:
            raise ImportError("openai package not installed. Run: pip install openai>=1.30.0") from exc

        self._default_model = default_model or "gpt-4o-mini"
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature
        self._run_logger = logger

    @property
    def provider_name(self) -> str:
        return "openai"

    def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self._default_model
        temperature = request.temperature if request.temperature is not None else self._default_temperature
        max_tokens = request.max_tokens or self._default_max_tokens

        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        t0 = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            log.error("OpenAI API error: %s", exc)
            raise

        latency_ms = (time.monotonic() - t0) * 1000.0
        content = response.choices[0].message.content or "" if response.choices else ""

        llm_response = LLMResponse(
            content=content,
            model=model,
            provider="openai",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
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

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.llm.client import LLMClient
from core.llm.types import LLMMessage, LLMRequest, LLMResponse, LLMUsage
from core.observability.logger import RunLogger


def _messages_to_input(messages: List[LLMMessage]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for msg in messages:
        payload.append({"role": msg.role, "content": msg.content})
    return payload


class OpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        default_model: Optional[str] = None,
        default_params: Optional[Dict[str, Any]] = None,
        logger: Optional[RunLogger] = None,
        client: Any = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.default_model = default_model
        self.default_params = default_params or {}
        self.logger = logger

        if client is not None:
            self.client = client
            return

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("openai package is required for OpenAIClient") from exc

        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if organization:
            kwargs["organization"] = organization
        if project:
            kwargs["project"] = project
        if timeout is not None:
            kwargs["timeout"] = timeout
        self.client = OpenAI(**kwargs)

    def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.default_model
        if not model:
            raise ValueError("LLMRequest.model is required when no default_model is set.")

        input_messages = _messages_to_input(request.messages)
        params: Dict[str, Any] = {
            "model": model,
            "input": input_messages,
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params["max_output_tokens"] = request.max_tokens

        provider_params = request.metadata.get("provider_params") if request.metadata else None
        if isinstance(provider_params, dict):
            params.update(provider_params)

        params.update(self.default_params)

        if self.logger is not None:
            self.logger.llm_request(
                {
                    "provider": "openai",
                    "model": model,
                    "messages": input_messages,
                    "request_params": {
                        "temperature": request.temperature,
                        "max_output_tokens": request.max_tokens,
                    },
                }
            )

        start = time.time()
        resp = self.client.responses.create(**params)
        latency_ms = (time.time() - start) * 1000.0

        text = getattr(resp, "output_text", None)
        if not text:
            text = ""
            output = getattr(resp, "output", None)
            if output:
                for item in output:
                    content = getattr(item, "content", None) or []
                    for part in content:
                        part_type = getattr(part, "type", None)
                        if part_type in {"output_text", "text"}:
                            text += getattr(part, "text", "")

        usage = None
        raw_usage = getattr(resp, "usage", None)
        if raw_usage is not None:
            usage = LLMUsage(
                prompt_tokens=getattr(raw_usage, "prompt_tokens", None),
                completion_tokens=getattr(raw_usage, "completion_tokens", None),
                total_tokens=getattr(raw_usage, "total_tokens", None),
            )

        response = LLMResponse(
            text=text,
            raw=resp,
            usage=usage,
            latency_ms=latency_ms,
            provider="openai",
            model=model,
        )

        if self.logger is not None:
            self.logger.llm_response(
                {
                    "provider": "openai",
                    "model": model,
                    "text": response.text,
                    "usage": usage,
                    "latency_ms": latency_ms,
                    "raw": resp,
                }
            )

        return response


class OpenAIClientStub(OpenAIClient):
    pass

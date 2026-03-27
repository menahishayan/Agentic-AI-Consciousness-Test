"""
Gemini provider — stub.

Set provider="gemini" in config.json and GEMINI_API_KEY in .env.
Full implementation requires google-generativeai>=0.5.0 (already in requirements.txt).

To implement: replace the NotImplementedError body with the google.generativeai SDK calls.
"""
from __future__ import annotations

from core.llm.base import AbstractLLMClient
from core.llm.types import LLMRequest, LLMResponse


class GeminiClient(AbstractLLMClient):
    def __init__(self, api_key: str, **kwargs) -> None:
        self._api_key = api_key
        # TODO: initialize google.generativeai client here

    @property
    def provider_name(self) -> str:
        return "gemini"

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError(
            "Gemini provider is a stub. To implement:\n"
            "  1. Install: pip install google-generativeai>=0.5.0\n"
            "  2. Import: import google.generativeai as genai\n"
            "  3. Initialize: genai.configure(api_key=self._api_key)\n"
            "  4. Call: model = genai.GenerativeModel('gemini-pro'); response = model.generate_content(...)\n"
            "  5. Return LLMResponse with content, model, provider fields populated."
        )

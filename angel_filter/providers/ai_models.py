"""Shared helpers for AI answer providers.

These providers call chat-style model APIs and normalize each model's answer
into the same ProviderResult shape used by the rest of the pipeline.
"""

from __future__ import annotations

import os
from typing import Any

from angel_filter.providers.base import BaseProvider, ProviderError, ProviderResult


ANSWER_SYSTEM_PROMPT = (
    "Answer the user's question directly and accurately. Be concise, name any "
    "important uncertainty, and do not invent citations or facts."
)
DEFAULT_MAX_TOKENS = 700

MODEL_OPTIONS: dict[str, dict[str, str]] = {
    "claude": {
        "default": "claude-sonnet-4-20250514",
        "fast": "claude-3-5-haiku-20241022",
    },
    "openai": {
        "default": "gpt-5.4-mini",
        "low_cost": "gpt-5-mini",
        "legacy_low_cost": "gpt-4.1-mini",
    },
    "gemini": {
        "default": "gemini-2.5-flash",
        "free_tier": "gemini-2.5-flash",
        "fast_free_tier": "gemini-2.5-flash-lite",
    },
}


class AIAnswerProvider(BaseProvider):
    """Base class for providers that return one model-generated answer."""

    api_key_env: str = ""
    model_env: str = ""
    model_preset_env: str = ""
    default_model: str = ""
    model_options_key: str = ""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 30.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        httpx_transport: Any | None = None,
    ):
        self.api_key = os.getenv(self.api_key_env) if api_key is None else api_key
        self.model = model or os.getenv(self.model_env) or self._model_from_preset()
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.httpx_transport = httpx_transport

    def _model_from_preset(self) -> str:
        options = MODEL_OPTIONS.get(self.model_options_key, {})
        preset = os.getenv(self.model_preset_env, "default")
        return options.get(preset, self.default_model)

    async def _post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ProviderError(f"{self.name} is missing {self.api_key_env}.")

        import httpx

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s,
                transport=self.httpx_transport,
            ) as client:
                response = await client.post(url, headers=headers, json=json_body)
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            raise ProviderError(f"{self.name} call failed: {exc}") from exc

    def _result(self, answer: str, raw: dict[str, Any]) -> list[ProviderResult]:
        answer = answer.strip()
        if not answer:
            raise ProviderError(f"{self.name} returned an empty answer.")

        return [
            ProviderResult(
                title=f"{self.name.title()} answer",
                snippet=answer,
                url=None,
                provider=self.name,
                rank_in_provider=0,
                sponsored=False,
                raw=raw,
            )
        ]


class ClaudeProvider(AIAnswerProvider):
    """Anthropic Claude provider."""

    name = "claude"
    api_key_env = "ANTHROPIC_API_KEY"
    model_env = "CLAUDE_MODEL"
    model_preset_env = "CLAUDE_MODEL_PRESET"
    default_model = "claude-sonnet-4-20250514"
    model_options_key = "claude"
    api_url = "https://api.anthropic.com/v1/messages"

    async def query(self, user_query: str, max_results: int = 10) -> list[ProviderResult]:
        data = await self._post_json(
            self.api_url,
            headers={
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json_body={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": ANSWER_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_query}],
            },
        )
        return self._result(_extract_claude_text(data), data)


class OpenAIProvider(AIAnswerProvider):
    """OpenAI provider using the Responses API."""

    name = "openai"
    api_key_env = "OPENAI_API_KEY"
    model_env = "OPENAI_MODEL"
    model_preset_env = "OPENAI_MODEL_PRESET"
    default_model = "gpt-5.4-mini"
    model_options_key = "openai"
    api_url = "https://api.openai.com/v1/responses"

    async def query(self, user_query: str, max_results: int = 10) -> list[ProviderResult]:
        data = await self._post_json(
            self.api_url,
            headers={
                "authorization": f"Bearer {self.api_key or ''}",
                "content-type": "application/json",
            },
            json_body={
                "model": self.model,
                "instructions": ANSWER_SYSTEM_PROMPT,
                "input": user_query,
                "max_output_tokens": self.max_tokens,
            },
        )
        return self._result(_extract_openai_text(data), data)


class GeminiProvider(AIAnswerProvider):
    """Google Gemini provider."""

    name = "gemini"
    api_key_env = "GEMINI_API_KEY"
    model_env = "GEMINI_MODEL"
    model_preset_env = "GEMINI_MODEL_PRESET"
    default_model = "gemini-2.5-flash"
    model_options_key = "gemini"
    api_base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    async def query(self, user_query: str, max_results: int = 10) -> list[ProviderResult]:
        url = f"{self.api_base_url}/{self.model}:generateContent"
        data = await self._post_json(
            url,
            headers={
                "x-goog-api-key": self.api_key or "",
                "content-type": "application/json",
            },
            json_body={
                "system_instruction": {
                    "parts": [{"text": ANSWER_SYSTEM_PROMPT}],
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_query}],
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": self.max_tokens,
                },
            },
        )
        return self._result(_extract_gemini_text(data), data)


def _extract_claude_text(data: dict[str, Any]) -> str:
    parts = []
    for item in data.get("content", []):
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
    answer = "\n".join(p for p in parts if p).strip()
    if not answer:
        raise ProviderError("claude response did not include text content.")
    return answer


def _extract_openai_text(data: dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data["output_text"]).strip()

    parts = []
    for output in data.get("output", []):
        for content in output.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                parts.append(content.get("text", ""))
    answer = "\n".join(p for p in parts if p).strip()
    if not answer:
        raise ProviderError("openai response did not include text content.")
    return answer


def _extract_gemini_text(data: dict[str, Any]) -> str:
    parts = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if "text" in part:
                parts.append(part["text"])
    answer = "\n".join(p for p in parts if p).strip()
    if not answer:
        raise ProviderError("gemini response did not include text content.")
    return answer

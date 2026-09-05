"""Groq-hosted provider for one-shot debugging inference."""

from __future__ import annotations

import os
from typing import Any

from providers.base import GenerationResult, LLMProvider, TextCallback

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY_ENVIRONMENT_VARIABLE = "GROQ_API_KEY"
GROQ_MODEL_ENVIRONMENT_VARIABLE = "GROQ_MODEL"
PROVIDER_NAME = "Groq"


def resolve_groq_model(cli_model: str | None = None) -> str:
    """Resolve the Groq model from CLI, environment, then default."""
    return (
        cli_model
        or os.environ.get(GROQ_MODEL_ENVIRONMENT_VARIABLE)
        or DEFAULT_GROQ_MODEL
    )


class GroqProvider(LLMProvider):
    """Generate one debugging response with the Groq Chat Completions API."""

    def __init__(
        self,
        model_name: str | None = None,
        max_tokens: int = 512,
        *,
        client: Any | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        api_key = os.environ.get(GROQ_API_KEY_ENVIRONMENT_VARIABLE)
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Export it before using the Groq provider."
            )

        self.model_name = resolve_groq_model(model_name)
        self.max_tokens = max_tokens
        if client is None:
            from groq import Groq

            client = Groq(api_key=api_key, max_retries=0)
        self._client = client

    def generate(
        self,
        prompt: str,
        *,
        on_text: TextCallback | None = None,
    ) -> GenerationResult:
        """Stream one Groq response and return its complete text and usage."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        stream = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.0,
            max_completion_tokens=self.max_tokens,
            stream=True,
        )

        response_parts: list[str] = []
        usage: Any | None = None
        for chunk in stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content or ""
                if content:
                    response_parts.append(content)
                    if on_text is not None:
                        on_text(content)
            groq_metadata = getattr(chunk, "x_groq", None)
            chunk_usage = getattr(groq_metadata, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage

        response_text = "".join(response_parts).strip()
        if not response_text:
            raise ValueError("Groq returned an empty response.")

        return GenerationResult(
            text=response_text,
            input_tokens=usage.prompt_tokens if usage is not None else None,
            output_tokens=usage.completion_tokens if usage is not None else None,
            reported_total_tokens=usage.total_tokens if usage is not None else None,
            provider_name=PROVIDER_NAME,
            model_name=self.model_name,
        )

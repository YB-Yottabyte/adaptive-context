"""Groq-hosted provider for one-shot debugging inference."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from providers.base import GenerationResult, LLMProvider, TextCallback

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY_ENVIRONMENT_VARIABLE = "GROQ_API_KEY"
GROQ_MODEL_ENVIRONMENT_VARIABLE = "GROQ_MODEL"
PROVIDER_NAME = "Groq"
_MISSING = object()


@dataclass
class _GroqStreamDiagnostics:
    """Collect non-sensitive structural metadata from one streamed response."""

    api_status: str = "unavailable"
    chunks_received: int = 0
    chunks_with_choices: int = 0
    choices_received: int = 0
    content_deltas: int = 0
    content_characters: int = 0
    reasoning_present: bool = False
    reasoning_characters: int = 0
    malformed_chunks: int = 0
    finish_reasons: set[str] = field(default_factory=set)
    usage: Any | None = None

    def observe(self, chunk: Any) -> str:
        """Inspect one chunk and return only its visible text delta."""
        self.chunks_received += 1
        choices = getattr(chunk, "choices", _MISSING)
        if choices is _MISSING or not isinstance(choices, (list, tuple)):
            self.malformed_chunks += 1
            self._observe_usage(chunk)
            return ""
        if not choices:
            self._observe_usage(chunk)
            return ""

        self.chunks_with_choices += 1
        self.choices_received += len(choices)
        choice = choices[0]
        self._observe_finish_reason(choice)
        delta = getattr(choice, "delta", _MISSING)
        if delta is _MISSING or delta is None:
            self.malformed_chunks += 1
            self._observe_usage(chunk)
            return ""

        self._observe_reasoning(delta)
        content = getattr(delta, "content", None)
        if content is not None and not isinstance(content, str):
            self.malformed_chunks += 1
            self._observe_usage(chunk)
            return ""
        if content:
            self.content_deltas += 1
            self.content_characters += len(content)
        self._observe_usage(chunk)
        return content or ""

    def summary(self) -> str:
        """Format safe metadata suitable for an exception shown to a user."""
        content_state = (
            f"present ({self.content_characters} chars)"
            if self.content_characters
            else "empty"
        )
        reasoning_state = (
            f"present ({self.reasoning_characters} chars)"
            if self.reasoning_present
            else "absent"
        )
        finish_reasons = (
            ",".join(sorted(self.finish_reasons))
            if self.finish_reasons
            else "not provided"
        )
        return (
            "Non-sensitive response metadata: "
            f"api_status={self.api_status}; chunks={self.chunks_received}; "
            f"chunks_with_choices={self.chunks_with_choices}; "
            f"choices={self.choices_received}; content={content_state}; "
            f"reasoning={reasoning_state}; finish_reason={finish_reasons}; "
            f"usage_tokens={self._usage_summary()}; "
            f"malformed_chunks={self.malformed_chunks}."
        )

    def _observe_reasoning(self, delta: Any) -> None:
        reasoning = getattr(delta, "reasoning", None)
        if reasoning is None:
            reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            self.reasoning_present = True
            if isinstance(reasoning, str):
                self.reasoning_characters += len(reasoning)

    def _observe_finish_reason(self, choice: Any) -> None:
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None:
            self.finish_reasons.add(self._safe_label(finish_reason))

    def _observe_usage(self, chunk: Any) -> None:
        groq_metadata = getattr(chunk, "x_groq", None)
        usage = getattr(groq_metadata, "usage", None)
        if usage is not None:
            self.usage = usage

    def _usage_summary(self) -> str:
        if self.usage is None:
            return "unavailable"
        values = (
            self._safe_count(getattr(self.usage, "prompt_tokens", None)),
            self._safe_count(getattr(self.usage, "completion_tokens", None)),
            self._safe_count(getattr(self.usage, "total_tokens", None)),
        )
        return f"input:{values[0]},output:{values[1]},total:{values[2]}"

    @staticmethod
    def _safe_count(value: Any) -> str:
        return str(value) if isinstance(value, int) and value >= 0 else "unavailable"

    @staticmethod
    def _safe_label(value: Any) -> str:
        text = str(value)
        return text if re.fullmatch(r"[A-Za-z0-9_.:-]{1,40}", text) else "unavailable"

    @classmethod
    def from_stream(cls, stream: Any) -> _GroqStreamDiagnostics:
        """Capture an HTTP status only when the SDK exposes a scalar code."""
        for candidate in (
            stream,
            getattr(stream, "response", None),
            getattr(stream, "_response", None),
        ):
            status = getattr(candidate, "status_code", None)
            if isinstance(status, int):
                return cls(api_status=str(status))
        return cls()


def resolve_groq_model(cli_model: str | None = None) -> str:
    """Resolve the Groq model from CLI, environment, then default."""
    return (
        cli_model
        or os.environ.get(GROQ_MODEL_ENVIRONMENT_VARIABLE)
        or DEFAULT_GROQ_MODEL
    )


def is_gpt_oss_model(model_name: str) -> bool:
    """Return whether a provider-qualified model ID selects GPT-OSS."""
    return model_name.rsplit("/", maxsplit=1)[-1].lower().startswith("gpt-oss-")


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

        request_options: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model_name,
            "temperature": 0.0,
            "max_completion_tokens": self.max_tokens,
            "stream": True,
        }
        if is_gpt_oss_model(self.model_name):
            request_options["reasoning_effort"] = "low"

        stream = self._client.chat.completions.create(**request_options)

        response_parts: list[str] = []
        diagnostics = _GroqStreamDiagnostics.from_stream(stream)
        for chunk in stream:
            content = diagnostics.observe(chunk)
            if content:
                response_parts.append(content)
                if on_text is not None:
                    on_text(content)

        response_text = "".join(response_parts).strip()
        if diagnostics.malformed_chunks:
            raise ValueError(
                f"Groq returned a malformed streamed response. {diagnostics.summary()}"
            )
        if not response_text:
            if diagnostics.chunks_with_choices == 0:
                message = "Groq returned no choices."
            elif diagnostics.reasoning_present:
                message = (
                    "Groq returned reasoning, but message.content was empty. The "
                    "completion may have ended before producing a final answer."
                )
            else:
                message = "Groq returned choices, but message.content was empty."
            raise ValueError(f"{message} {diagnostics.summary()}")

        usage = diagnostics.usage
        return GenerationResult(
            text=response_text,
            input_tokens=usage.prompt_tokens if usage is not None else None,
            output_tokens=usage.completion_tokens if usage is not None else None,
            reported_total_tokens=usage.total_tokens if usage is not None else None,
            provider_name=PROVIDER_NAME,
            model_name=self.model_name,
        )

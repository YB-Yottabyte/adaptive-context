"""Shared interface for language-model providers."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

TextCallback = Callable[[str], None]


@dataclass(frozen=True)
class GenerationResult:
    """Generated text and optional token usage reported by a provider."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    reported_total_tokens: int | None = None
    provider_name: str | None = None
    model_name: str | None = None

    @property
    def total_tokens(self) -> int | None:
        """Return the reported total, or calculate it when possible."""
        if self.reported_total_tokens is not None:
            return self.reported_total_tokens
        if self.input_tokens is not None and self.output_tokens is not None:
            return self.input_tokens + self.output_tokens
        return None


class LLMProvider(ABC):
    """Interface implemented by debugging language-model providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        on_text: TextCallback | None = None,
    ) -> GenerationResult:
        """Generate a response, optionally reporting visible text as it arrives."""

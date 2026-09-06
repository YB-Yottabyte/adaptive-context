"""Language-model provider discovery, configuration, and construction."""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Mapping
from importlib.util import find_spec

from providers.base import LLMProvider
from providers.groq_provider import (
    DEFAULT_GROQ_MODEL,
    GROQ_API_KEY_ENVIRONMENT_VARIABLE,
    GROQ_MODEL_ENVIRONMENT_VARIABLE,
    GroqProvider,
)
from providers.local_mlx import DEFAULT_MODEL as DEFAULT_LOCAL_MODEL
from providers.local_mlx import LocalMLXProvider

MODEL_ENVIRONMENT_VARIABLE = "CONTEXT_DEBUG_MODEL"


class ProviderFactory:
    """Discover available providers and create the selected implementation."""

    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        self.environment = os.environ if environment is None else environment

    def available_providers(self, *, local_model: str | None = None) -> tuple[str, ...]:
        """Return configured providers in primary-baseline-first order."""
        available: list[str] = []
        if self.environment.get(
            GROQ_API_KEY_ENVIRONMENT_VARIABLE
        ) and self.module_available("groq"):
            available.append("groq")

        configured_local_model = (
            local_model
            or self.environment.get(MODEL_ENVIRONMENT_VARIABLE)
            or DEFAULT_LOCAL_MODEL
        )
        if configured_local_model and self.local_runtime_available():
            available.append("local")
        return tuple(available)

    def validate(
        self, provider_name: str, available_providers: tuple[str, ...]
    ) -> None:
        """Raise a concise setup error when a requested provider cannot run."""
        if provider_name in available_providers:
            return
        if provider_name == "groq":
            raise ValueError("Groq provider is not configured. Set GROQ_API_KEY.")
        if provider_name == "local":
            raise ValueError(
                "Local Hugging Face provider is unavailable. It requires Apple "
                "Silicon and the local model dependencies."
            )
        raise ValueError(f"Unsupported provider: {provider_name}")

    def resolve_model(self, provider_name: str, cli_model: str | None) -> str:
        """Resolve a model without sharing defaults between providers."""
        if provider_name == "groq":
            return (
                cli_model
                or self.environment.get(GROQ_MODEL_ENVIRONMENT_VARIABLE)
                or DEFAULT_GROQ_MODEL
            )
        return (
            cli_model
            or self.environment.get(MODEL_ENVIRONMENT_VARIABLE)
            or DEFAULT_LOCAL_MODEL
        )

    def create(self, name: str, model_name: str, max_tokens: int) -> LLMProvider:
        """Create the selected language-model provider."""
        if name == "local":
            return LocalMLXProvider(model_name=model_name, max_tokens=max_tokens)
        if name == "groq":
            return GroqProvider(model_name=model_name, max_tokens=max_tokens)
        raise ValueError(f"Unsupported provider: {name}")

    def module_available(self, module_name: str) -> bool:
        """Return whether a provider dependency can be imported."""
        return find_spec(module_name) is not None

    def local_runtime_available(self) -> bool:
        """Return whether this process can run MLX locally."""
        return (
            sys.platform == "darwin"
            and platform.machine() == "arm64"
            and self.module_available("mlx")
            and self.module_available("mlx_lm")
        )

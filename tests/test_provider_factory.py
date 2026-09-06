from unittest.mock import patch

import pytest

from providers.factory import ProviderFactory


def test_available_providers_checks_configuration_and_dependencies() -> None:
    factory = ProviderFactory(
        {
            "GROQ_API_KEY": "configured",
            "CONTEXT_DEBUG_MODEL": "mlx-community/test-model",
        }
    )
    with (
        patch.object(factory, "module_available", return_value=True),
        patch.object(factory, "local_runtime_available", return_value=True),
    ):
        assert factory.available_providers() == ("groq", "local")


def test_available_providers_excludes_unconfigured_providers() -> None:
    factory = ProviderFactory({})
    with (
        patch.object(factory, "module_available", return_value=True),
        patch.object(factory, "local_runtime_available", return_value=False),
    ):
        assert factory.available_providers() == ()


def test_available_providers_recognizes_default_local_model() -> None:
    factory = ProviderFactory({})
    with patch.object(factory, "local_runtime_available", return_value=True):
        assert factory.available_providers() == ("local",)


def test_available_providers_requires_groq_dependency() -> None:
    factory = ProviderFactory({"GROQ_API_KEY": "set"})
    with (
        patch.object(factory, "module_available", return_value=False),
        patch.object(factory, "local_runtime_available", return_value=False),
    ):
        assert factory.available_providers() == ()


def test_validate_rejects_explicit_unavailable_provider() -> None:
    factory = ProviderFactory({})
    factory.validate("groq", ("groq",))

    with pytest.raises(
        ValueError,
        match="Local Hugging Face provider is unavailable",
    ):
        factory.validate("local", ("groq",))


def test_model_resolution_keeps_provider_defaults_separate() -> None:
    factory = ProviderFactory(
        {
            "GROQ_MODEL": "hosted-model",
            "CONTEXT_DEBUG_MODEL": "local-model",
        }
    )

    assert factory.resolve_model("groq", None) == "hosted-model"
    assert factory.resolve_model("local", None) == "local-model"
    assert factory.resolve_model("groq", "override") == "override"

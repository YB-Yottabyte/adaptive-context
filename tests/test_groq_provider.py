from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from providers.groq_provider import (
    DEFAULT_GROQ_MODEL,
    GroqProvider,
    resolve_groq_model,
)


def make_mock_client() -> tuple[SimpleNamespace, Mock]:
    """Create a minimal Groq client mock with streamed completion chunks."""
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Evidence "))],
            x_groq=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="considered:\n- return ignored")
                )
            ],
            x_groq=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        reasoning="private reasoning is not visible output",
                    )
                )
            ],
            x_groq=SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=120,
                    completion_tokens=30,
                    total_tokens=155,
                )
            ),
        ),
    ]
    create = Mock(return_value=iter(chunks))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client, create


def test_missing_groq_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
        GroqProvider()


def test_non_positive_max_tokens_is_rejected_before_client_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="max_tokens must be at least 1"):
        GroqProvider(max_tokens=0)


def test_sdk_retries_are_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    with patch("groq.Groq") as groq_client:
        GroqProvider()

    groq_client.assert_called_once_with(api_key="test-key", max_retries=0)


def test_default_groq_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    assert resolve_groq_model() == DEFAULT_GROQ_MODEL


def test_environment_groq_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_MODEL", "account-enabled-model")

    assert resolve_groq_model() == "account-enabled-model"


def test_cli_model_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_MODEL", "environment-model")

    assert resolve_groq_model("cli-model") == "cli-model"


def test_response_and_token_usage_are_extracted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client, create = make_mock_client()
    provider = GroqProvider(
        model_name="openai/gpt-oss-20b",
        max_tokens=256,
        client=client,
    )

    result = provider.generate("unchanged debugging prompt")

    create.assert_called_once_with(
        messages=[{"role": "user", "content": "unchanged debugging prompt"}],
        model="openai/gpt-oss-20b",
        temperature=0.0,
        max_completion_tokens=256,
        stream=True,
    )
    assert result.text == "Evidence considered:\n- return ignored"
    assert "private reasoning" not in result.text
    assert result.provider_name == "Groq"
    assert result.model_name == "openai/gpt-oss-20b"
    assert result.input_tokens == 120
    assert result.output_tokens == 30
    assert result.total_tokens == 155


def test_streamed_text_callback_receives_each_visible_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client, _ = make_mock_client()
    provider = GroqProvider(client=client)
    received: list[str] = []

    provider.generate("debug", on_text=received.append)

    assert received == ["Evidence ", "considered:\n- return ignored"]


def test_blank_prompt_is_rejected_without_calling_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client, create = make_mock_client()
    provider = GroqProvider(client=client)

    with pytest.raises(ValueError, match="prompt must not be empty"):
        provider.generate("  \n")

    create.assert_not_called()


def test_empty_stream_response_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    create = Mock(
        return_value=iter(
            [
                SimpleNamespace(
                    choices=[],
                    x_groq=SimpleNamespace(usage=None),
                )
            ]
        )
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    provider = GroqProvider(client=client)

    with pytest.raises(ValueError, match="Groq returned an empty response"):
        provider.generate("debug")


def test_missing_usage_metadata_leaves_token_counts_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="answer"))],
        x_groq=None,
    )
    create = Mock(return_value=iter([chunk]))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = GroqProvider(client=client).generate("debug")

    assert result.text == "answer"
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.total_tokens is None

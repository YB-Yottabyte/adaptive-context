from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from providers.groq_provider import (
    DEFAULT_GROQ_MODEL,
    GroqProvider,
    is_gpt_oss_model,
    resolve_groq_model,
)


class MockGroqStream:
    """Iterable response exposing only the status used by diagnostics."""

    def __init__(self, chunks: list[SimpleNamespace], status_code: int = 200) -> None:
        self.chunks = chunks
        self.status_code = status_code
        self.api_key = "secret-that-must-never-be-rendered"

    def __iter__(self) -> Iterator[SimpleNamespace]:
        return iter(self.chunks)


def make_client_with_stream(stream: object) -> SimpleNamespace:
    create = Mock(return_value=stream)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
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


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("openai/gpt-oss-20b", True),
        ("GPT-OSS-120B", True),
        ("llama-3.3-70b-versatile", False),
    ],
)
def test_gpt_oss_model_detection(model_name: str, expected: bool) -> None:
    assert is_gpt_oss_model(model_name) is expected


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
        reasoning_effort="low",
    )
    assert result.text == "Evidence considered:\n- return ignored"
    assert "private reasoning" not in result.text
    assert result.provider_name == "Groq"
    assert result.model_name == "openai/gpt-oss-20b"
    assert result.input_tokens == 120
    assert result.output_tokens == 30
    assert result.total_tokens == 155


def test_non_gpt_oss_request_does_not_add_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client, create = make_mock_client()
    provider = GroqProvider(model_name="llama-3.3-70b-versatile", client=client)

    provider.generate("debug")

    request_options = create.call_args.kwargs
    assert request_options["max_completion_tokens"] == 512
    assert "max_tokens" not in request_options
    assert "reasoning_effort" not in request_options


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
    stream = MockGroqStream(
        [SimpleNamespace(choices=[], x_groq=SimpleNamespace(usage=None))]
    )
    provider = GroqProvider(client=make_client_with_stream(stream))

    with pytest.raises(ValueError, match="Groq returned no choices") as error:
        provider.generate("debug")

    message = str(error.value)
    assert "api_status=200" in message
    assert "chunks=1" in message
    assert "chunks_with_choices=0" in message
    assert "choices=0" in message
    assert "finish_reason=not provided" in message
    assert "secret-that-must-never-be-rendered" not in message


def test_empty_message_content_reports_finish_reason_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    stream = MockGroqStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None),
                        finish_reason="stop",
                    )
                ],
                x_groq=None,
            )
        ],
        status_code=201,
    )
    provider = GroqProvider(client=make_client_with_stream(stream))

    with pytest.raises(ValueError, match="message.content was empty") as error:
        provider.generate("debug")

    message = str(error.value)
    assert "reasoning=absent" in message
    assert "finish_reason=stop" in message
    assert "api_status=201" in message


def test_reasoning_without_final_content_is_reported_without_exposing_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    private_reasoning = "private chain of thought that must not be rendered"
    stream = MockGroqStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            reasoning=private_reasoning,
                        ),
                        finish_reason="length",
                    )
                ],
                x_groq=SimpleNamespace(
                    usage=SimpleNamespace(
                        prompt_tokens=400,
                        completion_tokens=512,
                        total_tokens=912,
                    )
                ),
            )
        ]
    )
    provider = GroqProvider(client=make_client_with_stream(stream))

    with pytest.raises(
        ValueError,
        match="reasoning, but message.content was empty",
    ) as error:
        provider.generate("debug")

    message = str(error.value)
    assert f"reasoning=present ({len(private_reasoning)} chars)" in message
    assert "finish_reason=length" in message
    assert "usage_tokens=input:400,output:512,total:912" in message
    assert private_reasoning not in message


def test_malformed_stream_response_has_safe_structural_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    stream = MockGroqStream([SimpleNamespace(x_groq=None)])
    provider = GroqProvider(client=make_client_with_stream(stream))

    with pytest.raises(ValueError, match="malformed streamed response") as error:
        provider.generate("debug")

    message = str(error.value)
    assert "malformed_chunks=1" in message
    assert "api_status=200" in message
    assert "secret-that-must-never-be-rendered" not in message


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

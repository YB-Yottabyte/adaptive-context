"""Meaningful Groq request and response-failure coverage."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from providers.groq_provider import GroqProvider


class MockGroqStream:
    def __init__(self, chunks: list[SimpleNamespace], status_code: int = 200) -> None:
        self.chunks = chunks
        self.status_code = status_code
        self.api_key = "secret-that-must-never-be-rendered"

    def __iter__(self) -> Iterator[SimpleNamespace]:
        return iter(self.chunks)


def make_client(stream: object) -> tuple[SimpleNamespace, Mock]:
    create = Mock(return_value=stream)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client, create


def test_gpt_oss_request_uses_low_reasoning_and_max_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Diagnosis"))],
            x_groq=SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=120,
                    completion_tokens=30,
                    total_tokens=150,
                )
            ),
        )
    ]
    client, create = make_client(iter(chunks))
    provider = GroqProvider(
        model_name="openai/gpt-oss-20b",
        max_tokens=4096,
        client=client,
    )

    result = provider.generate("Debug this repository.")

    create.assert_called_once_with(
        messages=[{"role": "user", "content": "Debug this repository."}],
        model="openai/gpt-oss-20b",
        temperature=0.0,
        max_completion_tokens=4096,
        stream=True,
        reasoning_effort="low",
    )
    assert "max_tokens" not in create.call_args.kwargs
    assert result.text == "Diagnosis"
    assert result.total_tokens == 150


def test_empty_response_reports_safe_structural_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    stream = MockGroqStream(
        [SimpleNamespace(choices=[], x_groq=SimpleNamespace(usage=None))]
    )
    client, _ = make_client(stream)

    with pytest.raises(ValueError, match="Groq returned no choices") as error:
        GroqProvider(client=client).generate("Debug this repository.")

    message = str(error.value)
    assert "api_status=200" in message
    assert "chunks=1" in message
    assert "choices=0" in message
    assert "secret-that-must-never-be-rendered" not in message


def test_reasoning_without_final_content_has_actionable_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    private_reasoning = "private chain of thought"
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
    client, _ = make_client(stream)

    with pytest.raises(
        ValueError,
        match="reasoning, but message.content was empty",
    ) as error:
        GroqProvider(client=client).generate("Debug this repository.")

    message = str(error.value)
    assert "finish_reason=length" in message
    assert "usage_tokens=input:400,output:512,total:912" in message
    assert private_reasoning not in message

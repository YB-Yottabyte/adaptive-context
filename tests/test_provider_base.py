from providers.base import GenerationResult


def test_total_tokens_uses_provider_reported_total_when_available() -> None:
    result = GenerationResult(
        text="done",
        input_tokens=10,
        output_tokens=5,
        reported_total_tokens=20,
    )

    assert result.total_tokens == 20


def test_total_tokens_is_calculated_from_input_and_output_counts() -> None:
    result = GenerationResult(text="done", input_tokens=10, output_tokens=5)

    assert result.total_tokens == 15


def test_total_tokens_is_unknown_when_either_count_is_missing() -> None:
    result = GenerationResult(text="done", input_tokens=10)

    assert result.total_tokens is None

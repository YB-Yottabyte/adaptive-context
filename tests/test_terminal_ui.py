from io import StringIO
from threading import Event

import pytest
from rich.console import Console

from providers.base import GenerationResult
from retrieval import CodeChunk
from semantic_retrieval import ChromaRetriever
from terminal_ui import RESPONSE_FRAME_INTERVAL_SECONDS, TerminalUI


def make_terminal(
    output: StringIO,
    *,
    clock=lambda: 0.0,
    sleeper=lambda _: None,
) -> TerminalUI:
    return TerminalUI(
        Console(file=output, color_system=None, width=72),
        clock=clock,
        sleeper=sleeper,
    )


def structured_response() -> str:
    return (
        "Evidence considered:\n"
        "- test_payment.py defines the expected result.\n"
        "- payment.py ignores the returned value.\n\n"
        "Conclusion:\ncheckout() returns the original price.\n\n"
        "Relevant file:\npayment.py\n\n"
        "Suggested change:\n"
        "def checkout(price, discount):\n"
        "    return apply_discount(price, discount)\n\n"
        "Explanation:\nThe helper already computes the correct value."
    )


def test_terminal_output_remains_compact_and_shows_changed_lines() -> None:
    output = StringIO()
    terminal = make_terminal(output)
    terminal.display_startup("groq", "openai/gpt-oss-20b")
    terminal.display_retrieval_configuration(ChromaRetriever(), 3)
    terminal.display_retrieved_context(
        [
            CodeChunk("test_payment.py", "test", 1, 8),
            CodeChunk("payment.py", "checkout", 1, 7),
            CodeChunk("discount.py", "discount", 1, 2),
        ]
    )
    terminal.display_model_response(
        GenerationResult(text=structured_response()),
        selected_chunks=[
            CodeChunk(
                "payment.py",
                "def checkout(price, discount):\n"
                "    apply_discount(price, discount)\n"
                "    return price",
                1,
                3,
            )
        ],
    )

    rendered = output.getvalue()
    assert ">_ Adaptive Context  (static RAG baseline)" in rendered
    assert "model:    GPT-OSS 20B via Groq" in rendered
    assert "directory:" not in rendered
    assert "Retrieval: ChromaDB semantic search · top-k = 3" in rendered
    assert "● Retrieved 3 relevant chunks" in rendered
    assert "├─ test_payment.py:1-8" in rendered
    assert "└─ discount.py:1-2" in rendered
    assert "● Diagnosis points to payment.py" in rendered
    assert "├─ test_payment.py defines the expected result." in rendered
    assert "└─ payment.py ignores the returned value." in rendered
    assert "2 │     apply_discount(price, discount)" in rendered
    assert "3 │     return price" in rendered
    assert "2 └────→     return apply_discount(price, discount)" in rendered
    assert "Current code" not in rendered


def test_partial_suggestion_waits_before_rendering_diff() -> None:
    partial_response = (
        "Evidence considered:\n- module.py returns False.\n\n"
        "Conclusion:\nThe return value is wrong.\n\n"
        "Relevant file:\nmodule.py\n\n"
        "Suggested change:\ndef broken():"
    )
    output = StringIO()
    terminal = make_terminal(output)

    terminal.console.print(
        terminal.response_renderable(
            partial_response,
            final=False,
            selected_chunks=[
                CodeChunk("module.py", "def broken():\n    return False", 1, 2)
            ],
        )
    )

    rendered = output.getvalue()
    assert "Preparing suggested change…" in rendered
    assert "└────→" not in rendered


def test_fenced_suggestion_uses_the_same_line_diff_presentation() -> None:
    output = StringIO()
    terminal = make_terminal(output)
    response = structured_response().replace(
        "Suggested change:\ndef checkout(price, discount):\n"
        "    return apply_discount(price, discount)",
        "Suggested change:\n```python\n"
        "def checkout(price, discount):\n"
        "    return apply_discount(price, discount)\n```",
    )

    terminal.display_model_response(
        GenerationResult(text=response),
        selected_chunks=[
            CodeChunk(
                "payment.py",
                "def checkout(price, discount):\n"
                "    apply_discount(price, discount)\n"
                "    return price",
                1,
                3,
            )
        ],
    )

    rendered = output.getvalue()
    assert "Suggested change" in rendered
    assert "Suggested replacement" not in rendered
    assert "2 └────→     return apply_discount(price, discount)" in rendered


def test_unchanged_suggestion_is_reported_but_still_shown() -> None:
    output = StringIO()
    terminal = make_terminal(output)
    response = structured_response().replace(
        "Relevant file:\npayment.py",
        "Relevant file:\ndiscount.py",
    ).replace(
        "def checkout(price, discount):\n"
        "    return apply_discount(price, discount)",
        "```python\n"
        "def apply_discount(price, discount):\n"
        "    return price * (1 - discount)\n```",
    )

    terminal.display_model_response(
        GenerationResult(text=response),
        selected_chunks=[
            CodeChunk(
                "discount.py",
                "def apply_discount(price, discount):\n"
                "    return price * (1 - discount)",
                1,
                2,
            )
        ],
    )

    rendered = output.getvalue()
    assert "● Diagnosis incomplete" in rendered
    assert "repeated the current implementation" in rendered
    assert "suggestion is shown below for transparency" in rendered
    assert "Suggested change" in rendered
    assert "def apply_discount(price, discount):" in rendered
    assert "return price * (1 - discount)" in rendered
    assert "└────→" not in rendered


def test_incomplete_final_response_is_not_presented_as_a_valid_diagnosis() -> None:
    output = StringIO()
    terminal = make_terminal(output)
    incomplete_response = (
        "Evidence considered:\n"
        "- test_payment.py asserts checkout(100, 0.20) == 80.0\n\n"
        "Conclusion:\nEvidence considered:\n"
        "- test_payment.py asserts checkout(100, 0.20) == 80.0"
    )

    terminal.display_model_response(GenerationResult(text=incomplete_response))

    rendered = output.getvalue()
    assert "● Diagnosis incomplete" in rendered
    assert "The model did not return a complete file-and-fix suggestion." in rendered
    assert "Try increasing Top-K (for example, --top-k 3)." in rendered
    assert "Diagnosis suggests a likely issue" not in rendered
    assert "Conclusion" not in rendered


def test_top_k_adjustment_reports_available_chunk_limit() -> None:
    output = StringIO()
    terminal = make_terminal(output)

    terminal.display_top_k_adjustment(requested=5, available=3)

    rendered = " ".join(output.getvalue().split())
    assert (
        "Requested top-k = 5, but only 3 repository chunks are available; "
        "using top-k = 3."
    ) in rendered


def test_footer_shows_tokens_and_actual_execution_time() -> None:
    output = StringIO()
    terminal = make_terminal(output)
    result = GenerationResult(
        text="done",
        input_tokens=345,
        output_tokens=243,
        reported_total_tokens=588,
    )

    terminal.display_footer(result, 1.84)

    rendered = output.getvalue()
    assert "345 input · 243 output · 588 total tokens" in rendered
    assert "Completed in 1.8s" in rendered


def test_timed_stage_measures_the_action() -> None:
    times = iter((10.0, 10.02))
    terminal = make_terminal(StringIO(), clock=lambda: next(times))

    result, elapsed = terminal.run_timed_stage("Working...", lambda: "result")

    assert result == "result"
    assert elapsed == pytest.approx(0.02)


def test_generate_response_measures_and_reveals_chunks_during_generation() -> None:
    frame_rendered = Event()

    class StreamingProvider:
        def generate(self, prompt: str, *, on_text=None) -> GenerationResult:
            assert prompt == "debug this"
            assert callable(on_text)
            on_text(structured_response())
            assert frame_rendered.wait(timeout=1)
            return GenerationResult(text=structured_response())

    times = iter((10.0, 10.3, 10.8))
    sleep_calls: list[float] = []
    output = StringIO()

    def record_frame(duration: float) -> None:
        sleep_calls.append(duration)
        frame_rendered.set()

    terminal = make_terminal(
        output,
        clock=lambda: next(times),
        sleeper=record_frame,
    )

    result, first_token, model_total = terminal.generate_response(
        StreamingProvider(),  # type: ignore[arg-type]
        "debug this",
        selected_chunks=[
            CodeChunk(
                "payment.py",
                "def checkout(price, discount):\n"
                "    apply_discount(price, discount)\n"
                "    return price",
                1,
                3,
            )
        ],
    )

    assert result.text == structured_response()
    assert first_token == pytest.approx(0.3)
    assert model_total == pytest.approx(0.8)
    assert len(sleep_calls) > 1
    assert set(sleep_calls) == {RESPONSE_FRAME_INTERVAL_SECONDS}
    assert "● Diagnosis points to payment.py" in output.getvalue()

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
    assert "payment.py" in rendered
    assert "├─ test_payment.py defines the expected result." in rendered
    assert "└─ payment.py ignores the returned value." in rendered
    assert "2 │     apply_discount(price, discount)" in rendered
    assert "3 │     return price" in rendered
    assert "2 └────→     return apply_discount(price, discount)" in rendered
    assert "Current code" not in rendered


def test_exact_replacement_in_real_source_window_uses_diff_rendering() -> None:
    output = StringIO()
    terminal = make_terminal(output)
    response = (
        structured_response()
        .replace(
            "payment.py",
            "src/requests/sessions.py",
        )
        .replace(
            "def checkout(price, discount):\n"
            "    return apply_discount(price, discount)",
            "def rebuild_auth(prepared_request, response):\n"
            "    return get_netrc_auth(prepared_request.url)",
        )
    )

    terminal.display_model_response(
        GenerationResult(text=response),
        selected_chunks=[
            CodeChunk(
                "src/requests/sessions.py",
                "    return previous_helper_result\n\n"
                "def rebuild_auth(prepared_request, response):\n"
                "    return response.request.headers.get('Authorization')\n\n"
                "def rebuild_proxies(prepared_request, proxies):\n"
                "    return proxies",
                240,
                246,
            )
        ],
    )

    rendered = output.getvalue()
    assert "src/requests/sessions.py" in rendered
    assert "243 │     return response.request.headers.get('Authorization')" in rendered
    assert "243 └────→     return get_netrc_auth(prepared_request.url)" in rendered


def test_unmappable_real_repository_suggestion_uses_plain_code_fallback() -> None:
    output = StringIO()
    terminal = make_terminal(output)
    response = (
        structured_response()
        .replace(
            "payment.py",
            "src/requests/sessions.py",
        )
        .replace(
            "def checkout(price, discount):\n"
            "    return apply_discount(price, discount)",
            "def update_auth(prepared_request):\n"
            "    return get_netrc_auth(prepared_request.url)",
        )
    )

    terminal.display_model_response(
        GenerationResult(text=response),
        selected_chunks=[
            CodeChunk(
                "src/requests/sessions.py",
                "def rebuild_auth(prepared_request, response):\n"
                "    return response.request.headers.get('Authorization')\n\n"
                "def rebuild_proxies(prepared_request, proxies):\n"
                "    return proxies",
                241,
                245,
            )
        ],
    )

    rendered = output.getvalue()
    assert "def update_auth(prepared_request):" in rendered
    assert "return get_netrc_auth(prepared_request.url)" in rendered
    assert "└────→" not in rendered
    assert "response.request.headers.get" not in rendered


def test_labeled_exact_source_replacement_uses_repository_diff() -> None:
    output = StringIO()
    terminal = make_terminal(output)
    response = (
        "Evidence considered:\n- models.py checks Iterable.\n\n"
        "Conclusion:\nStream detection is too strict.\n\n"
        "Relevant file:\nsrc/requests/models.py\n\n"
        "Suggested change:\n"
        "# Original code:\n"
        "# if isinstance(data, Iterable) and not isinstance(\n"
        "#     data, (str, bytes, list, tuple, Mapping)\n"
        "# ):\n"
        "# Updated code:\n"
        'is_iterable = isinstance(data, Iterable) or hasattr(data, "__iter__")\n'
        "if is_iterable and not isinstance(\n"
        "    data, (str, bytes, list, tuple, Mapping)\n"
        "):\n\n"
        "Explanation:\nAccept delegated iterators."
    )

    terminal.display_model_response(
        GenerationResult(text=response),
        selected_chunks=[
            CodeChunk(
                "src/requests/models.py",
                '        body = body.encode("utf-8")\n\n'
                "        if isinstance(data, Iterable) and not isinstance(\n"
                "            data, (str, bytes, list, tuple, Mapping)\n"
                "        ):\n"
                "            length = super_len(data)",
                597,
                602,
            )
        ],
    )

    rendered = output.getvalue()
    assert "src/requests/models.py" in rendered
    assert "599 │         if isinstance(data, Iterable) and not isinstance(" in rendered
    assert "599 └────→         is_iterable = isinstance(data, Iterable)" in rendered
    assert 'hasattr(data, "__iter__")' in rendered


def test_labeled_simplified_change_uses_unique_retrieved_source_line() -> None:
    output = StringIO()
    terminal = make_terminal(output)
    response = (
        "Evidence considered:\n- models.py checks Iterable.\n\n"
        "Conclusion:\nStream detection is too strict.\n\n"
        "Relevant file:\nsrc/requests/models.py\n\n"
        "Suggested change:\n"
        "# Original snippet (simplified)\n"
        "# if isinstance(data, Iterable):\n"
        "#     # streamed body\n"
        "# else:\n"
        "#     # raw body\n\n"
        "# Updated snippet\n"
        'if hasattr(data, "__iter__"):\n'
        "    # streamed body\n"
        "else:\n"
        "    # raw body\n\n"
        "Explanation:\nAccept delegated iterators."
    )

    terminal.display_model_response(
        GenerationResult(text=response),
        selected_chunks=[
            CodeChunk(
                "src/requests/models.py",
                "        if isinstance(data, Iterable) and not isinstance(\n"
                "            data, (str, bytes, list, tuple, Mapping)\n"
                "        ):\n"
                "            length = super_len(data)",
                599,
                602,
            )
        ],
    )

    rendered = output.getvalue()
    assert "599 │         if isinstance(data, Iterable) and not isinstance(" in rendered
    assert (
        '599 └────→         if hasattr(data, "__iter__") and not isinstance('
        in rendered
    )
    assert "# Original snippet (simplified)" not in rendered


def test_labeled_change_outside_retrieved_lines_uses_plain_fallback() -> None:
    output = StringIO()
    terminal = make_terminal(output)
    response = (
        "Evidence considered:\n- models.py checks Iterable.\n\n"
        "Conclusion:\nStream detection is too strict.\n\n"
        "Relevant file:\nsrc/requests/models.py\n\n"
        "Suggested change:\n"
        "# Original code (simplified):\n"
        "# if isinstance(data, Iterable):\n"
        "#     # streamed body\n"
        "# else:\n"
        "#     # raw body\n\n"
        "# Updated code:\n"
        'if hasattr(data, "__iter__"):\n'
        "    # streamed body\n"
        "else:\n"
        "    # raw body\n\n"
        "Explanation:\nAccept delegated iterators."
    )

    terminal.display_model_response(
        GenerationResult(text=response),
        selected_chunks=[
            CodeChunk(
                "src/requests/models.py",
                "        ):\n"
                "            try:\n"
                "                length = super_len(data)",
                601,
                603,
            )
        ],
    )

    rendered = output.getvalue()
    assert "# Original code (simplified):" in rendered
    assert 'if hasattr(data, "__iter__"):' in rendered
    assert "└────→" not in rendered


def test_prefixed_simplified_change_uses_actual_repository_source() -> None:
    output = StringIO()
    terminal = make_terminal(output)
    response = (
        "Evidence considered:\n- models.py checks Iterable.\n\n"
        "Conclusion:\nStream detection is too strict.\n\n"
        "Relevant file:\nsrc/requests/models.py\n\n"
        "Suggested change:\n"
        "# Detect streamed bodies\n"
        "-        if isinstance(data, Iterable):\n"
        '+        if hasattr(data, "__iter__"):\n'
        "             # Record the current file position before reading.\n\n"
        "Explanation:\nAccept delegated iterators."
    )

    terminal.display_model_response(
        GenerationResult(text=response),
        selected_chunks=[
            CodeChunk(
                "src/requests/models.py",
                "        if isinstance(data, Iterable) and not isinstance(\n"
                "            data, (str, bytes, list, tuple, Mapping)\n"
                "        ):\n"
                "            length = super_len(data)",
                599,
                602,
            )
        ],
    )

    rendered = output.getvalue()
    assert "src/requests/models.py" in rendered
    assert "599 │         if isinstance(data, Iterable) and not isinstance(" in rendered
    assert (
        '599 └────→         if hasattr(data, "__iter__") and not isinstance('
        in rendered
    )
    assert "# Detect streamed bodies" not in rendered


def test_updated_snippet_with_unique_context_uses_repository_diff() -> None:
    output = StringIO()
    terminal = make_terminal(output)
    response = (
        "Evidence considered:\n- models.py checks tell().\n\n"
        "Conclusion:\nStream detection is too strict.\n\n"
        "Relevant file:\nsrc/requests/models.py\n\n"
        "Suggested change:\n"
        'if getattr(body, "tell", None) is not None or hasattr(body, "__iter__"):\n'
        "            # Record the current file position before reading.\n"
        "            # This will allow us to rewind a file in the event\n"
        "            # of a redirect.\n"
        "            try:\n"
        "                self._body_position = body.tell()\n"
        "            except OSError:\n"
        "                self._body_position = object()\n\n"
        "Explanation:\nAccept delegated iterators."
    )
    full_source = (
        "        if getattr(body, \"tell\", None) is not None:\n"
        "            # Record the current file position before reading.\n"
        "            # This will allow us to rewind a file in the event\n"
        "            # of a redirect.\n"
        "            try:\n"
        "                self._body_position = body.tell()\n"
        "            except OSError:\n"
        "                self._body_position = object()\n"
    )

    terminal.display_model_response(
        GenerationResult(text=response),
        selected_chunks=[
            CodeChunk(
                "src/requests/models.py",
                full_source,
                609,
                616,
            )
        ],
    )

    rendered = output.getvalue()
    assert "src/requests/models.py" in rendered
    assert '609 │         if getattr(body, "tell", None) is not None:' in rendered
    assert '609 └────→         if getattr(body, "tell", None) is not None or' in rendered
    assert 'hasattr(body, "__iter__"):' in rendered


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

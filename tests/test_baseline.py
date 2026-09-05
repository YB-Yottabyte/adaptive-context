import argparse
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from baseline import (
    RESPONSE_FRAME_INTERVAL_SECONDS,
    _changed_line_numbers,
    _model_response_renderable,
    _run_timed_stage,
    generate_streaming_response,
    get_available_providers,
    main,
    parse_debugging_summary,
    print_model_response,
    print_retrieved_context,
    print_run_footer,
    print_startup,
    read_bug_report,
    resolve_test_case,
    run_baseline,
    select_provider,
    select_top_k_value,
    validate_provider,
)
from providers.base import GenerationResult
from retrieval import CodeChunk


def test_read_bug_report(tmp_path: Path) -> None:
    (tmp_path / "bug_report.txt").write_text("Something broke.\n", encoding="utf-8")

    assert read_bug_report(tmp_path) == "Something broke."


def test_missing_test_case_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Test case directory does not exist"):
        read_bug_report(missing)


def test_test_case_path_must_be_a_directory(tmp_path: Path) -> None:
    test_case_file = tmp_path / "test_case.py"
    test_case_file.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="Test case path is not a directory"):
        read_bug_report(test_case_file)


def test_missing_bug_report(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing bug report"):
        read_bug_report(tmp_path)


def test_empty_bug_report_is_rejected(tmp_path: Path) -> None:
    report = tmp_path / "bug_report.txt"
    report.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(ValueError, match="Bug report is empty"):
        read_bug_report(tmp_path)


def test_short_test_case_name_resolves_under_repository_test_cases() -> None:
    resolved = resolve_test_case("payment_bug")

    assert resolved.name == "payment_bug"
    assert resolved.parent.name == "test_cases"
    assert resolved.is_absolute()


def test_explicit_test_case_path_is_preserved(tmp_path: Path) -> None:
    assert resolve_test_case(tmp_path) == tmp_path


def test_startup_and_model_output_remain_compact() -> None:
    output = StringIO()
    console = Console(file=output, color_system=None, width=72)

    print_startup("groq", "openai/gpt-oss-20b", console=console)
    print_retrieved_context(
        ["test_payment.py", "payment.py", "discount.py"],
        console=console,
    )
    print_model_response(
        GenerationResult(
            text=(
                "Evidence considered:\n"
                "- test_payment.py defines the expected result.\n"
                "- payment.py ignores the returned value.\n\n"
                "Conclusion:\ncheckout() returns the original price.\n\n"
                "Relevant file:\npayment.py\n\n"
                "Suggested change:\n"
                "def checkout(price, discount):\n"
                "    return apply_discount(price, discount)\n\n"
                "Explanation:\nThe helper already computes the correct value."
            ),
            input_tokens=10,
            output_tokens=8,
        ),
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
        console=console,
    )

    rendered = output.getvalue()
    assert "Context Debugging Baseline" in rendered
    assert "Using Groq — GPT-OSS 20B" in rendered
    assert "I'll trace the reported behavior" in rendered
    assert "payment_bug" not in rendered
    assert "top-k" not in rendered
    assert "● Retrieved 3 relevant files" in rendered
    assert "├─ test_payment.py" in rendered
    assert "└─ discount.py" in rendered
    assert "● Diagnosis points to payment.py" in rendered
    assert "Evidence" in rendered
    assert "├─ test_payment.py defines the expected result." in rendered
    assert "└─ payment.py ignores the returned value." in rendered
    assert "Suggested change" in rendered
    assert "│" in rendered
    assert "2 │     apply_discount(price, discount)" in rendered
    assert "3 │     return price" in rendered
    assert "2 └────→     return apply_discount(price, discount)" in rendered
    assert "Current code" not in rendered
    assert "Suggested replacement" not in rendered


def test_parse_debugging_summary_uses_public_sections() -> None:
    summary = parse_debugging_summary(
        "Evidence considered:\n• one\n• two\n\n"
        "Conclusion:\nA concise conclusion.\n\n"
        "Relevant file:\nmodule.py\n\n"
        "Suggested change:\nreturn fixed\n\n"
        "Explanation:\nThis uses observable evidence."
    )

    assert summary.evidence == ("one", "two")
    assert summary.conclusion == "A concise conclusion."
    assert summary.relevant_file == "module.py"


def test_parse_debugging_summary_falls_back_to_unstructured_response() -> None:
    response = "The checkout function ignores the helper's return value."

    summary = parse_debugging_summary(response)

    assert summary.evidence == ()
    assert summary.conclusion == response
    assert summary.relevant_file == ""
    assert summary.suggested_change == ""
    assert summary.explanation == ""


def test_parse_debugging_summary_can_disable_unstructured_fallback() -> None:
    summary = parse_debugging_summary("partial response", fallback_to_response=False)

    assert summary.conclusion == ""


def test_changed_line_numbers_identify_only_replaced_function_lines() -> None:
    current = (
        "def checkout(price, discount):\n"
        "    apply_discount(price, discount)\n"
        "    return price"
    )
    suggested = (
        "def checkout(price, discount):\n"
        "    return apply_discount(price, discount)"
    )

    current_changed, suggested_changed = _changed_line_numbers(current, suggested)

    assert current_changed == {2, 3}
    assert suggested_changed == {2}


def test_partial_suggestion_waits_for_complete_section_before_rendering_diff() -> None:
    partial_response = (
        "Evidence considered:\n- module.py returns False.\n\n"
        "Conclusion:\nThe return value is wrong.\n\n"
        "Relevant file:\nmodule.py\n\n"
        "Suggested change:\ndef broken():"
    )
    selected_chunks = [
        CodeChunk("module.py", "def broken():\n    return False", 1, 2)
    ]
    output = StringIO()
    console = Console(file=output, color_system=None)

    console.print(
        _model_response_renderable(
            partial_response,
            final=False,
            selected_chunks=selected_chunks,
        )
    )

    rendered = output.getvalue()
    assert "Preparing suggested change…" in rendered
    assert "└────→" not in rendered
    assert "Suggested replacement" not in rendered


def test_run_baseline_uses_transient_stages_without_completion_clutter(
    tmp_path: Path,
) -> None:
    (tmp_path / "bug_report.txt").write_text("A function fails.", encoding="utf-8")
    (tmp_path / "module.py").write_text(
        "def broken():\n    return False\n",
        encoding="utf-8",
    )

    class Provider:
        def generate(
            self,
            prompt: str,
            *,
            on_text: object = None,
        ) -> GenerationResult:
            assert "module.py" in prompt
            assert callable(on_text)
            on_text("Evidence")
            return GenerationResult(
                text=(
                    "Evidence considered:\n- module.py returns False.\n\n"
                    "Conclusion:\nbroken() returns the wrong value.\n\n"
                    "Relevant file:\nmodule.py\n\n"
                    "Suggested change:\n"
                    "def broken():\n    return True\n\n"
                    "Explanation:\nThe expected value is returned."
                ),
                input_tokens=12,
                output_tokens=9,
            )

    output = StringIO()
    console = Console(file=output, color_system=None, width=72)
    args = argparse.Namespace(
        test_case=tmp_path,
        provider="groq",
        model="openai/gpt-oss-20b",
        top_k=1,
        max_tokens=100,
    )

    with (
        patch("baseline.create_provider", return_value=Provider()),
        patch("baseline.sleep"),
    ):
        timings = run_baseline(args, console=console)

    rendered = output.getvalue()
    assert "Using Groq — GPT-OSS 20B" in rendered
    assert "● Retrieved 1 relevant file" in rendered
    assert "● Diagnosis points to module.py" in rendered
    assert "Suggested change" in rendered
    assert "│" in rendered
    assert "2 │     return False" in rendered
    assert "2 └────→     return True" in rendered
    assert "Completed in " in rendered
    assert "Test case loaded" not in rendered
    assert "Context prepared" not in rendered
    assert "First token received" not in rendered
    assert "top-k" not in rendered
    assert timings.total >= 0


def test_run_baseline_rejects_repository_without_python_before_provider_creation(
    tmp_path: Path,
) -> None:
    (tmp_path / "bug_report.txt").write_text("A function fails.", encoding="utf-8")
    args = argparse.Namespace(
        test_case=tmp_path,
        provider="groq",
        model="openai/gpt-oss-20b",
        top_k=1,
        max_tokens=100,
    )
    console = Console(file=StringIO(), color_system=None)

    with (
        patch("baseline.create_provider") as create_provider,
        pytest.raises(ValueError, match="No Python source files found"),
    ):
        run_baseline(args, console=console)

    create_provider.assert_not_called()


def test_run_footer_shows_tokens_and_actual_execution_time() -> None:
    output = StringIO()
    console = Console(file=output, color_system=None, width=72)
    result = GenerationResult(
        text="done",
        input_tokens=345,
        output_tokens=243,
        reported_total_tokens=588,
    )
    print_run_footer(result, 1.84, console=console)

    rendered = output.getvalue()
    assert "Retrieval" not in rendered
    assert "Context build" not in rendered
    assert "First token" not in rendered
    assert "Model total" not in rendered
    assert "────" not in rendered
    assert "345 input · 243 output · 588 total tokens" in rendered
    assert "Completed in 1.8s" in rendered


def test_timed_stage_measures_the_action() -> None:
    console = Console(file=StringIO(), color_system=None)
    with patch("baseline.perf_counter", side_effect=(10.0, 10.02)):
        result, elapsed = _run_timed_stage(
            "Working...",
            lambda: "result",
            console=console,
        )

    assert result == "result"
    assert elapsed == pytest.approx(0.02)


def test_get_available_providers_checks_configuration_and_dependencies() -> None:
    environment = {
        "GROQ_API_KEY": "configured",
        "CONTEXT_DEBUG_MODEL": "mlx-community/test-model",
    }

    with (
        patch("baseline._module_available", return_value=True),
        patch("baseline._local_runtime_available", return_value=True),
    ):
        assert get_available_providers(environment=environment) == ("groq", "local")


def test_get_available_providers_excludes_unconfigured_providers() -> None:
    with (
        patch("baseline._module_available", return_value=True),
        patch("baseline._local_runtime_available", return_value=False),
    ):
        assert get_available_providers(environment={}) == ()


def test_get_available_providers_recognizes_default_local_model() -> None:
    with patch("baseline._local_runtime_available", return_value=True):
        assert get_available_providers(environment={}) == ("local",)


def test_get_available_providers_accepts_explicit_local_model() -> None:
    with patch("baseline._local_runtime_available", return_value=True):
        assert get_available_providers(
            environment={},
            local_model="mlx-community/test-model",
        ) == ("local",)


def test_get_available_providers_requires_provider_dependency() -> None:
    with patch("baseline._module_available", return_value=False):
        assert get_available_providers(environment={"GROQ_API_KEY": "set"}) == ()


def test_validate_provider_rejects_explicit_unavailable_provider() -> None:
    validate_provider("groq", ("groq",))

    with pytest.raises(
        ValueError,
        match="Local Hugging Face provider is unavailable",
    ):
        validate_provider("local", ("groq",))


def test_select_provider_preserves_available_cli_override() -> None:
    assert select_provider("groq", ("groq",), interactive=True) == "groq"


def test_select_provider_automatically_uses_only_available_provider() -> None:
    with patch("baseline.questionary.select") as select:
        selected = select_provider(
            None,
            ("groq",),
            environment={},
            interactive=True,
        )

    assert selected == "groq"
    select.assert_not_called()


def test_select_provider_requires_override_for_noninteractive_multiple() -> None:
    with pytest.raises(ValueError, match="Multiple providers are available"):
        select_provider(None, ("groq", "local"), interactive=False)


def test_select_provider_reports_when_none_are_available() -> None:
    with pytest.raises(ValueError, match="No model providers are available"):
        select_provider(None, (), interactive=False)


def test_select_provider_uses_arrow_menu_when_interactive() -> None:
    question = Mock()
    question.ask.return_value = "local"
    with patch("baseline.questionary.select", return_value=question) as select:
        selected = select_provider(
            None,
            ("groq", "local"),
            environment={},
            interactive=True,
        )

    assert selected == "local"
    assert [choice.value for choice in select.call_args.kwargs["choices"]] == [
        "groq",
        "local",
    ]
    assert select.call_args.kwargs["pointer"] == "❯"


def test_select_provider_reports_cancelled_interactive_menu() -> None:
    question = Mock()
    question.ask.return_value = None
    with (
        patch("baseline.questionary.select", return_value=question),
        pytest.raises(ValueError, match="Model selection cancelled"),
    ):
        select_provider(
            None,
            ("groq", "local"),
            environment={},
            interactive=True,
        )


def test_select_top_k_value_preserves_explicit_experiment_value() -> None:
    with patch("baseline.questionary.text") as text_prompt:
        selected = select_top_k_value(5, interactive=True)

    assert selected == 5
    text_prompt.assert_not_called()


def test_select_top_k_value_prompts_during_interactive_run() -> None:
    question = Mock()
    question.ask.return_value = "4"
    with patch("baseline.questionary.text", return_value=question) as text_prompt:
        selected = select_top_k_value(None, interactive=True)

    assert selected == 4
    assert text_prompt.call_args.args == ("Top-K chunks to retrieve:",)
    assert callable(text_prompt.call_args.kwargs["validate"])


def test_select_top_k_value_requires_flag_for_noninteractive_run() -> None:
    with pytest.raises(ValueError, match="Pass --top-k"):
        select_top_k_value(None, interactive=False)


def test_select_top_k_value_reports_cancelled_prompt() -> None:
    question = Mock()
    question.ask.return_value = None
    with (
        patch("baseline.questionary.text", return_value=question),
        pytest.raises(ValueError, match="Top-K selection cancelled"),
    ):
        select_top_k_value(None, interactive=True)


def test_generate_streaming_response_measures_chunks_without_printing_them() -> None:
    class StreamingProvider:
        def generate(
            self,
            prompt: str,
            *,
            on_text: object = None,
        ) -> GenerationResult:
            assert prompt == "debug this"
            assert callable(on_text)
            on_text("Evidence considered:\n")
            on_text("- payment.py ignores a return value")
            return GenerationResult(text="complete", input_tokens=10, output_tokens=8)

    output = StringIO()
    console = Console(file=output, color_system=None)

    result, first_token, model_total = generate_streaming_response(
        StreamingProvider(),  # type: ignore[arg-type]
        "debug this",
        console=console,
    )

    rendered = output.getvalue()
    assert result.text == "complete"
    assert first_token <= model_total
    assert "First token received" not in rendered
    assert "Evidence considered:" not in rendered
    assert "payment.py ignores a return value" not in rendered
    assert "complete" in rendered


def test_slow_response_playback_is_excluded_from_model_timing() -> None:
    response_text = (
        "Evidence considered:\n- module.py returns False.\n\n"
        "Conclusion:\nThe return value is wrong.\n\n"
        "Relevant file:\nmodule.py\n\n"
        "Suggested change:\ndef broken():\n    return True\n\n"
        "Explanation:\nThe function now returns the expected value."
    )

    class StreamingProvider:
        def generate(
            self,
            prompt: str,
            *,
            on_text: object = None,
        ) -> GenerationResult:
            assert callable(on_text)
            on_text(response_text)
            return GenerationResult(text=response_text)

    output = StringIO()
    console = Console(file=output, color_system=None)
    selected_chunks = [
        CodeChunk("module.py", "def broken():\n    return False", 1, 2)
    ]
    with (
        patch("baseline.perf_counter", side_effect=(10.0, 10.3, 10.8)),
        patch("baseline.sleep") as sleep_mock,
    ):
        _, first_token, model_total = generate_streaming_response(
            StreamingProvider(),  # type: ignore[arg-type]
            "debug this",
            selected_chunks=selected_chunks,
            console=console,
        )

    assert first_token == pytest.approx(0.3)
    assert model_total == pytest.approx(0.8)
    assert sleep_mock.call_count > 1
    assert all(
        call.args == (RESPONSE_FRAME_INTERVAL_SECONDS,)
        for call in sleep_mock.call_args_list
    )
    rendered = output.getvalue()
    assert "● Diagnosis points to module.py" in rendered
    assert "Suggested change" in rendered
    assert "│" in rendered
    assert "2 │     return False" in rendered
    assert "2 └────→     return True" in rendered


def test_main_resolves_cli_configuration_before_running_pipeline(tmp_path: Path) -> None:
    cli_args = [
        "baseline.py",
        str(tmp_path),
        "--provider",
        "groq",
        "--model",
        "cli-model",
        "--top-k",
        "2",
        "--max-tokens",
        "64",
    ]

    with (
        patch("baseline.sys.argv", cli_args),
        patch("baseline.get_available_providers", return_value=("groq",)),
        patch("baseline.run_baseline") as run,
    ):
        main()

    args = run.call_args.args[0]
    assert args.provider == "groq"
    assert args.test_case == tmp_path
    assert args.model == "cli-model"
    assert args.top_k == 2
    assert args.max_tokens == 64


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--top-k", "0", "--top-k must be at least 1"),
        ("--max-tokens", "0", "--max-tokens must be at least 1"),
    ],
)
def test_main_rejects_non_positive_numeric_options(
    tmp_path: Path,
    option: str,
    value: str,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli_args = ["baseline.py", str(tmp_path), option, value]

    with patch("baseline.sys.argv", cli_args), pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert message in capsys.readouterr().err

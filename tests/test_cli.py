from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cli import BaselineCLI


def make_cli(*, environment: dict[str, str] | None = None) -> BaselineCLI:
    factory = Mock()
    runner = Mock()
    return BaselineCLI(
        provider_factory=factory,
        runner=runner,
        environment=environment or {},
    )


def test_parser_exposes_one_chroma_retrieval_pipeline() -> None:
    args = make_cli().build_parser().parse_args(["payment_bug"])

    assert args.test_case == "payment_bug"
    assert not hasattr(args, "retriever")


def test_short_test_case_name_resolves_under_test_cases() -> None:
    resolved = BaselineCLI.resolve_test_case("payment_bug")

    assert resolved.name == "payment_bug"
    assert resolved.parent.name == "test_cases"
    assert resolved.is_absolute()


def test_explicit_test_case_path_is_preserved(tmp_path: Path) -> None:
    assert BaselineCLI.resolve_test_case(tmp_path) == tmp_path


def test_select_provider_preserves_available_override() -> None:
    cli = make_cli()
    cli.provider_factory.validate.return_value = None

    assert cli.select_provider("groq", ("groq",), interactive=True) == "groq"
    cli.provider_factory.validate.assert_called_once_with("groq", ("groq",))


def test_select_provider_automatically_uses_only_available_provider() -> None:
    cli = make_cli()
    with patch("cli.questionary.select") as select:
        selected = cli.select_provider(None, ("groq",), interactive=True)

    assert selected == "groq"
    select.assert_not_called()


def test_select_provider_requires_override_for_noninteractive_multiple() -> None:
    with pytest.raises(ValueError, match="Multiple providers are available"):
        make_cli().select_provider(None, ("groq", "local"), interactive=False)


def test_select_provider_reports_when_none_are_available() -> None:
    with pytest.raises(ValueError, match="No model providers are available"):
        make_cli().select_provider(None, (), interactive=False)


def test_select_provider_uses_arrow_menu_when_interactive() -> None:
    question = Mock()
    question.ask.return_value = "local"
    cli = make_cli(environment={})
    with patch("cli.questionary.select", return_value=question) as select:
        selected = cli.select_provider(
            None,
            ("groq", "local"),
            interactive=True,
        )

    assert selected == "local"
    assert [choice.value for choice in select.call_args.kwargs["choices"]] == [
        "groq",
        "local",
    ]
    assert select.call_args.kwargs["pointer"] == "❯"


def test_select_provider_reports_cancelled_menu() -> None:
    question = Mock()
    question.ask.return_value = None
    with (
        patch("cli.questionary.select", return_value=question),
        pytest.raises(ValueError, match="Model selection cancelled"),
    ):
        make_cli().select_provider(
            None,
            ("groq", "local"),
            interactive=True,
        )


def test_select_top_k_preserves_explicit_experiment_value() -> None:
    with patch("cli.questionary.text") as text_prompt:
        selected = make_cli().select_top_k(5, interactive=True)

    assert selected == 5
    text_prompt.assert_not_called()


def test_select_top_k_prompts_during_interactive_run() -> None:
    question = Mock()
    question.ask.return_value = "4"
    with patch("cli.questionary.text", return_value=question) as text_prompt:
        selected = make_cli().select_top_k(None, interactive=True)

    assert selected == 4
    assert text_prompt.call_args.args == ("Top-K chunks to retrieve:",)
    assert callable(text_prompt.call_args.kwargs["validate"])


def test_select_top_k_requires_flag_for_noninteractive_run() -> None:
    with pytest.raises(ValueError, match="Pass --top-k"):
        make_cli().select_top_k(None, interactive=False)


def test_run_resolves_configuration_before_calling_runner(tmp_path: Path) -> None:
    cli = make_cli()
    cli.provider_factory.available_providers.return_value = ("groq",)
    cli.provider_factory.resolve_model.return_value = "cli-model"

    cli.run(
        [
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
    )

    config = cli.runner.run.call_args.args[0]
    assert config.provider == "groq"
    assert config.test_case == tmp_path
    assert config.model == "cli-model"
    assert config.top_k == 2
    assert config.max_tokens == 64


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--top-k", "0", "--top-k must be at least 1"),
        ("--max-tokens", "0", "--max-tokens must be at least 1"),
    ],
)
def test_run_rejects_non_positive_numeric_options(
    tmp_path: Path,
    option: str,
    value: str,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        make_cli().run([str(tmp_path), option, value])

    assert error.value.code == 2
    assert message in capsys.readouterr().err

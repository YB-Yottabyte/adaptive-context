import argparse
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cli import BaselineCLI
from debugging_case import CaseMode, DebuggingCase, DebuggingCaseLoader


def make_cli(*, environment: dict[str, str] | None = None) -> BaselineCLI:
    factory = Mock()
    runner = Mock()
    case_loader = Mock()
    case_loader.load_controlled.return_value = DebuggingCase(
        repository_path=Path("/controlled"),
        bug_report="Controlled failure",
        display_name="controlled",
        mode=CaseMode.CONTROLLED,
    )
    case_loader.load_repository.return_value = DebuggingCase(
        repository_path=Path("/repository"),
        bug_report="Real issue",
        display_name="repository",
        mode=CaseMode.REPOSITORY,
    )
    return BaselineCLI(
        provider_factory=factory,
        runner=runner,
        case_loader=case_loader,
        environment=environment or {},
    )


def parse(cli: BaselineCLI, values: list[str]) -> argparse.Namespace:
    return cli.build_parser().parse_args(values)


def test_parser_preserves_positional_controlled_case_mode() -> None:
    args = parse(make_cli(), ["payment_bug"])

    assert args.test_case == "payment_bug"
    assert args.example is None
    assert args.repo is None
    assert not hasattr(args, "retriever")


def test_short_test_case_name_resolves_under_test_cases() -> None:
    resolved = BaselineCLI.resolve_test_case("payment_bug")

    assert resolved.name == "payment_bug"
    assert resolved.parent.name == "test_cases"
    assert resolved.is_absolute()


def test_legacy_examples_prefix_maps_to_test_cases() -> None:
    resolved = BaselineCLI.resolve_controlled_case("examples/payment_bug")

    assert resolved.name == "payment_bug"
    assert resolved.parent.name == "test_cases"


def test_explicit_test_case_path_is_preserved(tmp_path: Path) -> None:
    assert BaselineCLI.resolve_test_case(tmp_path) == tmp_path


def test_controlled_case_loader_keeps_existing_mode_working(tmp_path: Path) -> None:
    (tmp_path / "bug_report.txt").write_text("Something broke.", encoding="utf-8")
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    cli = make_cli()
    cli.case_loader = DebuggingCaseLoader()

    case = cli.resolve_input(parse(cli, [str(tmp_path)]))

    assert case.repository_path == tmp_path
    assert case.bug_report == "Something broke."
    assert case.mode is CaseMode.CONTROLLED


def test_repo_and_issue_file_load_real_repository_mode(tmp_path: Path) -> None:
    cli = make_cli()
    repo = tmp_path / "repo"
    issue = tmp_path / "issue.txt"

    case = cli.resolve_input(
        parse(cli, ["--repo", str(repo), "--issue-file", str(issue)])
    )

    assert case.mode is CaseMode.REPOSITORY
    cli.case_loader.load_repository.assert_called_once_with(repo, issue, None)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--repo", "repo"], "--repo requires --issue-file"),
        (["--issue-file", "issue.txt"], "--issue-file requires --repo"),
        (
            ["payment_bug", "--example", "auth_bug"],
            "Use either TEST_CASE or --example",
        ),
        (
            [
                "--example",
                "payment_bug",
                "--repo",
                "repo",
                "--issue-file",
                "issue.txt",
            ],
            "Controlled example mode cannot be combined",
        ),
        (["--metadata", "metadata.json"], "--metadata is only valid"),
        ([], "Provide a controlled TEST_CASE/--example"),
    ],
)
def test_invalid_input_mode_combinations_are_rejected(
    arguments: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        make_cli().run([*arguments, "--provider", "groq", "--top-k", "1"])

    assert error.value.code == 2
    assert message in capsys.readouterr().err


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


def test_run_resolves_configuration_before_calling_runner() -> None:
    cli = make_cli()
    cli.provider_factory.available_providers.return_value = ("groq",)
    cli.provider_factory.resolve_model.return_value = "cli-model"

    cli.run(
        [
            "payment_bug",
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
    assert config.case.mode is CaseMode.CONTROLLED
    assert config.provider == "groq"
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
    option: str,
    value: str,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        make_cli().run(["payment_bug", option, value])

    assert error.value.code == 2
    assert message in capsys.readouterr().err

"""CLI coverage for the two supported input workflows."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from cli import BaselineCLI
from debugging_case import CaseMode


def make_cli() -> tuple[BaselineCLI, Mock]:
    provider_factory = Mock()
    provider_factory.available_providers.return_value = ("groq",)
    provider_factory.resolve_model.return_value = "openai/gpt-oss-20b"
    runner = Mock()
    return (
        BaselineCLI(provider_factory=provider_factory, runner=runner),
        runner,
    )


def test_controlled_case_runs_through_cli(tmp_path: Path) -> None:
    (tmp_path / "bug_report.txt").write_text("Checkout fails.", encoding="utf-8")
    (tmp_path / "payment.py").write_text("def checkout(): pass\n", encoding="utf-8")
    cli, runner = make_cli()

    cli.run([str(tmp_path), "--provider", "groq", "--top-k", "3"])

    config = runner.run.call_args.args[0]
    assert config.case.mode is CaseMode.CONTROLLED
    assert config.case.repository_path == tmp_path.resolve()
    assert config.case.bug_report == "Checkout fails."
    assert config.top_k == 3


def test_repository_and_issue_file_run_through_cli(tmp_path: Path) -> None:
    repository = tmp_path / "requests"
    repository.mkdir()
    issue = tmp_path / "requests_7432.txt"
    issue.write_text("Streaming upload fails after a redirect.", encoding="utf-8")
    cli, runner = make_cli()

    cli.run(
        [
            "--repo",
            str(repository),
            "--issue-file",
            str(issue),
            "--provider",
            "groq",
            "--top-k",
            "3",
        ]
    )

    case = runner.run.call_args.args[0].case
    assert case.mode is CaseMode.REPOSITORY
    assert case.repository_path == repository.resolve()
    assert case.bug_report == "Streaming upload fails after a redirect."


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--repo", "repo"], "--repo requires --issue-file"),
        (["--issue-file", "issue.txt"], "--issue-file requires --repo"),
        (
            ["payment_bug", "--repo", "repo", "--issue-file", "issue.txt"],
            "Controlled example mode cannot be combined",
        ),
        (["--metadata", "metadata.json"], "--metadata is only valid"),
    ],
)
def test_invalid_input_combinations_fail_cleanly(
    arguments: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli, _ = make_cli()

    with pytest.raises(SystemExit) as error:
        cli.run([*arguments, "--provider", "groq", "--top-k", "1"])

    assert error.value.code == 2
    assert message in capsys.readouterr().err

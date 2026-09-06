from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from rich.console import Console

from pipeline import BaselineRunConfig, BaselineRunner, BugReportLoader
from providers.base import GenerationResult
from retrieval import CodeChunk, RetrievedChunk
from terminal_ui import TerminalUI


def test_bug_report_loader_reads_non_empty_report(tmp_path: Path) -> None:
    (tmp_path / "bug_report.txt").write_text("Something broke.\n", encoding="utf-8")

    assert BugReportLoader().load(tmp_path) == "Something broke."


def test_bug_report_loader_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Test case directory does not exist"):
        BugReportLoader().load(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="Missing bug report"):
        BugReportLoader().load(tmp_path)

    (tmp_path / "bug_report.txt").write_text(" \n", encoding="utf-8")
    with pytest.raises(ValueError, match="Bug report is empty"):
        BugReportLoader().load(tmp_path)


def test_runner_performs_one_retrieval_before_one_model_call(tmp_path: Path) -> None:
    (tmp_path / "bug_report.txt").write_text("Checkout fails.", encoding="utf-8")
    selected_chunk = CodeChunk(
        "payment.py",
        "def checkout():\n    return False",
        1,
        2,
    )
    chunker = Mock()
    chunker.create_chunks.return_value = [selected_chunk]
    retriever = Mock(display_name="ChromaDB semantic search")
    retriever.retrieve.return_value = [RetrievedChunk(selected_chunk, 0.9)]
    provider = Mock()
    provider.generate.return_value = GenerationResult(text="Conclusion:\nLikely issue.")
    provider_factory = Mock()
    provider_factory.create.return_value = provider
    output = StringIO()
    terminal = TerminalUI(
        Console(file=output, color_system=None),
        sleeper=lambda _: None,
    )
    runner = BaselineRunner(
        chunker=chunker,
        retriever=retriever,
        provider_factory=provider_factory,
        terminal=terminal,
    )

    timings = runner.run(
        BaselineRunConfig(
            test_case=tmp_path,
            provider="groq",
            model="openai/gpt-oss-20b",
            top_k=1,
            max_tokens=100,
        )
    )

    retriever.retrieve.assert_called_once()
    assert retriever.retrieve.call_args.args[0] == "Checkout fails."
    assert retriever.retrieve.call_args.args[2:] == (1, tmp_path.resolve())
    provider.generate.assert_called_once()
    assert "payment.py" in provider.generate.call_args.args[0]
    assert "● Retrieved 1 relevant chunk" in output.getvalue()
    assert timings.total >= 0


def test_runner_rejects_repository_without_python_before_provider_creation(
    tmp_path: Path,
) -> None:
    (tmp_path / "bug_report.txt").write_text("A function fails.", encoding="utf-8")
    chunker = Mock()
    chunker.create_chunks.return_value = []
    provider_factory = Mock()
    terminal = TerminalUI(Console(file=StringIO(), color_system=None))
    runner = BaselineRunner(
        chunker=chunker,
        retriever=Mock(display_name="ChromaDB semantic search"),
        provider_factory=provider_factory,
        terminal=terminal,
    )

    with pytest.raises(ValueError, match="No Python source files found"):
        runner.run(
            BaselineRunConfig(
                test_case=tmp_path,
                provider="groq",
                model="model",
                top_k=1,
            )
        )

    provider_factory.create.assert_not_called()


def test_runner_caps_top_k_to_available_chunks_and_reports_it(tmp_path: Path) -> None:
    (tmp_path / "bug_report.txt").write_text("Checkout fails.", encoding="utf-8")
    chunks = [
        CodeChunk("payment.py", "def checkout(): return False", 1, 1),
        CodeChunk("test_payment.py", "def test_checkout(): pass", 1, 1),
    ]
    chunker = Mock()
    chunker.create_chunks.return_value = chunks
    retriever = Mock(display_name="ChromaDB semantic search")
    retriever.retrieve.return_value = [
        RetrievedChunk(chunk, 0.9) for chunk in chunks
    ]
    provider = Mock()
    provider.generate.return_value = GenerationResult(text="Incomplete response")
    provider_factory = Mock()
    provider_factory.create.return_value = provider
    output = StringIO()
    runner = BaselineRunner(
        chunker=chunker,
        retriever=retriever,
        provider_factory=provider_factory,
        terminal=TerminalUI(
            Console(file=output, color_system=None, width=100),
            sleeper=lambda _: None,
        ),
    )

    runner.run(
        BaselineRunConfig(
            test_case=tmp_path,
            provider="groq",
            model="model",
            top_k=5,
        )
    )

    assert retriever.retrieve.call_args.args[2] == 2
    assert (
        "Requested top-k = 5, but only 2 repository chunks are available; "
        "using top-k = 2."
    ) in output.getvalue()

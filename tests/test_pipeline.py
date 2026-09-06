"""Workflow coverage for one complete Static RAG pipeline pass."""

from io import StringIO
from pathlib import Path
from unittest.mock import Mock

from rich.console import Console

from debugging_case import CaseMode, DebuggingCase
from pipeline import BaselineRunConfig, BaselineRunner
from providers.base import GenerationResult
from retrieval import CodeChunk, RepositoryIndex, RetrievedChunk, SourceDocument
from terminal_ui import TerminalUI


def test_pipeline_performs_one_retrieval_and_one_model_call(tmp_path: Path) -> None:
    chunk = CodeChunk(
        "payment.py",
        "def checkout():\n    return False",
        1,
        2,
    )
    index = RepositoryIndex(
        (SourceDocument(chunk.source_path, chunk.content),),
        (chunk,),
    )
    chunker = Mock()
    chunker.create_index.return_value = index
    retriever = Mock(display_name="ChromaDB semantic search")
    retriever.retrieve.return_value = [RetrievedChunk(chunk, 0.9)]
    provider = Mock()
    provider.generate.return_value = GenerationResult(
        text="Conclusion:\ncheckout returns the wrong value.",
        input_tokens=40,
        output_tokens=10,
    )
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
    case = DebuggingCase(
        repository_path=tmp_path,
        bug_report="Checkout fails.",
        display_name="payment_bug",
        mode=CaseMode.CONTROLLED,
    )

    timings = runner.run(
        BaselineRunConfig(
            case=case,
            provider="groq",
            model="openai/gpt-oss-20b",
            top_k=1,
            max_tokens=100,
        )
    )

    chunker.create_index.assert_called_once_with(tmp_path.resolve())
    retriever.retrieve.assert_called_once_with(
        "Checkout fails.",
        [chunk],
        1,
        tmp_path.resolve(),
    )
    provider_factory.create.assert_called_once_with(
        "groq",
        "openai/gpt-oss-20b",
        100,
    )
    provider.generate.assert_called_once()
    retriever.close.assert_called_once_with()
    assert "payment.py" in provider.generate.call_args.args[0]
    assert timings.context_metrics is not None
    assert timings.context_metrics.source_files == 1
    assert timings.context_metrics.chunks_selected == 1
    assert "Retrieved 1 relevant chunk" in output.getvalue()
    assert "40 input · 10 output · 50 total tokens" in output.getvalue()
    assert "Completed in" in output.getvalue()

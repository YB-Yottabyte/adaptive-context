from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from rich.console import Console

from debugging_case import CaseMode, DebuggingCase, EvaluationMetadata
from pipeline import BaselineRunConfig, BaselineRunner
from providers.base import GenerationResult
from retrieval import CodeChunk, RepositoryIndex, RetrievedChunk, SourceDocument
from terminal_ui import TerminalUI


def make_case(
    repository: Path,
    *,
    real: bool = False,
    metadata: EvaluationMetadata | None = None,
) -> DebuggingCase:
    return DebuggingCase(
        repository_path=repository,
        bug_report="Checkout fails.",
        display_name="example/sample-repository" if real else repository.name,
        mode=CaseMode.REPOSITORY if real else CaseMode.CONTROLLED,
        metadata=metadata,
    )


def make_index(chunks: list[CodeChunk]) -> RepositoryIndex:
    documents_by_path = {
        chunk.source_path: SourceDocument(chunk.source_path, chunk.content)
        for chunk in chunks
    }
    return RepositoryIndex(tuple(documents_by_path.values()), tuple(chunks))


def make_runner(
    output: StringIO,
    *,
    chunker: Mock,
    retriever: Mock,
    provider_factory: Mock,
) -> BaselineRunner:
    return BaselineRunner(
        chunker=chunker,
        retriever=retriever,
        provider_factory=provider_factory,
        terminal=TerminalUI(
            Console(file=output, color_system=None, width=100),
            sleeper=lambda _: None,
        ),
    )


def make_provider_factory(response: str = "Conclusion:\nLikely issue.") -> Mock:
    provider = Mock()
    provider.generate.return_value = GenerationResult(
        text=response,
        input_tokens=40,
        output_tokens=10,
    )
    factory = Mock()
    factory.create.return_value = provider
    return factory


def test_runner_performs_one_retrieval_before_one_model_call(tmp_path: Path) -> None:
    selected_chunk = CodeChunk(
        "payment.py",
        "def checkout():\n    return False",
        1,
        2,
    )
    chunker = Mock()
    chunker.create_index.return_value = make_index([selected_chunk])
    retriever = Mock(display_name="ChromaDB semantic search")
    retriever.retrieve.return_value = [RetrievedChunk(selected_chunk, 0.9)]
    provider_factory = make_provider_factory()
    output = StringIO()
    runner = make_runner(
        output,
        chunker=chunker,
        retriever=retriever,
        provider_factory=provider_factory,
    )

    timings = runner.run(
        BaselineRunConfig(
            case=make_case(tmp_path),
            provider="groq",
            model="openai/gpt-oss-20b",
            top_k=1,
            max_tokens=100,
        )
    )

    retriever.retrieve.assert_called_once()
    assert retriever.retrieve.call_args.args[0] == "Checkout fails."
    assert retriever.retrieve.call_args.args[2:] == (1, tmp_path.resolve())
    provider = provider_factory.create.return_value
    provider.generate.assert_called_once()
    assert "payment.py" in provider.generate.call_args.args[0]
    assert "● Retrieved 1 relevant chunk" in output.getvalue()
    assert "Indexed repository" not in output.getvalue()
    assert timings.total >= 0
    assert timings.context_metrics is not None
    retriever.close.assert_called_once_with()


def test_runner_rejects_repository_without_python_before_provider_creation(
    tmp_path: Path,
) -> None:
    chunker = Mock()
    chunker.create_index.return_value = RepositoryIndex((), ())
    provider_factory = Mock()
    runner = make_runner(
        StringIO(),
        chunker=chunker,
        retriever=Mock(display_name="ChromaDB semantic search"),
        provider_factory=provider_factory,
    )

    with pytest.raises(ValueError, match="No Python source files found"):
        runner.run(
            BaselineRunConfig(
                case=make_case(tmp_path),
                provider="groq",
                model="model",
                top_k=1,
            )
        )

    provider_factory.create.assert_not_called()
    runner.retriever.close.assert_called_once_with()


def test_runner_closes_retriever_after_completion_footer(tmp_path: Path) -> None:
    selected_chunk = CodeChunk("payment.py", "def checkout(): return False", 1, 1)
    chunker = Mock()
    chunker.create_index.return_value = make_index([selected_chunk])
    retriever = Mock(display_name="ChromaDB semantic search")
    retriever.retrieve.return_value = [RetrievedChunk(selected_chunk, 0.9)]
    output = StringIO()
    retriever.close.side_effect = lambda: (
        ("Completed in" in output.getvalue())
        or pytest.fail("retriever closed before completion footer")
    )
    runner = make_runner(
        output,
        chunker=chunker,
        retriever=retriever,
        provider_factory=make_provider_factory(),
    )

    runner.run(
        BaselineRunConfig(
            case=make_case(tmp_path),
            provider="groq",
            model="model",
            top_k=1,
        )
    )

    retriever.close.assert_called_once_with()


def test_real_run_maps_suggestion_against_full_selected_source_file(
    tmp_path: Path,
) -> None:
    full_source = (
        "def prepare_body(data):\n"
        "    if isinstance(data, Iterable) and not isinstance(\n"
        "        data, (str, bytes, list, tuple, Mapping)\n"
        "    ):\n"
        "        return stream(data)\n"
    )
    selected_chunk = CodeChunk(
        "src/requests/models.py",
        "    ):\n        return stream(data)",
        4,
        5,
    )
    chunker = Mock()
    chunker.create_index.return_value = RepositoryIndex(
        (SourceDocument("src/requests/models.py", full_source),),
        (selected_chunk,),
    )
    retriever = Mock(display_name="ChromaDB semantic search")
    retriever.retrieve.return_value = [RetrievedChunk(selected_chunk, 0.9)]
    response = (
        "Evidence considered:\n- models.py checks Iterable.\n\n"
        "Conclusion:\nStream detection is too strict.\n\n"
        "Relevant file:\nsrc/requests/models.py\n\n"
        "Suggested change:\n"
        "# Original snippet (simplified)\n"
        "# if isinstance(data, Iterable):\n"
        "#     ...\n"
        "# Updated snippet\n"
        'if hasattr(data, "__iter__"):\n'
        "    ...\n\n"
        "Explanation:\nAccept delegated iterators."
    )
    output = StringIO()
    runner = make_runner(
        output,
        chunker=chunker,
        retriever=retriever,
        provider_factory=make_provider_factory(response),
    )

    runner.run(
        BaselineRunConfig(
            case=make_case(tmp_path, real=True),
            provider="groq",
            model="model",
            top_k=1,
        )
    )

    rendered = output.getvalue()
    assert "2 │     if isinstance(data, Iterable) and not isinstance(" in rendered
    assert '2 └────→     if hasattr(data, "__iter__") and not isinstance(' in rendered


def test_runner_caps_top_k_to_available_chunks_and_reports_it(tmp_path: Path) -> None:
    chunks = [
        CodeChunk("payment.py", "def checkout(): return False", 1, 1),
        CodeChunk("test_payment.py", "def test_checkout(): pass", 1, 1),
    ]
    chunker = Mock()
    chunker.create_index.return_value = make_index(chunks)
    retriever = Mock(display_name="ChromaDB semantic search")
    retriever.retrieve.return_value = [RetrievedChunk(chunk, 0.9) for chunk in chunks]
    output = StringIO()
    runner = make_runner(
        output,
        chunker=chunker,
        retriever=retriever,
        provider_factory=make_provider_factory("Incomplete response"),
    )

    runner.run(
        BaselineRunConfig(
            case=make_case(tmp_path),
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


def test_real_repository_displays_index_and_context_metrics(tmp_path: Path) -> None:
    chunks = [CodeChunk("src/package/service.py", "def run(): pass", 20, 20)]
    chunker = Mock()
    chunker.create_index.return_value = make_index(chunks)
    retriever = Mock(display_name="ChromaDB semantic search")
    retriever.retrieve.return_value = [RetrievedChunk(chunks[0], 0.9)]
    output = StringIO()
    runner = make_runner(
        output,
        chunker=chunker,
        retriever=retriever,
        provider_factory=make_provider_factory(),
    )

    timings = runner.run(
        BaselineRunConfig(
            case=make_case(tmp_path, real=True),
            provider="groq",
            model="model",
            top_k=1,
        )
    )

    rendered = output.getvalue()
    assert "Repository\n  example/sample-repository" in rendered
    assert "Python files: 1" in rendered
    assert "Chunks indexed: 1" in rendered
    assert "Chunks sent to model: 1" in rendered
    assert "Repository source tokens (estimated):" in rendered
    assert "Retrieved context tokens (estimated):" in rendered
    assert "Repository source selected:" in rendered
    assert timings.context_metrics is not None
    assert timings.context_metrics.source_files == 1


def test_evaluation_metadata_never_enters_model_prompt(tmp_path: Path) -> None:
    chunk = CodeChunk("src/package/service.py", "def run(): pass", 1, 1)
    chunker = Mock()
    chunker.create_index.return_value = make_index([chunk])
    retriever = Mock(display_name="ChromaDB semantic search")
    retriever.retrieve.return_value = [RetrievedChunk(chunk, 0.9)]
    provider_factory = make_provider_factory()
    metadata = EvaluationMetadata(
        repo="SECRET_REPOSITORY_LABEL",
        issue_number=1234,
        base_commit="SECRET_BASE_COMMIT",
        gold_patch_files=("src/package/service.py",),
    )
    output = StringIO()
    runner = make_runner(
        output,
        chunker=chunker,
        retriever=retriever,
        provider_factory=provider_factory,
    )

    timings = runner.run(
        BaselineRunConfig(
            case=make_case(tmp_path, real=True, metadata=metadata),
            provider="groq",
            model="model",
            top_k=1,
        )
    )

    prompt = provider_factory.create.return_value.generate.call_args.args[0]
    assert "SECRET_REPOSITORY_LABEL" not in prompt
    assert "SECRET_BASE_COMMIT" not in prompt
    assert "issue_number" not in prompt
    assert "src/package/service.py" in prompt
    assert "Gold file retrieved: Yes" in output.getvalue()
    assert "✓ rank 1" in output.getvalue()
    assert timings.retrieval_evaluation is not None
    assert timings.retrieval_evaluation.best_gold_file_rank == 1

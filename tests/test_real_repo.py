"""Real-repository loading and evaluation workflow coverage."""

from io import StringIO
from pathlib import Path
from unittest.mock import Mock

from rich.console import Console

from debugging_case import CaseMode, DebuggingCase, DebuggingCaseLoader
from pipeline import BaselineRunConfig, BaselineRunner
from providers.base import GenerationResult
from retrieval import CodeChunk, RepositoryIndex, RetrievedChunk, SourceDocument
from terminal_ui import TerminalUI


def test_real_repository_case_loads_with_separate_issue_and_metadata(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "requests"
    repository.mkdir()
    issue = tmp_path / "requests_7432.txt"
    issue.write_text("Streaming body cannot be rewound.", encoding="utf-8")
    metadata = tmp_path / "requests_7432_metadata.json"
    metadata.write_text(
        """{
  "repo": "psf/requests",
  "issue_number": 7432,
  "base_commit": "0b401c7",
  "gold_patch_files": ["./src/requests/models.py"]
}
""",
        encoding="utf-8",
    )

    case = DebuggingCaseLoader().load_repository(repository, issue, metadata)

    assert case.mode is CaseMode.REPOSITORY
    assert case.repository_path == repository.resolve()
    assert case.bug_report == "Streaming body cannot be rewound."
    assert case.display_name == "psf/requests"
    assert case.metadata is not None
    assert case.metadata.base_commit == "0b401c7"
    assert case.metadata.gold_patch_files == ("src/requests/models.py",)


def test_metadata_stays_outside_prompt_and_gold_file_is_evaluated(
    tmp_path: Path,
) -> None:
    source_path = "src/requests/models.py"
    chunk = CodeChunk(source_path, "def prepare_body(): pass", 1, 1)
    chunker = Mock()
    chunker.create_index.return_value = RepositoryIndex(
        (SourceDocument(source_path, chunk.content),),
        (chunk,),
    )
    retriever = Mock(display_name="ChromaDB semantic search")
    retriever.retrieve.return_value = [RetrievedChunk(chunk, 0.9)]
    provider = Mock()
    provider.generate.return_value = GenerationResult(
        text="Conclusion:\nStream detection is too strict.",
        input_tokens=100,
        output_tokens=20,
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
    metadata = DebuggingCaseLoader().load_metadata(
        _write_metadata_file(tmp_path)
    )
    case = DebuggingCase(
        repository_path=tmp_path,
        bug_report="Streaming body cannot be rewound.",
        display_name="psf/requests",
        mode=CaseMode.REPOSITORY,
        metadata=metadata,
    )

    timings = runner.run(
        BaselineRunConfig(
            case=case,
            provider="groq",
            model="openai/gpt-oss-20b",
            top_k=1,
        )
    )

    prompt = provider.generate.call_args.args[0]
    assert "SECRET_REPOSITORY_LABEL" not in prompt
    assert "SECRET_BASE_COMMIT" not in prompt
    assert "issue_number" not in prompt
    assert source_path in prompt
    assert timings.retrieval_evaluation is not None
    assert timings.retrieval_evaluation.gold_file_retrieved
    assert timings.retrieval_evaluation.best_gold_file_rank == 1
    assert "Gold file retrieved: Yes" in output.getvalue()
    assert "rank 1" in output.getvalue()


def _write_metadata_file(tmp_path: Path) -> Path:
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        """{
  "repo": "SECRET_REPOSITORY_LABEL",
  "issue_number": 7432,
  "base_commit": "SECRET_BASE_COMMIT",
  "gold_patch_files": ["src/requests/models.py"]
}
""",
        encoding="utf-8",
    )
    return metadata

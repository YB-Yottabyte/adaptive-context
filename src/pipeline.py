"""Application orchestration for one static retrieval and diagnosis pass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from context_builder import DebuggingContextBuilder
from providers.factory import ProviderFactory
from retrieval import RepositoryChunker, RepositoryRetriever, RetrievedChunk
from semantic_retrieval import ChromaRetriever
from terminal_ui import TerminalUI


@dataclass(frozen=True)
class BaselineRunConfig:
    """Validated inputs for one baseline debugging run."""

    test_case: Path
    provider: str
    model: str
    top_k: int
    max_tokens: int = 512


@dataclass(frozen=True)
class RunTimings:
    """Measured pipeline timings and actual end-to-end duration."""

    test_case_load: float
    retrieval: float
    context_build: float
    first_token: float
    model_total: float
    total_execution: float

    @property
    def total(self) -> float:
        """Return the actual elapsed time for the complete baseline run."""
        return self.total_execution


class BugReportLoader:
    """Load and validate the developer-observed behavior for a test case."""

    def load(self, test_case_dir: Path) -> str:
        """Read a non-empty bug_report.txt from a test-case directory."""
        if not test_case_dir.exists():
            raise FileNotFoundError(
                f"Test case directory does not exist: {test_case_dir}"
            )
        if not test_case_dir.is_dir():
            raise NotADirectoryError(
                f"Test case path is not a directory: {test_case_dir}"
            )

        bug_report_path = test_case_dir / "bug_report.txt"
        if not bug_report_path.is_file():
            raise FileNotFoundError(f"Missing bug report: {bug_report_path}")

        bug_report = bug_report_path.read_text(encoding="utf-8").strip()
        if not bug_report:
            raise ValueError(f"Bug report is empty: {bug_report_path}")
        return bug_report


class BaselineRunner:
    """Coordinate one fixed Top-K retrieval followed by one LLM diagnosis."""

    def __init__(
        self,
        *,
        loader: BugReportLoader | None = None,
        chunker: RepositoryChunker | None = None,
        retriever: RepositoryRetriever | None = None,
        context_builder: DebuggingContextBuilder | None = None,
        provider_factory: ProviderFactory | None = None,
        terminal: TerminalUI | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.loader = loader or BugReportLoader()
        self.chunker = chunker or RepositoryChunker()
        self.retriever = retriever or ChromaRetriever()
        self.context_builder = context_builder or DebuggingContextBuilder()
        self.provider_factory = provider_factory or ProviderFactory()
        self.terminal = terminal or TerminalUI(clock=clock)
        self.clock = clock

    def run(self, config: BaselineRunConfig) -> RunTimings:
        """Execute the non-agentic baseline and return measured timings."""
        run_started_at = self.clock()
        test_case_dir = config.test_case.resolve()
        self.terminal.display_startup(config.provider, config.model)
        self.terminal.display_retrieval_configuration(self.retriever, config.top_k)
        self.terminal.console.print()

        def search_repository() -> tuple[
            str,
            list[RetrievedChunk],
            float,
            float,
            int,
        ]:
            load_started_at = self.clock()
            bug_report = self.loader.load(test_case_dir)
            test_case_load = self.clock() - load_started_at

            chunking_started_at = self.clock()
            chunks = self.chunker.create_chunks(test_case_dir)
            chunking_elapsed = self.clock() - chunking_started_at
            if not chunks:
                raise ValueError(f"No Python source files found in: {test_case_dir}")

            effective_top_k = min(config.top_k, len(chunks))

            ranking_started_at = self.clock()
            retrieved = self.retriever.retrieve(
                bug_report,
                chunks,
                effective_top_k,
                test_case_dir,
            )
            ranking_elapsed = self.clock() - ranking_started_at
            return (
                bug_report,
                retrieved,
                test_case_load,
                chunking_elapsed + ranking_elapsed,
                len(chunks),
            )

        search_result, _ = self.terminal.run_timed_stage(
            "Searching the repository for relevant code...",
            search_repository,
        )
        (
            bug_report,
            retrieved,
            test_case_load,
            retrieval_elapsed,
            available_chunks,
        ) = search_result
        if config.top_k > available_chunks:
            self.terminal.display_top_k_adjustment(config.top_k, available_chunks)
        selected_chunks = [item.chunk for item in retrieved]
        self.terminal.display_retrieved_context(selected_chunks)
        self.terminal.console.print()

        prompt, context_elapsed = self.terminal.run_timed_stage(
            "Tracing the failing behavior through the selected files...",
            lambda: self.context_builder.build_prompt(bug_report, selected_chunks),
        )
        self.terminal.console.print()

        provider = self.provider_factory.create(
            config.provider,
            config.model,
            config.max_tokens,
        )
        result, first_token, model_total = self.terminal.generate_response(
            provider,
            prompt,
            selected_chunks=selected_chunks,
        )
        total_execution = self.clock() - run_started_at
        timings = RunTimings(
            test_case_load=test_case_load,
            retrieval=retrieval_elapsed,
            context_build=context_elapsed,
            first_token=first_token,
            model_total=model_total,
            total_execution=total_execution,
        )
        self.terminal.display_footer(result, total_execution)
        return timings

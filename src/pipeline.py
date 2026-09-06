"""Application orchestration for one static retrieval and diagnosis pass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from context_builder import DebuggingContextBuilder
from context_metrics import (
    ContextMetrics,
    ContextMetricsCalculator,
    RetrievalEvaluation,
)
from debugging_case import DebuggingCase
from providers.factory import ProviderFactory
from retrieval import (
    CodeChunk,
    RepositoryChunker,
    RepositoryIndex,
    RepositoryRetriever,
    RetrievedChunk,
)
from semantic_retrieval import ChromaRetriever
from terminal_ui import TerminalUI


@dataclass(frozen=True)
class BaselineRunConfig:
    """Validated inputs for one baseline debugging run."""

    case: DebuggingCase
    provider: str
    model: str
    top_k: int
    max_tokens: int = 512


@dataclass(frozen=True)
class RunTimings:
    """Measured pipeline timings and actual end-to-end duration."""

    retrieval: float
    context_build: float
    first_token: float
    model_total: float
    total_execution: float
    context_metrics: ContextMetrics | None = None
    retrieval_evaluation: RetrievalEvaluation | None = None

    @property
    def total(self) -> float:
        """Return the actual elapsed time for the complete baseline run."""
        return self.total_execution


class BaselineRunner:
    """Coordinate one fixed Top-K retrieval followed by one LLM diagnosis."""

    def __init__(
        self,
        *,
        chunker: RepositoryChunker | None = None,
        retriever: RepositoryRetriever | None = None,
        context_builder: DebuggingContextBuilder | None = None,
        metrics_calculator: ContextMetricsCalculator | None = None,
        provider_factory: ProviderFactory | None = None,
        terminal: TerminalUI | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.chunker = chunker or RepositoryChunker()
        self.retriever = retriever or ChromaRetriever()
        self.context_builder = context_builder or DebuggingContextBuilder()
        self.metrics_calculator = metrics_calculator or ContextMetricsCalculator()
        self.provider_factory = provider_factory or ProviderFactory()
        self.terminal = terminal or TerminalUI(clock=clock)
        self.clock = clock

    def run(self, config: BaselineRunConfig) -> RunTimings:
        """Execute the non-agentic baseline and release retrieval resources."""
        try:
            return self._run(config)
        finally:
            close = getattr(self.retriever, "close", None)
            if callable(close):
                close()

    def _run(self, config: BaselineRunConfig) -> RunTimings:
        """Execute the fixed one-retrieval, one-generation workflow."""
        run_started_at = self.clock()
        case = config.case
        repository_dir = case.repository_path.resolve()
        self.terminal.display_startup(config.provider, config.model)
        self.terminal.display_retrieval_configuration(self.retriever, config.top_k)
        if case.is_real_repository:
            self.terminal.display_repository(case.display_name, repository_dir)
        self.terminal.console.print()

        def search_repository() -> tuple[
            list[RetrievedChunk],
            RepositoryIndex,
            float,
            int,
        ]:
            chunking_started_at = self.clock()
            repository_index = self.chunker.create_index(repository_dir)
            chunking_elapsed = self.clock() - chunking_started_at
            chunks = list(repository_index.chunks)
            if not chunks:
                raise ValueError(f"No Python source files found in: {repository_dir}")

            effective_top_k = min(config.top_k, len(chunks))

            ranking_started_at = self.clock()
            retrieved = self.retriever.retrieve(
                case.bug_report,
                chunks,
                effective_top_k,
                repository_dir,
            )
            ranking_elapsed = self.clock() - ranking_started_at
            return (
                retrieved,
                repository_index,
                chunking_elapsed + ranking_elapsed,
                len(chunks),
            )

        search_result, _ = self.terminal.run_timed_stage(
            "Searching the repository for relevant code...",
            search_repository,
        )
        (
            retrieved,
            repository_index,
            retrieval_elapsed,
            available_chunks,
        ) = search_result
        if config.top_k > available_chunks:
            self.terminal.display_top_k_adjustment(config.top_k, available_chunks)
        selected_chunks = [item.chunk for item in retrieved]
        if case.is_real_repository:
            self.terminal.display_repository_index(
                repository_index.source_file_count,
                len(repository_index.chunks),
            )
        self.terminal.display_retrieved_context(selected_chunks)
        retrieval_evaluation = (
            self.metrics_calculator.evaluate_retrieval(case.metadata, selected_chunks)
            if case.metadata is not None
            else None
        )
        if retrieval_evaluation is not None:
            self.terminal.display_retrieval_evaluation(retrieval_evaluation)
        self.terminal.console.print()

        formatted_context = self.context_builder.format_repository_context(
            selected_chunks
        )
        context_metrics = self.metrics_calculator.calculate(
            repository_index,
            selected_chunks,
            formatted_context,
        )

        prompt, context_elapsed = self.terminal.run_timed_stage(
            "Tracing the failing behavior through the selected files...",
            lambda: self.context_builder.build_prompt(case.bug_report, selected_chunks),
        )
        self.terminal.console.print()

        provider = self.provider_factory.create(
            config.provider,
            config.model,
            config.max_tokens,
        )
        presentation_chunks = self._presentation_source_chunks(
            repository_index,
            selected_chunks,
        )
        result, first_token, model_total = self.terminal.generate_response(
            provider,
            prompt,
            selected_chunks=presentation_chunks,
        )
        total_execution = self.clock() - run_started_at
        timings = RunTimings(
            retrieval=retrieval_elapsed,
            context_build=context_elapsed,
            first_token=first_token,
            model_total=model_total,
            total_execution=total_execution,
            context_metrics=context_metrics,
            retrieval_evaluation=retrieval_evaluation,
        )
        if case.is_real_repository:
            self.terminal.display_context_metrics(context_metrics)
        self.terminal.display_footer(result, total_execution)
        return timings

    @staticmethod
    def _presentation_source_chunks(
        repository_index: RepositoryIndex,
        selected_chunks: list[CodeChunk],
    ) -> list[CodeChunk]:
        """Expose full selected files only to the terminal source matcher."""
        selected_paths = {chunk.source_path for chunk in selected_chunks}
        return [
            CodeChunk(
                source_path=document.source_path,
                content=document.content,
                start_line=1,
                end_line=len(document.content.splitlines()),
            )
            for document in repository_index.documents
            if document.source_path in selected_paths
        ]

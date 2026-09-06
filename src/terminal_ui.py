"""Rich terminal presentation for baseline debugging runs."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from queue import Empty, Queue
from threading import Thread
from time import perf_counter, sleep

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from debugging_result import (
    DebuggingSummaryParser,
    SourceChangeAnalyzer,
    SourceDefinition,
)
from providers.base import GenerationResult, LLMProvider
from retrieval import CodeChunk, RepositoryRetriever

PROVIDER_LABELS = {"local": "Local MLX", "groq": "Groq"}
RESPONSE_CHARACTERS_PER_FRAME = 2
RESPONSE_FRAME_INTERVAL_SECONDS = 0.04
INTRODUCTION = (
    "I'll trace the reported behavior through the most relevant files and look "
    "for the smallest fix."
)


class TerminalUI:
    """Own all terminal rendering and smooth response playback."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        clock: Callable[[], float] = perf_counter,
        sleeper: Callable[[float], None] = sleep,
        characters_per_frame: int = RESPONSE_CHARACTERS_PER_FRAME,
        frame_interval: float = RESPONSE_FRAME_INTERVAL_SECONDS,
    ) -> None:
        self.console = console or Console()
        self.clock = clock
        self.sleeper = sleeper
        self.characters_per_frame = characters_per_frame
        self.frame_interval = frame_interval
        self.summary_parser = DebuggingSummaryParser()
        self.change_analyzer = SourceChangeAnalyzer(self.summary_parser)

    @staticmethod
    def format_model_label(model_name: str) -> str:
        """Return a concise display label without changing the model value."""
        short_name = model_name.rsplit("/", maxsplit=1)[-1]
        match = re.fullmatch(r"gpt-oss-(\d+)b", short_name, flags=re.IGNORECASE)
        if match:
            return f"GPT-OSS {match.group(1)}B"
        return short_name

    def display_startup(self, provider_name: str, model_name: str) -> None:
        """Render a Codex-inspired welcome panel without exposing a path."""
        heading = Text(">_ ", style="dim")
        heading.append("Adaptive Context", style="bold")
        heading.append("  (static RAG baseline)", style="dim")

        model_line = Text("model:    ", style="dim")
        model_line.append(self.format_model_label(model_name), style="bold")
        model_line.append(f" via {PROVIDER_LABELS[provider_name]}", style="dim")

        self.console.print(
            Panel(
                Group(heading, Text(""), model_line),
                box=box.ROUNDED,
                border_style="dim",
                padding=(0, 2),
                width=min(68, self.console.width),
            )
        )
        self.console.print()
        self.console.print(INTRODUCTION)

    def run_timed_stage[T](
        self,
        running_message: str,
        action: Callable[[], T],
    ) -> tuple[T, float]:
        """Run one real pipeline action with transient status and timing."""
        with self.console.status(
            running_message,
            spinner="dots",
            spinner_style="cyan",
        ):
            started_at = self.clock()
            result = action()
            elapsed = self.clock() - started_at
        return result, elapsed

    def display_retrieved_context(self, chunks: Sequence[CodeChunk]) -> None:
        """Render selected Top-K chunks and their source locations."""
        noun = "chunk" if len(chunks) == 1 else "chunks"
        heading = Text("● ", style="green")
        heading.append(f"Retrieved {len(chunks)} relevant {noun}", style="bold")
        self.console.print(heading)
        for index, chunk in enumerate(chunks):
            branch = "└─" if index == len(chunks) - 1 else "├─"
            self.console.print(
                f"  {branch} {chunk.source_path}:{chunk.start_line}-{chunk.end_line}"
            )

    def display_retrieval_configuration(
        self,
        retriever: RepositoryRetriever,
        top_k: int,
    ) -> None:
        """Identify the configured static retrieval method and result count."""
        self.console.print(
            f"Retrieval: {retriever.display_name} · top-k = {top_k}",
            style="dim",
        )

    def display_top_k_adjustment(self, requested: int, available: int) -> None:
        """Explain when Top-K is capped by the available repository chunks."""
        self.console.print(
            f"Requested top-k = {requested}, but only {available} repository "
            f"chunks are available; using top-k = {available}.",
            style="yellow",
        )

    def response_renderable(
        self,
        response_text: str,
        *,
        final: bool,
        selected_chunks: Sequence[CodeChunk] = (),
    ) -> Group:
        """Build the progressively renderable public debugging summary."""
        summary = self.summary_parser.parse(
            response_text,
            fallback_to_response=final,
        )
        relevant_file = self.summary_parser.relevant_file_name(summary.relevant_file)
        current_definition = self.change_analyzer.find_current_definition(
            summary,
            selected_chunks,
        )
        suggestion_repeats_source = False
        if current_definition is not None and summary.suggested_change:
            before_changed, after_changed = self.change_analyzer.changed_line_numbers(
                current_definition.code,
                summary.suggested_change,
            )
            suggestion_repeats_source = not before_changed and not after_changed
        diagnosis_incomplete = final and (
            not summary.is_complete or suggestion_repeats_source
        )
        heading = Text("● ", style="yellow" if diagnosis_incomplete else "green")
        if diagnosis_incomplete:
            heading.append("Diagnosis incomplete", style="bold")
        elif relevant_file:
            heading.append("Diagnosis points to ", style="bold")
            heading.append(relevant_file, style="bold")
        elif not final:
            heading.append("Reviewing the selected evidence", style="bold")
        else:
            heading.append("Diagnosis suggests a likely issue", style="bold")
        renderables: list[RenderableType] = [heading]
        suggestion_complete = final or bool(
            re.search(
                r"^\s*Explanation:\s*$",
                response_text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        )

        if summary.evidence:
            renderables.extend((Text(""), Text("  Evidence", style="bold")))
            for index, evidence in enumerate(summary.evidence):
                branch = "└─" if index == len(summary.evidence) - 1 else "├─"
                renderables.append(Padding(Text(f"{branch} {evidence}"), (0, 0, 0, 2)))

        if diagnosis_incomplete:
            incomplete_message = (
                "The model repeated the current implementation without proposing "
                "a code change."
                if suggestion_repeats_source
                else "The model did not return a complete file-and-fix suggestion."
            )
            renderables.extend(
                (
                    Text(""),
                    Padding(
                        Text(incomplete_message),
                        (0, 0, 0, 2),
                    ),
                    Padding(
                        Text(
                            "The suggestion is shown below for transparency; "
                            "rerun or select a stronger model before relying on it."
                            if suggestion_repeats_source
                            else "Try increasing Top-K (for example, --top-k 3). "
                            "If this repeats, increase --max-tokens."
                        ),
                        (0, 0, 0, 2),
                    ),
                )
            )
            if not suggestion_repeats_source:
                return Group(*renderables)

        if summary.conclusion:
            renderables.extend(
                (Text(""), Padding(Text(summary.conclusion), (0, 0, 0, 2)))
            )

        if (
            current_definition is not None
            and not suggestion_repeats_source
            and summary.suggested_change
            and suggestion_complete
        ):
            renderables.extend(
                (Text(""), Text("  Suggested change", style="bold"), Text(""))
            )
            renderables.extend(
                Padding(line, (0, 0, 0, 2))
                for line in self._changed_code_renderables(
                    current_definition,
                    summary.suggested_change,
                )
            )
        elif summary.suggested_change and suggestion_complete:
            renderables.extend(
                (Text(""), Text("  Suggested change", style="bold"), Text(""))
            )
            if self.change_analyzer.looks_like_python(summary.suggested_change):
                renderables.append(
                    Syntax(
                        summary.suggested_change,
                        "python",
                        theme="ansi_dark",
                        background_color="default",
                        word_wrap=True,
                        padding=(0, 0, 0, 2),
                    )
                )
            else:
                renderables.append(
                    Padding(Text(summary.suggested_change), (0, 0, 0, 2))
                )
        elif summary.suggested_change:
            renderables.extend(
                (
                    Text(""),
                    Padding(
                        Text("Preparing suggested change…", style="dim"),
                        (0, 0, 0, 2),
                    ),
                )
            )

        if summary.explanation:
            renderables.extend(
                (Text(""), Padding(Text(summary.explanation), (0, 0, 0, 2)))
            )
        return Group(*renderables)

    def display_model_response(
        self,
        result: GenerationResult,
        *,
        selected_chunks: Sequence[CodeChunk] = (),
    ) -> None:
        """Render a complete public evidence summary as one compact finding."""
        self.console.print(
            self.response_renderable(
                result.text,
                final=True,
                selected_chunks=selected_chunks,
            )
        )

    def generate_response(
        self,
        provider: LLMProvider,
        prompt: str,
        *,
        selected_chunks: Sequence[CodeChunk] = (),
    ) -> tuple[GenerationResult, float, float]:
        """Measure generation while revealing buffered model chunks smoothly."""
        first_token: float | None = None
        model_total = 0.0
        result: GenerationResult | None = None
        generation_error: Exception | None = None
        chunks: Queue[str] = Queue()
        status = self.console.status(
            "Analyzing the likely root cause...",
            spinner="dots",
            spinner_style="cyan",
        )
        status.start()
        status_running = True
        model_started_at = self.clock()

        def on_text(text: str) -> None:
            nonlocal first_token
            if text and first_token is None:
                first_token = self.clock() - model_started_at
            if text:
                chunks.put(text)

        def run_generation() -> None:
            nonlocal generation_error, model_total, result
            try:
                result = provider.generate(prompt, on_text=on_text)
            except Exception as error:  # noqa: BLE001 - re-raised on the UI thread
                generation_error = error
            finally:
                model_total = self.clock() - model_started_at

        worker = Thread(target=run_generation, daemon=True)
        worker.start()
        live: Live | None = None
        streamed_text = ""
        pending_text = ""
        try:
            while worker.is_alive() or not chunks.empty() or pending_text:
                if not pending_text:
                    try:
                        pending_text = chunks.get(timeout=self.frame_interval)
                    except Empty:
                        continue
                    while True:
                        try:
                            pending_text += chunks.get_nowait()
                        except Empty:
                            break

                if live is None:
                    status.stop()
                    status_running = False
                    live = Live(
                        self.response_renderable(
                            "",
                            final=False,
                            selected_chunks=selected_chunks,
                        ),
                        console=self.console,
                        refresh_per_second=30,
                    )
                    live.start()

                frame = pending_text[: self.characters_per_frame]
                pending_text = pending_text[self.characters_per_frame :]
                streamed_text += frame
                live.update(
                    self.response_renderable(
                        streamed_text,
                        final=False,
                        selected_chunks=selected_chunks,
                    ),
                    refresh=True,
                )
                self.sleeper(self.frame_interval)

            worker.join()
            if generation_error is not None:
                raise generation_error
            if result is None:
                raise RuntimeError("Model generation ended without a result.")

            if live is None:
                status.stop()
                status_running = False
                self._play_response(result.text, selected_chunks=selected_chunks)
            else:
                live.update(
                    self.response_renderable(
                        result.text,
                        final=True,
                        selected_chunks=selected_chunks,
                    ),
                    refresh=True,
                )
        finally:
            if status_running:
                status.stop()
            if live is not None:
                live.stop()

        if first_token is None:
            first_token = model_total
        return result, first_token, model_total

    def display_footer(
        self,
        result: GenerationResult,
        total_execution: float,
    ) -> None:
        """Render token usage and actual end-to-end execution time."""
        self.console.print()
        if result.total_tokens is None:
            self.console.print("  Token usage unavailable", style="dim")
        else:
            self.console.print(
                f"  {result.input_tokens} input · {result.output_tokens} output · "
                f"{result.total_tokens} total tokens",
                style="dim",
            )
        self.console.print(f"  Completed in {total_execution:.1f}s", style="dim")

    def _play_response(
        self,
        response_text: str,
        *,
        selected_chunks: Sequence[CodeChunk],
    ) -> None:
        """Reveal a completed response at a steady, readable terminal pace."""
        with Live(
            self.response_renderable(
                "",
                final=False,
                selected_chunks=selected_chunks,
            ),
            console=self.console,
            refresh_per_second=30,
        ) as live:
            for end in range(
                self.characters_per_frame,
                len(response_text) + self.characters_per_frame,
                self.characters_per_frame,
            ):
                live.update(
                    self.response_renderable(
                        response_text[:end],
                        final=False,
                        selected_chunks=selected_chunks,
                    ),
                    refresh=True,
                )
                self.sleeper(self.frame_interval)
            live.update(
                self.response_renderable(
                    response_text,
                    final=True,
                    selected_chunks=selected_chunks,
                ),
                refresh=True,
            )

    def _changed_code_renderables(
        self,
        current: SourceDefinition,
        suggested_code: str,
    ) -> list[Text]:
        """Render affected lines as a red path ending at replacement code."""
        current_changed, suggested_changed = self.change_analyzer.changed_line_numbers(
            current.code,
            suggested_code,
        )
        current_lines = current.code.splitlines()
        suggested_lines = suggested_code.splitlines()
        removed_lines = [
            (current.start_line + line - 1, current_lines[line - 1])
            for line in sorted(current_changed)
        ]
        replacement_lines = [
            (current.start_line + line - 1, suggested_lines[line - 1])
            for line in sorted(suggested_changed)
        ]
        displayed_numbers = [number for number, _ in removed_lines]
        displayed_numbers.extend(number for number, _ in replacement_lines)
        width = max((len(str(number)) for number in displayed_numbers), default=1)

        renderables: list[Text] = []
        for line_number, line in removed_lines:
            removed = Text(f"{line_number:>{width}} │ ", style="bold red")
            removed.append(line, style="red")
            renderables.append(removed)

        replacement_line = (
            replacement_lines[0][0]
            if replacement_lines
            else min(displayed_numbers, default=current.start_line)
        )
        connector = Text(f"{replacement_line:>{width}} └────→ ", style="bold red")
        if replacement_lines:
            connector.append(replacement_lines[0][1], style="green")
        else:
            connector.append("remove these lines", style="dim")
        renderables.append(connector)

        for line_number, line in replacement_lines[1:]:
            continuation = Text(f"{line_number:>{width}}        ")
            continuation.append(line, style="green")
            renderables.append(continuation)
        return renderables

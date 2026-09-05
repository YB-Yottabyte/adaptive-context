"""Command-line entry point for the one-shot context debugging baseline."""

from __future__ import annotations

import argparse
import ast
import os
import platform
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter, sleep

import questionary
from questionary import Choice
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.syntax import Syntax
from rich.text import Text

from context_builder import build_debugging_prompt
from providers.base import GenerationResult, LLMProvider
from providers.groq_provider import (
    DEFAULT_GROQ_MODEL,
    GROQ_API_KEY_ENVIRONMENT_VARIABLE,
    GROQ_MODEL_ENVIRONMENT_VARIABLE,
    GroqProvider,
    resolve_groq_model,
)
from providers.local_mlx import DEFAULT_MODEL as DEFAULT_LOCAL_MODEL
from providers.local_mlx import LocalMLXProvider
from retrieval import CodeChunk, create_repository_chunks, select_top_k

MODEL_ENVIRONMENT_VARIABLE = "CONTEXT_DEBUG_MODEL"
TEST_CASES_DIRECTORY = Path(__file__).resolve().parent.parent / "test_cases"
PROVIDER_LABELS = {"local": "Local MLX", "groq": "Groq"}
RESPONSE_CHARACTERS_PER_FRAME = 5
RESPONSE_FRAME_INTERVAL_SECONDS = 0.03
INTRODUCTION = (
    "I'll trace the reported behavior through the most relevant files and look "
    "for the smallest fix."
)
SUMMARY_SECTION_PATTERN = re.compile(
    r"^\s*(Evidence considered|Conclusion|Relevant file|Suggested change|Explanation)"
    r":\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
CONSOLE = Console()


@dataclass(frozen=True)
class RunTimings:
    """Measured pipeline timings and actual end-to-end run duration."""

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


@dataclass(frozen=True)
class DebuggingSummary:
    """Public model output sections used by the compact terminal renderer."""

    evidence: tuple[str, ...]
    conclusion: str
    relevant_file: str
    suggested_change: str
    explanation: str


@dataclass(frozen=True)
class SourceDefinition:
    """A definition extracted from retrieved context with its source location."""

    code: str
    start_line: int


def read_bug_report(test_case_dir: Path) -> str:
    """Read and validate a debugging test case's bug report."""
    if not test_case_dir.exists():
        raise FileNotFoundError(
            f"Test case directory does not exist: {test_case_dir}"
        )
    if not test_case_dir.is_dir():
        raise NotADirectoryError(f"Test case path is not a directory: {test_case_dir}")

    bug_report_path = test_case_dir / "bug_report.txt"
    if not bug_report_path.is_file():
        raise FileNotFoundError(f"Missing bug report: {bug_report_path}")

    bug_report = bug_report_path.read_text(encoding="utf-8").strip()
    if not bug_report:
        raise ValueError(f"Bug report is empty: {bug_report_path}")
    return bug_report


def create_provider(name: str, model_name: str, max_tokens: int) -> LLMProvider:
    """Create the selected language-model provider."""
    if name == "local":
        return LocalMLXProvider(model_name=model_name, max_tokens=max_tokens)
    if name == "groq":
        return GroqProvider(model_name=model_name, max_tokens=max_tokens)
    raise ValueError(f"Unsupported provider: {name}")


def resolve_model(provider_name: str, cli_model: str | None) -> str:
    """Resolve a model without letting one provider's defaults affect another."""
    if provider_name == "groq":
        return resolve_groq_model(cli_model)
    return (
        cli_model or os.environ.get(MODEL_ENVIRONMENT_VARIABLE) or DEFAULT_LOCAL_MODEL
    )


def resolve_test_case(value: str | Path) -> Path:
    """Resolve a short test-case name or preserve an explicit filesystem path."""
    candidate = Path(value)
    if not candidate.is_absolute() and len(candidate.parts) == 1:
        return TEST_CASES_DIRECTORY / candidate
    return candidate


def _module_available(module_name: str) -> bool:
    """Return whether a provider dependency can be imported."""
    return find_spec(module_name) is not None


def _local_runtime_available() -> bool:
    """Return whether this process has the platform and packages needed by MLX."""
    return (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
        and _module_available("mlx")
        and _module_available("mlx_lm")
    )


def get_available_providers(
    *,
    environment: Mapping[str, str] | None = None,
    local_model: str | None = None,
) -> tuple[str, ...]:
    """Return configured providers in primary-baseline-first order."""
    environment = os.environ if environment is None else environment
    available: list[str] = []

    if environment.get(GROQ_API_KEY_ENVIRONMENT_VARIABLE) and _module_available(
        "groq"
    ):
        available.append("groq")

    configured_local_model = (
        local_model
        or environment.get(MODEL_ENVIRONMENT_VARIABLE)
        or DEFAULT_LOCAL_MODEL
    )
    if configured_local_model and _local_runtime_available():
        available.append("local")

    return tuple(available)


def validate_provider(provider_name: str, available_providers: tuple[str, ...]) -> None:
    """Raise a concise setup error when a requested provider cannot run."""
    if provider_name in available_providers:
        return
    if provider_name == "groq":
        raise ValueError("Groq provider is not configured. Set GROQ_API_KEY.")
    if provider_name == "local":
        raise ValueError(
            "Local Hugging Face provider is unavailable. It requires Apple "
            "Silicon and the local model dependencies."
        )
    raise ValueError(f"Unsupported provider: {provider_name}")


def _provider_selection_label(
    provider_name: str,
    *,
    cli_model: str | None,
    environment: Mapping[str, str],
) -> str:
    """Build the concise label used during provider selection."""
    if provider_name == "groq":
        model_name = (
            cli_model
            or environment.get(GROQ_MODEL_ENVIRONMENT_VARIABLE)
            or DEFAULT_GROQ_MODEL
        )
        return f"Groq — {format_model_label(model_name)}"
    return "Local — Hugging Face"


def select_provider(
    requested_provider: str | None,
    available_providers: tuple[str, ...],
    *,
    cli_model: str | None = None,
    environment: Mapping[str, str] | None = None,
    interactive: bool | None = None,
) -> str:
    """Validate, automatically choose, or interactively select a provider."""
    environment = os.environ if environment is None else environment
    if requested_provider is not None:
        validate_provider(requested_provider, available_providers)
        return requested_provider

    if not available_providers:
        raise ValueError(
            "No model providers are available. Set GROQ_API_KEY for Groq, or use "
            "Apple Silicon with the local model dependencies installed."
        )

    if len(available_providers) == 1:
        return available_providers[0]

    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        raise ValueError(
            "Multiple providers are available. Pass --provider groq or "
            "--provider local for a non-interactive run."
        )

    choices = [
        Choice(
            title=_provider_selection_label(
                provider_name,
                cli_model=cli_model,
                environment=environment,
            ),
            value=provider_name,
        )
        for provider_name in available_providers
    ]
    selected = questionary.select(
        "Select model:",
        choices=choices,
        qmark="",
        pointer="❯",
        instruction="↑/↓ to move • Enter to select",
        use_arrow_keys=True,
        use_shortcuts=False,
    ).ask()
    if selected is None:
        raise ValueError("Model selection cancelled.")
    return selected


def _validate_top_k_answer(value: str) -> bool | str:
    """Validate the positive integer entered in the interactive Top-K prompt."""
    try:
        is_valid = int(value) >= 1
    except ValueError:
        is_valid = False
    return True if is_valid else "Enter a whole number greater than zero."


def select_top_k_value(
    requested_top_k: int | None,
    *,
    interactive: bool | None = None,
) -> int:
    """Use an explicit Top-K or ask for one during an interactive run."""
    if requested_top_k is not None:
        if requested_top_k < 1:
            raise ValueError("top_k must be at least 1")
        return requested_top_k

    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        raise ValueError(
            "Top-K was not provided. Pass --top-k with a positive integer for a "
            "non-interactive run."
        )

    answer = questionary.text(
        "Top-K chunks to retrieve:",
        qmark="",
        instruction="Enter a positive integer",
        validate=_validate_top_k_answer,
    ).ask()
    if answer is None:
        raise ValueError("Top-K selection cancelled.")
    return int(answer)


def format_model_label(model_name: str) -> str:
    """Return a concise display label without changing the configured model."""
    short_name = model_name.rsplit("/", maxsplit=1)[-1]
    match = re.fullmatch(r"gpt-oss-(\d+)b", short_name, flags=re.IGNORECASE)
    if match:
        return f"GPT-OSS {match.group(1)}B"
    return short_name


def print_startup(
    provider_name: str,
    model_name: str,
    *,
    console: Console = CONSOLE,
) -> None:
    """Render a compact model introduction before repository work begins."""
    console.print("Context Debugging Baseline", style="bold")
    console.print()
    model_line = (
        f"Using {PROVIDER_LABELS[provider_name]} — {format_model_label(model_name)}"
    )
    console.print(model_line, style="bold")
    console.print()
    console.print(INTRODUCTION)


def _run_timed_stage[T](
    running_message: str,
    action: Callable[[], T],
    *,
    console: Console,
) -> tuple[T, float]:
    """Run one real pipeline action with transient status and timing."""
    with console.status(running_message, spinner="dots", spinner_style="cyan"):
        started_at = perf_counter()
        result = action()
        elapsed = perf_counter() - started_at
    return result, elapsed


def print_retrieved_context(
    source_paths: list[str],
    *,
    console: Console = CONSOLE,
) -> None:
    """Render the selected Top-K paths as one concise completed state."""
    noun = "file" if len(source_paths) == 1 else "files"
    heading = Text("● ", style="green")
    heading.append(f"Retrieved {len(source_paths)} relevant {noun}", style="bold")
    console.print(heading)
    for index, source_path in enumerate(source_paths):
        branch = "└─" if index == len(source_paths) - 1 else "├─"
        console.print(f"  {branch} {source_path}")


def parse_debugging_summary(
    response_text: str,
    *,
    fallback_to_response: bool = True,
) -> DebuggingSummary:
    """Parse the model's public response sections for concise terminal rendering."""
    matches = list(SUMMARY_SECTION_PATTERN.finditer(response_text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        content_end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(response_text)
        )
        sections[match.group(1).lower()] = response_text[match.end() : content_end].strip()

    evidence = tuple(
        line.strip().lstrip("-• ")
        for line in sections.get("evidence considered", "").splitlines()
        if line.strip().lstrip("-• ")
    )
    return DebuggingSummary(
        evidence=evidence,
        conclusion=(
            sections.get("conclusion", "")
            or (response_text.strip() if fallback_to_response else "")
        ),
        relevant_file=sections.get("relevant file", ""),
        suggested_change=sections.get("suggested change", ""),
        explanation=sections.get("explanation", ""),
    )


def _relevant_file_name(relevant_file: str) -> str:
    """Extract the last Python path named by the model."""
    paths = re.findall(r"[\w./-]+\.py\b", relevant_file)
    if paths:
        return paths[-1]
    return relevant_file.splitlines()[0].strip("`* ") if relevant_file else ""


def _looks_like_python(code: str) -> bool:
    """Return whether a suggested replacement is suitable for syntax rendering."""
    starts = ("def ", "async def ", "class ", "return ", "from ", "import ")
    return any(line.lstrip().startswith(starts) for line in code.splitlines())


def _definition_name(code: str) -> str:
    """Return the first function or class name from a suggested replacement."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
    return ""


def _current_definition(
    summary: DebuggingSummary,
    selected_chunks: Sequence[CodeChunk],
) -> SourceDefinition | None:
    """Find the diagnosed definition in the already-retrieved repository context."""
    definition_name = _definition_name(summary.suggested_change)
    relevant_file = _relevant_file_name(summary.relevant_file).removeprefix("./")
    if not definition_name or not relevant_file:
        return None

    exact_chunks = [
        chunk for chunk in selected_chunks if chunk.source_path == relevant_file
    ]
    candidate_chunks = exact_chunks or [
        chunk
        for chunk in selected_chunks
        if Path(chunk.source_path).name == Path(relevant_file).name
    ]
    for chunk in candidate_chunks:
        try:
            tree = ast.parse(chunk.content)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ) and node.name == definition_name:
                lines = chunk.content.splitlines()
                return SourceDefinition(
                    code="\n".join(lines[node.lineno - 1 : node.end_lineno]),
                    start_line=chunk.start_line + node.lineno - 1,
                )
    return None


def _changed_line_numbers(before: str, after: str) -> tuple[set[int], set[int]]:
    """Return one-based changed line numbers for two concise code blocks."""
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    before_changed: set[int] = set()
    after_changed: set[int] = set()
    matcher = SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        before_changed.update(range(before_start + 1, before_end + 1))
        after_changed.update(range(after_start + 1, after_end + 1))
    return before_changed, after_changed


def _python_block(code: str, *, highlight_lines: set[int]) -> Syntax:
    """Build a compact Python block with only changed lines emphasized."""
    return Syntax(
        code,
        "python",
        theme="ansi_dark",
        background_color="default",
        word_wrap=True,
        padding=(0, 0, 0, 2),
        highlight_lines=highlight_lines,
    )


def _changed_code_renderables(
    current: SourceDefinition,
    suggested_code: str,
) -> list[Text]:
    """Render only affected lines as a red path ending at replacement code."""
    current_changed, suggested_changed = _changed_line_numbers(
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
    displayed_line_numbers = [line_number for line_number, _ in removed_lines]
    displayed_line_numbers.extend(line_number for line_number, _ in replacement_lines)
    line_number_width = max(
        (len(str(line_number)) for line_number in displayed_line_numbers),
        default=1,
    )

    renderables: list[Text] = []
    for line_number, line in removed_lines:
        removed = Text(f"{line_number:>{line_number_width}} │ ", style="bold red")
        removed.append(line, style="red")
        renderables.append(removed)

    replacement_line_number = (
        replacement_lines[0][0]
        if replacement_lines
        else min(displayed_line_numbers, default=current.start_line)
    )
    connector = Text(
        f"{replacement_line_number:>{line_number_width}} └────→ ",
        style="bold red",
    )
    if replacement_lines:
        connector.append(replacement_lines[0][1], style="green")
    else:
        connector.append("remove these lines", style="dim")
    renderables.append(connector)

    for line_number, line in replacement_lines[1:]:
        continuation = Text(f"{line_number:>{line_number_width}}        ")
        continuation.append(line, style="green")
        renderables.append(continuation)
    return renderables


def _model_response_renderable(
    response_text: str,
    *,
    final: bool,
    selected_chunks: Sequence[CodeChunk] = (),
) -> Group:
    """Build the progressively renderable public debugging summary."""
    summary = parse_debugging_summary(
        response_text,
        fallback_to_response=final,
    )
    relevant_file = _relevant_file_name(summary.relevant_file)
    heading = Text("● ", style="green")
    if relevant_file:
        heading.append("Diagnosis points to ", style="bold")
        heading.append(relevant_file, style="bold")
    elif not final:
        heading.append("Reviewing the selected evidence", style="bold")
    else:
        heading.append("Diagnosis suggests a likely issue", style="bold")
    renderables: list[RenderableType] = [heading]
    suggestion_complete = final or bool(
        re.search(r"^\s*Explanation:\s*$", response_text, flags=re.IGNORECASE | re.MULTILINE)
    )

    if summary.evidence:
        renderables.extend((Text(""), Text("  Evidence", style="bold")))
        for index, evidence in enumerate(summary.evidence):
            branch = "└─" if index == len(summary.evidence) - 1 else "├─"
            renderables.append(Padding(Text(f"{branch} {evidence}"), (0, 0, 0, 2)))

    if summary.conclusion:
        renderables.extend(
            (Text(""), Padding(Text(summary.conclusion), (0, 0, 0, 2)))
        )

    current_definition = _current_definition(summary, selected_chunks)
    if (
        current_definition is not None
        and summary.suggested_change
        and suggestion_complete
    ):
        renderables.extend((Text(""), Text("  Suggested change", style="bold"), Text("")))
        renderables.extend(
            Padding(line, (0, 0, 0, 2))
            for line in _changed_code_renderables(
                current_definition,
                summary.suggested_change,
            )
        )
    elif summary.suggested_change and suggestion_complete:
        renderables.extend(
            (Text(""), Text("  Suggested replacement", style="bold"), Text(""))
        )
        if _looks_like_python(summary.suggested_change):
            renderables.append(
                _python_block(
                    summary.suggested_change,
                    highlight_lines=set(),
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
                Padding(Text("Preparing suggested change…", style="dim"), (0, 0, 0, 2)),
            )
        )

    if summary.explanation:
        renderables.extend(
            (Text(""), Padding(Text(summary.explanation), (0, 0, 0, 2)))
        )

    return Group(*renderables)


def print_model_response(
    result: GenerationResult,
    *,
    selected_chunks: Sequence[CodeChunk] = (),
    console: Console = CONSOLE,
) -> None:
    """Render the complete public evidence summary as one compact finding."""
    console.print(
        _model_response_renderable(
            result.text,
            final=True,
            selected_chunks=selected_chunks,
        )
    )


def _play_response(
    response_text: str,
    *,
    selected_chunks: Sequence[CodeChunk],
    console: Console,
) -> None:
    """Reveal a completed response at a steady, readable terminal pace."""
    with Live(
        _model_response_renderable(
            "",
            final=False,
            selected_chunks=selected_chunks,
        ),
        console=console,
        refresh_per_second=30,
    ) as live:
        for end in range(
            RESPONSE_CHARACTERS_PER_FRAME,
            len(response_text) + RESPONSE_CHARACTERS_PER_FRAME,
            RESPONSE_CHARACTERS_PER_FRAME,
        ):
            live.update(
                _model_response_renderable(
                    response_text[:end],
                    final=False,
                    selected_chunks=selected_chunks,
                ),
                refresh=True,
            )
            sleep(RESPONSE_FRAME_INTERVAL_SECONDS)
        live.update(
            _model_response_renderable(
                response_text,
                final=True,
                selected_chunks=selected_chunks,
            ),
            refresh=True,
        )


def generate_streaming_response(
    provider: LLMProvider,
    prompt: str,
    *,
    selected_chunks: Sequence[CodeChunk] = (),
    console: Console = CONSOLE,
) -> tuple[GenerationResult, float, float]:
    """Measure the provider's real stream, then reveal its response smoothly."""
    first_token: float | None = None
    status = console.status(
        "Analyzing the likely root cause...",
        spinner="dots",
        spinner_style="cyan",
    )
    status.start()
    model_started_at = perf_counter()

    def on_text(text: str) -> None:
        nonlocal first_token
        if not text:
            return
        if first_token is None:
            first_token = perf_counter() - model_started_at

    try:
        result = provider.generate(prompt, on_text=on_text)
        model_total = perf_counter() - model_started_at
    except Exception:
        status.stop()
        raise

    status.stop()
    if first_token is None:
        first_token = model_total
    _play_response(
        result.text,
        selected_chunks=selected_chunks,
        console=console,
    )
    return result, first_token, model_total


def print_run_footer(
    result: GenerationResult,
    total_execution: float,
    *,
    console: Console = CONSOLE,
) -> None:
    """Render token usage and actual end-to-end execution time."""
    console.print()
    if result.total_tokens is None:
        console.print("  Token usage unavailable", style="dim")
    else:
        console.print(
            f"  {result.input_tokens} input · {result.output_tokens} output · "
            f"{result.total_tokens} total tokens",
            style="dim",
        )
    console.print(f"  Completed in {total_execution:.1f}s", style="dim")


def run_baseline(
    args: argparse.Namespace,
    *,
    console: Console = CONSOLE,
) -> RunTimings:
    """Run one retrieval pass followed by one model generation."""
    run_started_at = perf_counter()
    test_case_dir = args.test_case.resolve()
    print_startup(args.provider, args.model, console=console)
    console.print()

    with console.status(
        "Searching the repository for relevant code...",
        spinner="dots",
        spinner_style="cyan",
    ):
        test_case_started_at = perf_counter()
        bug_report = read_bug_report(test_case_dir)
        test_case_load = perf_counter() - test_case_started_at

        chunking_started_at = perf_counter()
        chunks = create_repository_chunks(test_case_dir)
        chunking_elapsed = perf_counter() - chunking_started_at
        if not chunks:
            raise ValueError(f"No Python source files found in: {test_case_dir}")

        ranking_started_at = perf_counter()
        retrieved = select_top_k(bug_report, chunks, args.top_k)
        ranking_elapsed = perf_counter() - ranking_started_at
    retrieval_elapsed = chunking_elapsed + ranking_elapsed
    print_retrieved_context(
        [item.chunk.source_path for item in retrieved],
        console=console,
    )
    console.print()

    selected_chunks = [item.chunk for item in retrieved]

    def build_context() -> str:
        return build_debugging_prompt(bug_report, selected_chunks)

    prompt, context_elapsed = _run_timed_stage(
        "Tracing the failing behavior through the selected files...",
        build_context,
        console=console,
    )
    console.print()

    provider = create_provider(args.provider, args.model, args.max_tokens)
    result, first_token, model_total = generate_streaming_response(
        provider,
        prompt,
        selected_chunks=selected_chunks,
        console=console,
    )
    total_execution = perf_counter() - run_started_at
    timings = RunTimings(
        test_case_load=test_case_load,
        retrieval=retrieval_elapsed,
        context_build=context_elapsed,
        first_token=first_token,
        model_total=model_total,
        total_execution=total_execution,
    )
    print_run_footer(result, total_execution, console=console)
    return timings


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run one-shot repository retrieval and LLM debugging."
    )
    parser.add_argument(
        "--provider",
        choices=("local", "groq"),
        default=None,
        help=(
            "Language-model provider. Required for reproducible non-interactive "
            "runs when multiple providers are configured."
        ),
    )
    parser.add_argument(
        "test_case",
        metavar="TEST_CASE",
        help=(
            "Test-case name under test_cases/ (for example, payment_bug) or an "
            "explicit repository path."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Provider model override. Otherwise uses CONTEXT_DEBUG_MODEL for local "
            "or GROQ_MODEL for Groq."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of repository chunks to retrieve; prompts when omitted.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens (default: 512).",
    )
    return parser


def main() -> None:
    """Parse command-line arguments and run the baseline."""
    parser = build_argument_parser()
    args = parser.parse_args()
    args.test_case = resolve_test_case(args.test_case)
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")

    try:
        available_providers = get_available_providers(
            local_model=args.model if args.provider == "local" else None
        )
        args.provider = select_provider(
            args.provider,
            available_providers,
            cli_model=args.model,
        )
        args.model = resolve_model(args.provider, args.model)
        args.top_k = select_top_k_value(args.top_k)
        run_baseline(args)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

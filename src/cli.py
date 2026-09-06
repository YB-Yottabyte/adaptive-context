"""Command-line configuration for the context debugging baseline."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import questionary
from questionary import Choice

from pipeline import BaselineRunConfig, BaselineRunner
from providers.factory import ProviderFactory
from providers.groq_provider import DEFAULT_GROQ_MODEL, GROQ_MODEL_ENVIRONMENT_VARIABLE
from terminal_ui import TerminalUI

TEST_CASES_DIRECTORY = Path(__file__).resolve().parent.parent / "test_cases"


class BaselineCLI:
    """Resolve interactive CLI choices and launch a baseline run."""

    def __init__(
        self,
        *,
        provider_factory: ProviderFactory | None = None,
        runner: BaselineRunner | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.environment = os.environ if environment is None else environment
        self.provider_factory = provider_factory or ProviderFactory(self.environment)
        self.runner = runner or BaselineRunner(provider_factory=self.provider_factory)

    def build_parser(self) -> argparse.ArgumentParser:
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
                "Provider model override. Otherwise uses CONTEXT_DEBUG_MODEL for "
                "local or GROQ_MODEL for Groq."
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

    def run(self, argv: Sequence[str] | None = None) -> None:
        """Parse arguments, resolve choices, and run the application service."""
        parser = self.build_parser()
        args = parser.parse_args(argv)
        if args.top_k is not None and args.top_k < 1:
            parser.error("--top-k must be at least 1")
        if args.max_tokens < 1:
            parser.error("--max-tokens must be at least 1")

        try:
            available = self.provider_factory.available_providers(
                local_model=args.model if args.provider == "local" else None
            )
            provider_name = self.select_provider(
                args.provider,
                available,
                cli_model=args.model,
            )
            config = BaselineRunConfig(
                test_case=self.resolve_test_case(args.test_case),
                provider=provider_name,
                model=self.provider_factory.resolve_model(provider_name, args.model),
                top_k=self.select_top_k(args.top_k),
                max_tokens=args.max_tokens,
            )
            self.runner.run(config)
        except (FileNotFoundError, NotADirectoryError, ValueError) as error:
            parser.error(str(error))

    @staticmethod
    def resolve_test_case(value: str | Path) -> Path:
        """Resolve a short test-case name or preserve an explicit path."""
        candidate = Path(value)
        if not candidate.is_absolute() and len(candidate.parts) == 1:
            return TEST_CASES_DIRECTORY / candidate
        return candidate

    def select_provider(
        self,
        requested_provider: str | None,
        available_providers: tuple[str, ...],
        *,
        cli_model: str | None = None,
        interactive: bool | None = None,
    ) -> str:
        """Validate, automatically choose, or interactively select a provider."""
        if requested_provider is not None:
            self.provider_factory.validate(requested_provider, available_providers)
            return requested_provider
        if not available_providers:
            raise ValueError(
                "No model providers are available. Set GROQ_API_KEY for Groq, or "
                "use Apple Silicon with the local model dependencies installed."
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
                title=self._provider_selection_label(name, cli_model=cli_model),
                value=name,
            )
            for name in available_providers
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

    def select_top_k(
        self,
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
                "Top-K was not provided. Pass --top-k with a positive integer for "
                "a non-interactive run."
            )

        answer = questionary.text(
            "Top-K chunks to retrieve:",
            qmark="",
            instruction="Enter a positive integer",
            validate=self.validate_top_k_answer,
        ).ask()
        if answer is None:
            raise ValueError("Top-K selection cancelled.")
        return int(answer)

    @staticmethod
    def validate_top_k_answer(value: str) -> bool | str:
        """Validate the positive integer entered in the Top-K prompt."""
        try:
            is_valid = int(value) >= 1
        except ValueError:
            is_valid = False
        return True if is_valid else "Enter a whole number greater than zero."

    def _provider_selection_label(
        self,
        provider_name: str,
        *,
        cli_model: str | None,
    ) -> str:
        """Build the concise label used during provider selection."""
        if provider_name == "groq":
            model_name = (
                cli_model
                or self.environment.get(GROQ_MODEL_ENVIRONMENT_VARIABLE)
                or DEFAULT_GROQ_MODEL
            )
            return f"Groq — {TerminalUI.format_model_label(model_name)}"
        return "Local — Hugging Face"

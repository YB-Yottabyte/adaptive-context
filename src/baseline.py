"""Executable entry point for the one-shot context debugging baseline."""

from cli import BaselineCLI


def main() -> None:
    """Run the baseline command-line application."""
    BaselineCLI().run()


if __name__ == "__main__":
    main()

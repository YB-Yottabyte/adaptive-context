"""Deterministic repository chunking and shared retrieval data structures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_CHUNK_LINES = 80
DEFAULT_OVERLAP_LINES = 20
IGNORED_DIRECTORIES = {
    ".chroma",
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
GENERATED_PYTHON_SUFFIXES = (".generated.py", "_pb2.py", "_pb2_grpc.py")


@dataclass(frozen=True)
class CodeChunk:
    """A contiguous piece of source code and its repository metadata."""

    source_path: str
    content: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class RetrievedChunk:
    """A code chunk paired with its cosine-similarity score."""

    chunk: CodeChunk
    score: float


class RepositoryRetriever(Protocol):
    """Interface shared by one-pass repository retrieval implementations."""

    name: str
    display_name: str

    def retrieve(
        self,
        query: str,
        chunks: Sequence[CodeChunk],
        top_k: int,
        repository_dir: Path,
    ) -> list[RetrievedChunk]:
        """Return at most Top-K chunks from one retrieval pass."""


class RepositoryChunker:
    """Discover and deterministically chunk Python files in one repository."""

    def __init__(
        self,
        *,
        chunk_lines: int = DEFAULT_CHUNK_LINES,
        overlap_lines: int = DEFAULT_OVERLAP_LINES,
    ) -> None:
        if chunk_lines < 1:
            raise ValueError("chunk_lines must be at least 1")
        if overlap_lines < 0 or overlap_lines >= chunk_lines:
            raise ValueError("overlap_lines must be between 0 and chunk_lines - 1")
        self.chunk_lines = chunk_lines
        self.overlap_lines = overlap_lines

    def find_python_files(self, repository_dir: Path) -> list[Path]:
        """Return repository Python files in deterministic relative-path order."""
        repository_dir = repository_dir.resolve()
        if not repository_dir.exists():
            raise FileNotFoundError(
                f"Test case directory does not exist: {repository_dir}"
            )
        if not repository_dir.is_dir():
            raise NotADirectoryError(
                f"Test case path is not a directory: {repository_dir}"
            )

        files = [
            path
            for path in repository_dir.rglob("*.py")
            if path.is_file()
            and not path.name.startswith(".")
            and not path.name.endswith(GENERATED_PYTHON_SUFFIXES)
            and not any(
                part in IGNORED_DIRECTORIES or part.startswith(".")
                for part in path.relative_to(repository_dir).parts[:-1]
            )
        ]
        return sorted(
            files,
            key=lambda path: path.relative_to(repository_dir).as_posix(),
        )

    def chunk_file(self, file_path: Path, repository_dir: Path) -> list[CodeChunk]:
        """Split one source file into overlapping, contiguous line chunks."""
        lines = file_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return []

        relative_path = (
            file_path.resolve().relative_to(repository_dir.resolve()).as_posix()
        )
        step = self.chunk_lines - self.overlap_lines
        chunks: list[CodeChunk] = []
        for start_index in range(0, len(lines), step):
            selected_lines = lines[start_index : start_index + self.chunk_lines]
            if not selected_lines:
                break
            chunks.append(
                CodeChunk(
                    source_path=relative_path,
                    content="\n".join(selected_lines),
                    start_line=start_index + 1,
                    end_line=start_index + len(selected_lines),
                )
            )
            if start_index + self.chunk_lines >= len(lines):
                break
        return chunks

    def create_chunks(self, repository_dir: Path) -> list[CodeChunk]:
        """Discover Python files and convert them to retrievable chunks."""
        chunks: list[CodeChunk] = []
        for file_path in self.find_python_files(repository_dir):
            chunks.extend(self.chunk_file(file_path, repository_dir))
        return chunks

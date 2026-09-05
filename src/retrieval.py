"""One-shot repository chunking and TF-IDF retrieval utilities."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHUNK_LINES = 80
DEFAULT_OVERLAP_LINES = 20
IGNORED_DIRECTORIES = {".git", ".idea", ".venv", "__pycache__"}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


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


def find_python_files(repository_dir: Path) -> list[Path]:
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
        and not any(
            part in IGNORED_DIRECTORIES or part.startswith(".")
            for part in path.relative_to(repository_dir).parts[:-1]
        )
    ]
    return sorted(files, key=lambda path: path.relative_to(repository_dir).as_posix())


def chunk_file(
    file_path: Path,
    repository_dir: Path,
    *,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
) -> list[CodeChunk]:
    """Split one source file into overlapping, contiguous line chunks."""
    if chunk_lines < 1:
        raise ValueError("chunk_lines must be at least 1")
    if overlap_lines < 0 or overlap_lines >= chunk_lines:
        raise ValueError("overlap_lines must be between 0 and chunk_lines - 1")

    lines = file_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []

    relative_path = file_path.resolve().relative_to(repository_dir.resolve()).as_posix()
    step = chunk_lines - overlap_lines
    chunks: list[CodeChunk] = []
    for start_index in range(0, len(lines), step):
        selected_lines = lines[start_index : start_index + chunk_lines]
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
        if start_index + chunk_lines >= len(lines):
            break
    return chunks


def create_repository_chunks(
    repository_dir: Path,
    *,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
) -> list[CodeChunk]:
    """Discover Python files and convert them to retrievable chunks."""
    chunks: list[CodeChunk] = []
    for file_path in find_python_files(repository_dir):
        chunks.extend(
            chunk_file(
                file_path,
                repository_dir,
                chunk_lines=chunk_lines,
                overlap_lines=overlap_lines,
            )
        )
    return chunks


def _tokenize(text: str) -> list[str]:
    """Tokenize natural language and code, splitting snake_case identifiers."""
    return TOKEN_PATTERN.findall(text.lower())


def _inverse_document_frequencies(documents: Sequence[list[str]]) -> dict[str, float]:
    document_count = len(documents)
    document_frequencies: Counter[str] = Counter()
    for document in documents:
        document_frequencies.update(set(document))

    return {
        term: math.log((1 + document_count) / (1 + frequency)) + 1
        for term, frequency in document_frequencies.items()
    }


def _tfidf_vector(tokens: Iterable[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(token for token in tokens if token in idf)
    return {term: count * idf[term] for term, count in counts.items()}


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0

    dot_product = sum(weight * right.get(term, 0.0) for term, weight in left.items())
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def rank_chunks(query: str, chunks: Sequence[CodeChunk]) -> list[RetrievedChunk]:
    """Rank chunks once by TF-IDF cosine similarity to the query."""
    if not chunks:
        return []

    tokenized_chunks = [_tokenize(chunk.content) for chunk in chunks]
    idf = _inverse_document_frequencies(tokenized_chunks)
    query_vector = _tfidf_vector(_tokenize(query), idf)

    ranked = [
        RetrievedChunk(
            chunk=chunk,
            score=_cosine_similarity(
                query_vector,
                _tfidf_vector(tokens, idf),
            ),
        )
        for chunk, tokens in zip(chunks, tokenized_chunks, strict=True)
    ]
    return sorted(
        ranked,
        key=lambda item: (
            -item.score,
            item.chunk.source_path,
            item.chunk.start_line,
            item.chunk.end_line,
        ),
    )


def select_top_k(
    query: str,
    chunks: Sequence[CodeChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Return up to Top-K chunks from one deterministic retrieval pass."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    return rank_chunks(query, chunks)[:top_k]

"""Deterministic context-size estimates and retrieval evaluation models."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil

from debugging_case import EvaluationMetadata
from retrieval import CodeChunk, RepositoryIndex


class CharacterTokenEstimator:
    """Estimate tokens consistently as approximately four characters each."""

    characters_per_token = 4

    def estimate(self, text: str) -> int:
        """Return a deterministic approximate token count for non-empty text."""
        return ceil(len(text) / self.characters_per_token) if text else 0


@dataclass(frozen=True)
class ContextMetrics:
    """Repository indexing and selected-context measurements."""

    source_files: int
    chunks_indexed: int
    chunks_selected: int
    repository_source_tokens: int
    retrieved_context_tokens: int
    context_selected_percent: float


@dataclass(frozen=True)
class GoldFileMatch:
    """The retrieval rank for one metadata-provided gold file."""

    source_path: str
    rank: int | None


@dataclass(frozen=True)
class RetrievalEvaluation:
    """Post-retrieval comparison against optional gold patch paths."""

    matches: tuple[GoldFileMatch, ...]

    @property
    def gold_file_retrieved(self) -> bool:
        return any(match.rank is not None for match in self.matches)

    @property
    def best_gold_file_rank(self) -> int | None:
        ranks = [match.rank for match in self.matches if match.rank is not None]
        return min(ranks) if ranks else None


class ContextMetricsCalculator:
    """Measure indexed and selected context without provider-specific tokenizers."""

    def __init__(self, estimator: CharacterTokenEstimator | None = None) -> None:
        self.estimator = estimator or CharacterTokenEstimator()

    def calculate(
        self,
        repository_index: RepositoryIndex,
        selected_chunks: list[CodeChunk],
        formatted_context: str,
    ) -> ContextMetrics:
        """Calculate reproducible estimates for one retrieval result."""
        repository_text = "\n".join(
            document.content for document in repository_index.documents
        )
        repository_tokens = self.estimator.estimate(repository_text)
        retrieved_tokens = self.estimator.estimate(formatted_context)
        selected_percent = self._source_coverage_percent(
            repository_index,
            selected_chunks,
        )
        return ContextMetrics(
            source_files=repository_index.source_file_count,
            chunks_indexed=len(repository_index.chunks),
            chunks_selected=len(selected_chunks),
            repository_source_tokens=repository_tokens,
            retrieved_context_tokens=retrieved_tokens,
            context_selected_percent=selected_percent,
        )

    @staticmethod
    def _source_coverage_percent(
        repository_index: RepositoryIndex,
        selected_chunks: list[CodeChunk],
    ) -> float:
        """Return unique selected source-line coverage without counting overlap."""
        documents = {
            document.source_path: document.content
            for document in repository_index.documents
        }
        selected_lines: dict[str, set[int]] = defaultdict(set)
        for chunk in selected_chunks:
            selected_lines[chunk.source_path].update(
                range(chunk.start_line, chunk.end_line + 1)
            )

        repository_characters = sum(len(content) for content in documents.values())
        if repository_characters == 0:
            return 0.0

        selected_characters = 0
        for source_path, line_numbers in selected_lines.items():
            content = documents.get(source_path)
            if content is None:
                continue
            lines = content.splitlines(keepends=True)
            selected_characters += sum(
                len(lines[line_number - 1])
                for line_number in line_numbers
                if 1 <= line_number <= len(lines)
            )
        return selected_characters / repository_characters * 100

    @staticmethod
    def evaluate_retrieval(
        metadata: EvaluationMetadata,
        selected_chunks: list[CodeChunk],
    ) -> RetrievalEvaluation:
        """Compare selected chunk paths with evaluation-only gold paths."""
        ranks_by_path: dict[str, int] = {}
        for rank, chunk in enumerate(selected_chunks, start=1):
            normalized = chunk.source_path.replace("\\", "/").removeprefix("./")
            ranks_by_path.setdefault(normalized, rank)
        matches = tuple(
            GoldFileMatch(path, ranks_by_path.get(path))
            for path in metadata.gold_patch_files
        )
        return RetrievalEvaluation(matches)

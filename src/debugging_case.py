"""Input models and loaders shared by controlled and real repository runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class CaseMode(str, Enum):
    """Supported debugging input workflows."""

    CONTROLLED = "controlled"
    REPOSITORY = "repository"


@dataclass(frozen=True)
class EvaluationMetadata:
    """Optional post-retrieval ground truth that is never sent to the model."""

    repo: str | None = None
    issue_number: int | str | None = None
    base_commit: str | None = None
    gold_patch_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class DebuggingCase:
    """Common input consumed by the one-shot retrieval and model pipeline."""

    repository_path: Path
    bug_report: str
    display_name: str
    mode: CaseMode
    metadata: EvaluationMetadata | None = None

    @property
    def is_real_repository(self) -> bool:
        """Return whether this case uses a separate issue file and repository."""
        return self.mode is CaseMode.REPOSITORY


class DebuggingCaseLoader:
    """Validate and load either supported input workflow."""

    def load_controlled(self, repository_path: Path) -> DebuggingCase:
        """Load a self-contained controlled case with bug_report.txt."""
        repository_path = self._validate_repository(repository_path)
        bug_report = self._read_text_file(
            repository_path / "bug_report.txt",
            missing_label="Missing bug report",
            empty_label="Bug report is empty",
        )
        return DebuggingCase(
            repository_path=repository_path,
            bug_report=bug_report,
            display_name=repository_path.name,
            mode=CaseMode.CONTROLLED,
        )

    def load_repository(
        self,
        repository_path: Path,
        issue_file: Path,
        metadata_file: Path | None = None,
    ) -> DebuggingCase:
        """Load a local repository and a separate issue report."""
        repository_path = self._validate_repository(repository_path)
        issue_file = issue_file.resolve()
        bug_report = self._read_text_file(
            issue_file,
            missing_label="Issue file does not exist",
            empty_label="Issue file is empty",
        )
        metadata = self.load_metadata(metadata_file) if metadata_file else None
        return DebuggingCase(
            repository_path=repository_path,
            bug_report=bug_report,
            display_name=(
                metadata.repo if metadata and metadata.repo else repository_path.name
            ),
            mode=CaseMode.REPOSITORY,
            metadata=metadata,
        )

    def load_metadata(self, metadata_file: Path) -> EvaluationMetadata:
        """Load optional evaluation-only metadata from a JSON object."""
        metadata_file = metadata_file.resolve()
        raw_text = self._read_text_file(
            metadata_file,
            missing_label="Metadata file does not exist",
            empty_label="Metadata file is empty",
        )
        try:
            raw_data: Any = json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Metadata file is not valid JSON: {metadata_file}"
            ) from error
        if not isinstance(raw_data, dict):
            raise ValueError("Metadata must contain a JSON object.")  # noqa: TRY004

        gold_files = raw_data.get("gold_patch_files", [])
        if not isinstance(gold_files, list) or not all(
            isinstance(path, str) and path.strip() for path in gold_files
        ):
            raise ValueError("metadata gold_patch_files must be a list of paths")

        return EvaluationMetadata(
            repo=self._optional_string(raw_data, "repo"),
            issue_number=self._optional_issue_number(raw_data.get("issue_number")),
            base_commit=self._optional_string(raw_data, "base_commit"),
            gold_patch_files=tuple(self._normalize_path(path) for path in gold_files),
        )

    @staticmethod
    def _validate_repository(repository_path: Path) -> Path:
        repository_path = repository_path.resolve()
        if not repository_path.exists():
            raise FileNotFoundError(
                f"Repository directory does not exist: {repository_path}"
            )
        if not repository_path.is_dir():
            raise NotADirectoryError(
                f"Repository path is not a directory: {repository_path}"
            )
        return repository_path

    @staticmethod
    def _read_text_file(
        file_path: Path,
        *,
        missing_label: str,
        empty_label: str,
    ) -> str:
        if not file_path.is_file():
            raise FileNotFoundError(f"{missing_label}: {file_path}")
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"{empty_label}: {file_path}")
        return content

    @staticmethod
    def _optional_string(data: dict[str, Any], key: str) -> str | None:
        value = data.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"metadata {key} must be a string")  # noqa: TRY004
        return value

    @staticmethod
    def _optional_issue_number(value: Any) -> int | str | None:
        if value is None or isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip():
            return value
        raise ValueError("metadata issue_number must be an integer or string")

    @staticmethod
    def _normalize_path(value: str) -> str:
        return value.strip().replace("\\", "/").removeprefix("./")

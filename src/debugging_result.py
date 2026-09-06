"""Parse model output and compare suggested code with retrieved source."""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from retrieval import CodeChunk

SUMMARY_SECTION_PATTERN = re.compile(
    r"^\s*(Evidence considered|Conclusion|Relevant file|Suggested change|Explanation)"
    r":\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class DebuggingSummary:
    """Public model output sections used by the terminal renderer."""

    evidence: tuple[str, ...]
    conclusion: str
    relevant_file: str
    suggested_change: str
    explanation: str

    @property
    def is_complete(self) -> bool:
        """Return whether every required public diagnosis section has content."""
        return all(
            (
                self.evidence,
                self.conclusion,
                self.relevant_file,
                self.suggested_change,
                self.explanation,
            )
        )


@dataclass(frozen=True)
class SourceDefinition:
    """A definition extracted from retrieved context with its source location."""

    code: str
    start_line: int


class DebuggingSummaryParser:
    """Convert the model's text contract into a structured debugging result."""

    def parse(
        self,
        response_text: str,
        *,
        fallback_to_response: bool = True,
    ) -> DebuggingSummary:
        """Parse the public response sections used for terminal presentation."""
        matches = list(SUMMARY_SECTION_PATTERN.finditer(response_text))
        sections: dict[str, str] = {}
        for index, match in enumerate(matches):
            content_end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(response_text)
            )
            sections[match.group(1).lower()] = response_text[
                match.end() : content_end
            ].strip()

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
            suggested_change=self._strip_code_fence(
                sections.get("suggested change", "")
            ),
            explanation=sections.get("explanation", ""),
        )

    @staticmethod
    def _strip_code_fence(value: str) -> str:
        """Normalize a fenced model suggestion into plain source code."""
        lines = value.strip().splitlines()
        if (
            len(lines) >= 2
            and re.fullmatch(r"```(?:python|py)?", lines[0].strip(), re.IGNORECASE)
            and lines[-1].strip() == "```"
        ):
            return "\n".join(lines[1:-1]).strip()
        return value

    def relevant_file_name(self, relevant_file: str) -> str:
        """Extract the last Python path named by the model."""
        paths = re.findall(r"[\w./-]+\.py\b", relevant_file)
        if paths:
            return paths[-1]
        return relevant_file.splitlines()[0].strip("`* ") if relevant_file else ""


class SourceChangeAnalyzer:
    """Locate and line-diff a suggested definition in retrieved code."""

    def __init__(self, parser: DebuggingSummaryParser | None = None) -> None:
        self.parser = parser or DebuggingSummaryParser()

    def find_current_definition(
        self,
        summary: DebuggingSummary,
        selected_chunks: Sequence[CodeChunk],
    ) -> SourceDefinition | None:
        """Find the diagnosed definition in already-retrieved context."""
        definition_name = self._definition_name(summary.suggested_change)
        relevant_file = self.parser.relevant_file_name(
            summary.relevant_file
        ).removeprefix("./")
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
                if (
                    isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                    )
                    and node.name == definition_name
                ):
                    lines = chunk.content.splitlines()
                    return SourceDefinition(
                        code="\n".join(lines[node.lineno - 1 : node.end_lineno]),
                        start_line=chunk.start_line + node.lineno - 1,
                    )
        return None

    def changed_line_numbers(
        self,
        before: str,
        after: str,
    ) -> tuple[set[int], set[int]]:
        """Return one-based changed line numbers for two concise code blocks."""
        before_lines = before.splitlines()
        after_lines = after.splitlines()
        before_changed: set[int] = set()
        after_changed: set[int] = set()
        matcher = SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
        for (
            tag,
            before_start,
            before_end,
            after_start,
            after_end,
        ) in matcher.get_opcodes():
            if tag == "equal":
                continue
            before_changed.update(range(before_start + 1, before_end + 1))
            after_changed.update(range(after_start + 1, after_end + 1))
        return before_changed, after_changed

    @staticmethod
    def looks_like_python(code: str) -> bool:
        """Return whether a replacement is suitable for syntax rendering."""
        starts = ("def ", "async def ", "class ", "return ", "from ", "import ")
        return any(line.lstrip().startswith(starts) for line in code.splitlines())

    @staticmethod
    def _definition_name(code: str) -> str:
        """Return the first function or class name from replacement code."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ""
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return node.name
        return ""

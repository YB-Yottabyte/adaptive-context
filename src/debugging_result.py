"""Parse model output and compare suggested code with retrieved source."""

from __future__ import annotations

import ast
import re
import textwrap
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

    source_path: str
    code: str
    start_line: int


@dataclass(frozen=True)
class SourceChange:
    """A proposed replacement mapped to source present in retrieved context."""

    current: SourceDefinition
    suggested_code: str


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

    def find_source_change(
        self,
        summary: DebuggingSummary,
        selected_chunks: Sequence[CodeChunk],
    ) -> SourceChange | None:
        """Map a model suggestion to unambiguous retrieved source when possible."""
        current_definition = self.find_current_definition(summary, selected_chunks)
        if current_definition is not None:
            return SourceChange(current_definition, summary.suggested_change)
        return self._find_labeled_source_change(
            summary,
            selected_chunks,
        ) or self._find_prefixed_source_change(
            summary,
            selected_chunks,
        ) or self._find_contextual_source_change(summary, selected_chunks)

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
        definitions: dict[tuple[str, int, str], SourceDefinition] = {}
        for chunk in candidate_chunks:
            definition = self._find_definition_in_chunk(chunk, definition_name)
            if definition is not None:
                key = (
                    definition.source_path,
                    definition.start_line,
                    definition.code,
                )
                definitions[key] = definition

        if len(definitions) == 1:
            return next(iter(definitions.values()))
        return None

    def _find_labeled_source_change(
        self,
        summary: DebuggingSummary,
        selected_chunks: Sequence[CodeChunk],
    ) -> SourceChange | None:
        """Map labeled Original/Updated model blocks to retrieved source."""
        blocks = self._labeled_change_blocks(summary.suggested_change)
        relevant_file = self.parser.relevant_file_name(
            summary.relevant_file
        ).removeprefix("./")
        if blocks is None or not relevant_file:
            return None
        original_code, updated_code = blocks
        candidate_chunks = self._matching_file_chunks(relevant_file, selected_chunks)
        if not candidate_chunks:
            return None

        exact_matches = self._exact_block_matches(
            original_code,
            updated_code,
            candidate_chunks,
        )
        if len(exact_matches) == 1:
            return next(iter(exact_matches.values()))

        replacement = self._single_line_replacement(original_code, updated_code)
        if replacement is None:
            return None
        return self._find_fragment_source_change(replacement, candidate_chunks)

    def _find_prefixed_source_change(
        self,
        summary: DebuggingSummary,
        selected_chunks: Sequence[CodeChunk],
    ) -> SourceChange | None:
        """Map one model-generated minus/plus replacement to retrieved source."""
        relevant_file = self.parser.relevant_file_name(
            summary.relevant_file
        ).removeprefix("./")
        if not relevant_file:
            return None
        candidate_chunks = self._matching_file_chunks(relevant_file, selected_chunks)
        if not candidate_chunks:
            return None

        removed_lines: list[str] = []
        added_lines: list[str] = []
        for line in summary.suggested_change.splitlines():
            removed = re.match(r"^\s*-(?!--)(?P<code>.*)$", line)
            added = re.match(r"^\s*\+(?!\+\+)(?P<code>.*)$", line)
            if removed:
                removed_lines.append(removed.group("code"))
            elif added:
                added_lines.append(added.group("code"))
        if not removed_lines or not added_lines:
            return None

        removed_code = "\n".join(removed_lines)
        added_code = "\n".join(added_lines)
        exact_matches = self._exact_block_matches(
            removed_code,
            added_code,
            candidate_chunks,
        )
        if len(exact_matches) == 1:
            return next(iter(exact_matches.values()))
        if len(removed_lines) != 1 or len(added_lines) != 1:
            return None
        replacement = self._single_line_replacement(removed_code, added_code)
        if replacement is None:
            return None
        return self._find_fragment_source_change(replacement, candidate_chunks)

    def _find_contextual_source_change(
        self,
        summary: DebuggingSummary,
        selected_chunks: Sequence[CodeChunk],
    ) -> SourceChange | None:
        """Map one edited line surrounded by exact retrieved-file context."""
        relevant_file = self.parser.relevant_file_name(
            summary.relevant_file
        ).removeprefix("./")
        if not relevant_file:
            return None
        candidate_chunks = self._matching_file_chunks(relevant_file, selected_chunks)
        suggested_lines = summary.suggested_change.strip().splitlines()
        if not candidate_chunks or len(suggested_lines) < 3:
            return None

        matches: dict[tuple[str, int, str, str], SourceChange] = {}
        window_size = len(suggested_lines)
        for chunk in candidate_chunks:
            source_lines = chunk.content.splitlines()
            for start_index in range(len(source_lines) - window_size + 1):
                actual_lines = source_lines[start_index : start_index + window_size]
                differing = [
                    index
                    for index, (actual, suggested) in enumerate(
                        zip(actual_lines, suggested_lines, strict=True)
                    )
                    if actual.strip() != suggested.strip()
                ]
                matching_nonblank = sum(
                    bool(actual.strip())
                    for actual, suggested in zip(
                        actual_lines,
                        suggested_lines,
                        strict=True,
                    )
                    if actual.strip() == suggested.strip()
                )
                if len(differing) != 1 or matching_nonblank < 2:
                    continue

                changed_index = differing[0]
                current_line = actual_lines[changed_index]
                proposed_content = suggested_lines[changed_index].strip()
                if not current_line.strip() or not proposed_content:
                    continue
                source_indent = re.match(r"^[ \t]*", current_line).group()
                suggested_line = f"{source_indent}{proposed_content}"
                start_line = chunk.start_line + start_index + changed_index
                change = SourceChange(
                    current=SourceDefinition(
                        source_path=chunk.source_path,
                        code=current_line,
                        start_line=start_line,
                    ),
                    suggested_code=suggested_line,
                )
                key = (
                    chunk.source_path,
                    start_line,
                    current_line,
                    suggested_line,
                )
                matches[key] = change
        if len(matches) == 1:
            return next(iter(matches.values()))
        return None

    @staticmethod
    def _find_fragment_source_change(
        replacement: tuple[str, str, str],
        candidate_chunks: Sequence[CodeChunk],
    ) -> SourceChange | None:
        """Apply one unambiguous proposed fragment replacement to actual source."""
        old_fragment, new_fragment, shared_prefix = replacement
        fragment_matches: dict[tuple[str, int, str, str], SourceChange] = {}
        for chunk in candidate_chunks:
            for line_index, source_line in enumerate(chunk.content.splitlines()):
                if source_line.count(old_fragment) != 1:
                    continue
                before_fragment = source_line.partition(old_fragment)[0]
                if shared_prefix.strip() and not before_fragment.rstrip().endswith(
                    shared_prefix.strip()
                ):
                    continue
                suggested_line = source_line.replace(
                    old_fragment,
                    new_fragment,
                    1,
                )
                start_line = chunk.start_line + line_index
                change = SourceChange(
                    current=SourceDefinition(
                        source_path=chunk.source_path,
                        code=source_line,
                        start_line=start_line,
                    ),
                    suggested_code=suggested_line,
                )
                key = (
                    chunk.source_path,
                    start_line,
                    source_line,
                    suggested_line,
                )
                fragment_matches[key] = change
        if len(fragment_matches) == 1:
            return next(iter(fragment_matches.values()))
        return None

    @staticmethod
    def _matching_file_chunks(
        relevant_file: str,
        selected_chunks: Sequence[CodeChunk],
    ) -> list[CodeChunk]:
        """Return exact-path chunks, or unambiguous basename candidates."""
        exact_chunks = [
            chunk for chunk in selected_chunks if chunk.source_path == relevant_file
        ]
        if exact_chunks:
            return exact_chunks
        basename_chunks = [
            chunk
            for chunk in selected_chunks
            if Path(chunk.source_path).name == Path(relevant_file).name
        ]
        paths = {chunk.source_path for chunk in basename_chunks}
        return basename_chunks if len(paths) == 1 else []

    @classmethod
    def _labeled_change_blocks(cls, suggestion: str) -> tuple[str, str] | None:
        """Extract model-labeled Original and Updated code sections."""
        lines = suggestion.splitlines()
        original_marker = re.compile(
            r"^\s*#?\s*Original (?:code|snippet)"
            r"(?:\s*\(simplified\))?\s*:?\s*$",
            re.IGNORECASE,
        )
        updated_marker = re.compile(
            r"^\s*#?\s*Updated (?:code|snippet)\s*:?\s*$",
            re.IGNORECASE,
        )
        original_index = next(
            (index for index, line in enumerate(lines) if original_marker.match(line)),
            None,
        )
        if original_index is None:
            return None
        updated_index = next(
            (
                index
                for index, line in enumerate(
                    lines[original_index + 1 :], original_index + 1
                )
                if updated_marker.match(line)
            ),
            None,
        )
        if updated_index is None:
            return None

        original_lines = cls._uncomment_block(lines[original_index + 1 : updated_index])
        original_code = "\n".join(original_lines).strip("\n")
        updated_code = "\n".join(lines[updated_index + 1 :]).strip("\n")
        if not original_code.strip() or not updated_code.strip():
            return None
        return original_code, updated_code

    @staticmethod
    def _uncomment_block(lines: list[str]) -> list[str]:
        """Remove one presentation-only comment prefix from an original block."""
        nonempty = [line for line in lines if line.strip()]
        if not nonempty or not all(re.match(r"^\s*#", line) for line in nonempty):
            return lines
        uncommented: list[str] = []
        for line in lines:
            match = re.match(r"^(?P<indent>\s*)# ?(?P<content>.*)$", line)
            uncommented.append(
                f"{match.group('indent')}{match.group('content')}" if match else line
            )
        return uncommented

    @classmethod
    def _exact_block_matches(
        cls,
        original_code: str,
        updated_code: str,
        chunks: Sequence[CodeChunk],
    ) -> dict[tuple[str, int, str, str], SourceChange]:
        """Find exact contiguous original blocks in retrieved file chunks."""
        original_lines = cls._normalized_lines(original_code)
        updated_lines = cls._normalized_lines(updated_code)
        if not original_lines or not updated_lines:
            return {}
        matches: dict[tuple[str, int, str, str], SourceChange] = {}
        for chunk in chunks:
            source_lines = chunk.content.splitlines()
            window_size = len(original_lines)
            for start_index in range(len(source_lines) - window_size + 1):
                actual_lines = source_lines[start_index : start_index + window_size]
                if cls._normalized_lines("\n".join(actual_lines)) != original_lines:
                    continue
                base_indent = re.match(r"^[ \t]*", actual_lines[0]).group()
                suggested_lines = [
                    f"{base_indent}{line}" if line else "" for line in updated_lines
                ]
                current_code = "\n".join(actual_lines)
                suggested_code = "\n".join(suggested_lines)
                start_line = chunk.start_line + start_index
                change = SourceChange(
                    current=SourceDefinition(
                        source_path=chunk.source_path,
                        code=current_code,
                        start_line=start_line,
                    ),
                    suggested_code=suggested_code,
                )
                key = (
                    chunk.source_path,
                    start_line,
                    current_code,
                    suggested_code,
                )
                matches[key] = change
        return matches

    @staticmethod
    def _normalized_lines(code: str) -> list[str]:
        """Normalize outer indentation while preserving code within a block."""
        return textwrap.dedent(code).strip().splitlines()

    @classmethod
    def _single_line_replacement(
        cls,
        original_code: str,
        updated_code: str,
    ) -> tuple[str, str, str] | None:
        """Derive one changed fragment from otherwise identical labeled blocks."""
        original_lines = cls._normalized_lines(original_code)
        updated_lines = cls._normalized_lines(updated_code)
        matcher = SequenceMatcher(a=original_lines, b=updated_lines, autojunk=False)
        changes = [opcode for opcode in matcher.get_opcodes() if opcode[0] != "equal"]
        if len(changes) != 1:
            return None
        tag, old_start, old_end, new_start, new_end = changes[0]
        if tag != "replace" or old_end - old_start != 1 or new_end - new_start != 1:
            return None

        old_line = original_lines[old_start]
        new_line = updated_lines[new_start]
        prefix_length = 0
        while (
            prefix_length < min(len(old_line), len(new_line))
            and old_line[prefix_length] == new_line[prefix_length]
        ):
            prefix_length += 1
        suffix_length = 0
        remaining = min(len(old_line), len(new_line)) - prefix_length
        while (
            suffix_length < remaining
            and old_line[-suffix_length - 1] == new_line[-suffix_length - 1]
        ):
            suffix_length += 1

        old_end_index = (
            len(old_line) - suffix_length if suffix_length else len(old_line)
        )
        new_end_index = (
            len(new_line) - suffix_length if suffix_length else len(new_line)
        )
        old_fragment = old_line[prefix_length:old_end_index]
        new_fragment = new_line[prefix_length:new_end_index]
        shared_prefix = old_line[:prefix_length]
        if (
            len(old_fragment.strip()) < 4
            or not new_fragment.strip()
            or not shared_prefix.strip()
        ):
            return None
        return old_fragment, new_fragment, shared_prefix

    def _find_definition_in_chunk(
        self,
        chunk: CodeChunk,
        definition_name: str,
    ) -> SourceDefinition | None:
        """Extract one complete named definition from actual chunk source."""
        matches: dict[tuple[int, str], SourceDefinition] = {}
        try:
            tree = ast.parse(chunk.content)
        except SyntaxError:
            tree = None
        if tree is not None:
            lines = chunk.content.splitlines()
            for node in ast.walk(tree):
                if (
                    isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                    )
                    and node.name == definition_name
                ):
                    code = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                    start_line = chunk.start_line + node.lineno - 1
                    matches[(start_line, code)] = SourceDefinition(
                        source_path=chunk.source_path,
                        code=code,
                        start_line=start_line,
                    )

        if not matches:
            bounded = self._find_bounded_definition(chunk, definition_name)
            if bounded is not None:
                matches[(bounded.start_line, bounded.code)] = bounded

        if len(matches) == 1:
            return next(iter(matches.values()))
        return None

    @staticmethod
    def _find_bounded_definition(
        chunk: CodeChunk,
        definition_name: str,
    ) -> SourceDefinition | None:
        """Find a complete definition within a non-standalone source window."""
        lines = chunk.content.splitlines()
        header_pattern = re.compile(
            rf"^(?P<indent>[ \t]*)(?:async[ \t]+def|def|class)"
            rf"[ \t]+{re.escape(definition_name)}\b"
        )
        candidates: list[SourceDefinition] = []
        for start_index, line in enumerate(lines):
            header = header_pattern.match(line)
            if header is None:
                continue

            definition_indent = len(header.group("indent").expandtabs())
            end_index: int | None = None
            for index in range(start_index + 1, len(lines)):
                candidate = lines[index]
                if not candidate.strip() or candidate.lstrip().startswith("#"):
                    continue
                indentation = len(candidate) - len(candidate.lstrip(" \t"))
                indentation = len(candidate[:indentation].expandtabs())
                if indentation <= definition_indent:
                    end_index = index
                    break

            # Without a following source boundary, the retrieved window may have
            # truncated the definition. Keep the plain-code fallback in that case.
            if end_index is None:
                continue

            candidate_lines = lines[start_index:end_index]
            candidate_code = "\n".join(candidate_lines).rstrip()
            try:
                tree = ast.parse(textwrap.dedent(candidate_code))
            except SyntaxError:
                continue
            if len(tree.body) != 1:
                continue
            node = tree.body[0]
            if not (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == definition_name
            ):
                continue
            complete_code = "\n".join(candidate_lines[: node.end_lineno])
            candidates.append(
                SourceDefinition(
                    source_path=chunk.source_path,
                    code=complete_code,
                    start_line=chunk.start_line + start_index,
                )
            )

        unique = {
            (candidate.start_line, candidate.code): candidate
            for candidate in candidates
        }
        if len(unique) == 1:
            return next(iter(unique.values()))
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
            tree = ast.parse(textwrap.dedent(code))
        except SyntaxError:
            return ""
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return node.name
        return ""

"""Prompt construction for repository debugging."""

from collections.abc import Sequence

from retrieval import CodeChunk


def format_repository_context(chunks: Sequence[CodeChunk]) -> str:
    """Format selected chunks with their repository-relative filenames."""
    sections = [
        f"--- FILE: {chunk.source_path} ---\n{chunk.content}" for chunk in chunks
    ]
    return "\n\n".join(sections)


def build_debugging_prompt(bug_report: str, chunks: Sequence[CodeChunk]) -> str:
    """Build the complete one-shot prompt sent to the coding model."""
    context = format_repository_context(chunks)
    return f"""Use this general Python debugging pattern when tracing return values:

def transform(value):
    return value + 1

def caller(value):
    transform(value)
    return value

Here caller discards the returned value. The smallest self-contained fix is:

def caller(value):
    return transform(value)

Now independently diagnose the repository below using only the shown code. Name the file
containing the line that must change and show the complete rewritten function. If the relevant
code is not inside a function, show the smallest self-contained changed block. Do not repeat
the input.

BUG REPORT:
{bug_report.strip()}

SELECTED REPOSITORY CONTEXT:

{context}

Return only this concise, user-visible debugging summary. Base the evidence on the
shown files. Do not provide hidden chain-of-thought or describe private internal
reasoning. Use no Markdown heading markers and no extra sections:

Evidence considered:
- 2 to 4 short bullets citing observable behavior from the shown files

Conclusion:
A concise root-cause statement.

Relevant file:
The repository-relative path of the file to change.

Suggested change:
The complete rewritten function containing the fix, without Markdown fences. If the
relevant code is not a function, provide the smallest self-contained changed block.

Explanation:
A brief explanation of why the change fixes the observed behavior.
"""

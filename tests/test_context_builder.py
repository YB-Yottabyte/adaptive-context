from context_builder import DebuggingContextBuilder
from retrieval import CodeChunk

BUILDER = DebuggingContextBuilder()


def test_format_repository_context_includes_file_headers_and_code() -> None:
    chunks = [
        CodeChunk("payment.py", "def checkout():\n    pass", 1, 2),
        CodeChunk("utils/discount.py", "RATE = 0.2", 1, 1),
    ]

    context = BUILDER.format_repository_context(chunks)

    assert "--- FILE: payment.py ---\ndef checkout():" in context
    assert "--- FILE: utils/discount.py ---\nRATE = 0.2" in context


def test_build_debugging_prompt_requests_public_evidence_summary() -> None:
    chunk = CodeChunk("payment.py", "return price", 1, 1)

    prompt = BUILDER.build_prompt("Discount is ignored.", [chunk])

    assert "BUG REPORT:\nDiscount is ignored." in prompt
    assert "SELECTED REPOSITORY CONTEXT:" in prompt
    assert "--- FILE: payment.py ---\nreturn price" in prompt
    assert "Evidence considered:" in prompt
    assert "Conclusion:" in prompt
    assert "Relevant file:" in prompt
    assert "Suggested change:" in prompt
    assert "complete rewritten function containing the fix" in prompt
    assert "smallest self-contained changed block" in prompt
    assert "Explanation:" in prompt
    assert "Do not provide hidden chain-of-thought" in prompt

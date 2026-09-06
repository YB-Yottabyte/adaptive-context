from debugging_result import (
    DebuggingSummary,
    DebuggingSummaryParser,
    SourceChangeAnalyzer,
)
from retrieval import CodeChunk


def test_parser_uses_public_sections() -> None:
    summary = DebuggingSummaryParser().parse(
        "Evidence considered:\n• one\n• two\n\n"
        "Conclusion:\nA concise conclusion.\n\n"
        "Relevant file:\nmodule.py\n\n"
        "Suggested change:\nreturn fixed\n\n"
        "Explanation:\nThis uses observable evidence."
    )

    assert summary.evidence == ("one", "two")
    assert summary.conclusion == "A concise conclusion."
    assert summary.relevant_file == "module.py"
    assert summary.is_complete


def test_parser_falls_back_to_unstructured_response() -> None:
    response = "The checkout function ignores the helper's return value."

    summary = DebuggingSummaryParser().parse(response)

    assert summary.evidence == ()
    assert summary.conclusion == response
    assert summary.relevant_file == ""
    assert summary.suggested_change == ""
    assert summary.explanation == ""
    assert not summary.is_complete


def test_parser_can_disable_unstructured_fallback() -> None:
    summary = DebuggingSummaryParser().parse(
        "partial response",
        fallback_to_response=False,
    )

    assert summary.conclusion == ""


def test_parser_removes_markdown_fence_from_suggested_code() -> None:
    summary = DebuggingSummaryParser().parse(
        "Evidence considered:\n- caller ignores the result\n\n"
        "Conclusion:\nThe original value is returned.\n\n"
        "Relevant file:\npayment.py\n\n"
        "Suggested change:\n```python\n"
        "def checkout(price, discount):\n"
        "    return apply_discount(price, discount)\n"
        "```\n\n"
        "Explanation:\nReturn the computed value."
    )

    assert summary.suggested_change == (
        "def checkout(price, discount):\n"
        "    return apply_discount(price, discount)"
    )


def test_changed_line_numbers_identify_only_replaced_lines() -> None:
    current = (
        "def checkout(price, discount):\n"
        "    apply_discount(price, discount)\n"
        "    return price"
    )
    suggested = (
        "def checkout(price, discount):\n    return apply_discount(price, discount)"
    )

    current_changed, suggested_changed = SourceChangeAnalyzer().changed_line_numbers(
        current,
        suggested,
    )

    assert current_changed == {2, 3}
    assert suggested_changed == {2}


def test_analyzer_finds_definition_and_preserves_repository_line_number() -> None:
    summary = DebuggingSummary(
        evidence=(),
        conclusion="",
        relevant_file="src/payment.py",
        suggested_change="def checkout():\n    return True",
        explanation="",
    )
    chunk = CodeChunk(
        "src/payment.py",
        "VALUE = 1\n\ndef checkout():\n    return False",
        20,
        23,
    )

    definition = SourceChangeAnalyzer().find_current_definition(summary, [chunk])

    assert definition is not None
    assert definition.source_path == "src/payment.py"
    assert definition.code == "def checkout():\n    return False"
    assert definition.start_line == 22

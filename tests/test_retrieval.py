from pathlib import Path

import pytest

from retrieval import (
    CodeChunk,
    chunk_file,
    create_repository_chunks,
    find_python_files,
    rank_chunks,
    select_top_k,
)


def test_find_python_files_is_recursive_sorted_and_ignores_hidden_dirs(
    tmp_path: Path,
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / ".venv").mkdir()
    (tmp_path / "z.py").write_text("z = 1\n", encoding="utf-8")
    (tmp_path / "nested" / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not Python\n", encoding="utf-8")
    (tmp_path / ".venv" / "ignored.py").write_text("ignored = 1\n", encoding="utf-8")

    files = find_python_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in files] == [
        "nested/a.py",
        "z.py",
    ]


def test_chunk_file_creates_overlapping_chunks_with_metadata(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")

    chunks = chunk_file(source, tmp_path, chunk_lines=3, overlap_lines=1)

    assert chunks == [
        CodeChunk("module.py", "one\ntwo\nthree", 1, 3),
        CodeChunk("module.py", "three\nfour\nfive", 3, 5),
    ]


@pytest.mark.parametrize(
    ("chunk_lines", "overlap_lines", "message"),
    [
        (0, 0, "chunk_lines must be at least 1"),
        (3, -1, "overlap_lines must be between 0 and chunk_lines - 1"),
        (3, 3, "overlap_lines must be between 0 and chunk_lines - 1"),
    ],
)
def test_chunk_file_rejects_invalid_chunk_configuration(
    tmp_path: Path,
    chunk_lines: int,
    overlap_lines: int,
    message: str,
) -> None:
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        chunk_file(
            source,
            tmp_path,
            chunk_lines=chunk_lines,
            overlap_lines=overlap_lines,
        )


def test_chunk_file_ignores_empty_python_file(tmp_path: Path) -> None:
    source = tmp_path / "empty.py"
    source.write_text("", encoding="utf-8")

    assert chunk_file(source, tmp_path) == []


def test_create_repository_chunks_handles_small_repository(tmp_path: Path) -> None:
    (tmp_path / "small.py").write_text("value = 1\n", encoding="utf-8")

    chunks = create_repository_chunks(tmp_path)

    assert chunks == [CodeChunk("small.py", "value = 1", 1, 1)]


def test_tfidf_retrieval_ranks_relevant_chunk_first() -> None:
    chunks = [
        CodeChunk("logging.py", "def write_log(message): pass", 1, 1),
        CodeChunk("payment.py", "def checkout(price, discount): return price", 1, 1),
        CodeChunk("email.py", "def send_email(address): pass", 1, 1),
    ]

    ranked = rank_chunks("checkout discount returns the original price", chunks)

    assert ranked[0].chunk.source_path == "payment.py"
    assert ranked[0].score > ranked[1].score


def test_retrieval_matches_snake_case_identifiers_to_words() -> None:
    chunks = [
        CodeChunk("discount.py", "def apply_discount(price): pass", 1, 1),
        CodeChunk("email.py", "def send_message(address): pass", 1, 1),
    ]

    ranked = rank_chunks("apply discount", chunks)

    assert ranked[0].chunk.source_path == "discount.py"
    assert ranked[0].score > 0


def test_ranking_is_deterministic_when_scores_tie() -> None:
    chunks = [
        CodeChunk("z.py", "unrelated", 1, 1),
        CodeChunk("a.py", "unrelated", 1, 1),
    ]

    first_run = rank_chunks("missing term", chunks)
    second_run = rank_chunks("missing term", list(reversed(chunks)))

    assert [item.chunk.source_path for item in first_run] == ["a.py", "z.py"]
    assert first_run == second_run


def test_select_top_k_limits_results() -> None:
    chunks = [CodeChunk(f"{index}.py", f"value {index}", 1, 1) for index in range(4)]

    selected = select_top_k("value", chunks, top_k=2)

    assert len(selected) == 2


def test_select_top_k_rejects_non_positive_value() -> None:
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        select_top_k("query", [], top_k=0)


def test_select_top_k_returns_all_available_chunks_when_k_is_larger() -> None:
    chunks = [
        CodeChunk("b.py", "beta", 1, 1),
        CodeChunk("a.py", "alpha", 1, 1),
    ]

    selected = select_top_k("alpha", chunks, top_k=10)

    assert [item.chunk.source_path for item in selected] == ["a.py", "b.py"]

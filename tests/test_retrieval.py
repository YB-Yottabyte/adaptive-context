from pathlib import Path

import pytest

from retrieval import (
    CodeChunk,
    RepositoryChunker,
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

    files = RepositoryChunker().find_python_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in files] == [
        "nested/a.py",
        "z.py",
    ]


def test_find_python_files_excludes_generated_and_cache_content(
    tmp_path: Path,
) -> None:
    (tmp_path / "build").mkdir()
    (tmp_path / ".chroma").mkdir()
    (tmp_path / "application.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / ".private.py").write_text("secret = True\n", encoding="utf-8")
    (tmp_path / "service_pb2.py").write_text("generated = True\n", encoding="utf-8")
    (tmp_path / "build" / "artifact.py").write_text("built = True\n", encoding="utf-8")
    (tmp_path / ".chroma" / "index.py").write_text("index = True\n", encoding="utf-8")

    files = RepositoryChunker().find_python_files(tmp_path)

    assert [path.name for path in files] == ["application.py"]


def test_chunk_file_creates_overlapping_chunks_with_metadata(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")

    chunks = RepositoryChunker(chunk_lines=3, overlap_lines=1).chunk_file(
        source,
        tmp_path,
    )

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
        RepositoryChunker(
            chunk_lines=chunk_lines,
            overlap_lines=overlap_lines,
        )


def test_chunk_file_ignores_empty_python_file(tmp_path: Path) -> None:
    source = tmp_path / "empty.py"
    source.write_text("", encoding="utf-8")

    assert RepositoryChunker().chunk_file(source, tmp_path) == []


def test_create_repository_chunks_handles_small_repository(tmp_path: Path) -> None:
    (tmp_path / "small.py").write_text("value = 1\n", encoding="utf-8")

    chunks = RepositoryChunker().create_chunks(tmp_path)

    assert chunks == [CodeChunk("small.py", "value = 1", 1, 1)]

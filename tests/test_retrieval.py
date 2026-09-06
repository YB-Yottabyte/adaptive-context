"""Repository discovery, chunking, and fixed Top-K retrieval coverage."""

from pathlib import Path

import chromadb

from retrieval import CodeChunk, RepositoryChunker
from semantic_retrieval import ChromaRetriever


class KeywordEmbedding:
    """Small deterministic embedding used instead of downloading a model."""

    vocabulary = ("checkout", "discount", "permission")

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, *(float(text.lower().count(term)) for term in self.vocabulary)]
            for text in texts
        ]


def test_python_discovery_excludes_ignored_directories(tmp_path: Path) -> None:
    (tmp_path / "package").mkdir()
    for ignored in (".git", ".venv", "__pycache__", "build", ".chroma"):
        directory = tmp_path / ignored
        directory.mkdir()
        (directory / "ignored.py").write_text("ignored = True\n", encoding="utf-8")
    (tmp_path / "package" / "service.py").write_text(
        "def service(): pass\n",
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text("def main(): pass\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not Python\n", encoding="utf-8")

    files = RepositoryChunker().find_python_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in files] == [
        "main.py",
        "package/service.py",
    ]


def test_chunks_include_relative_paths_and_line_ranges(tmp_path: Path) -> None:
    (tmp_path / "package").mkdir()
    source = tmp_path / "package" / "service.py"
    source.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")

    index = RepositoryChunker(chunk_lines=3, overlap_lines=1).create_index(tmp_path)

    assert list(index.chunks) == [
        CodeChunk("package/service.py", "one\ntwo\nthree", 1, 3),
        CodeChunk("package/service.py", "three\nfour\nfive", 3, 5),
    ]


def test_semantic_retrieval_returns_fixed_top_k(tmp_path: Path) -> None:
    retriever = ChromaRetriever(
        client=chromadb.EphemeralClient(),
        embedding_function=KeywordEmbedding(),
    )
    chunks = [
        CodeChunk("notifications.py", "def send_email(): pass", 1, 1),
        CodeChunk(
            "payment.py",
            "def checkout(price, discount): return price",
            1,
            1,
        ),
        CodeChunk("permissions.py", "def has_permission(): return False", 1, 1),
    ]

    try:
        results = retriever.retrieve(
            "checkout ignores the discount",
            chunks,
            2,
            tmp_path,
        )
    finally:
        retriever.close()

    assert len(results) == 2
    assert results[0].chunk.source_path == "payment.py"
    assert results[0].score > results[1].score

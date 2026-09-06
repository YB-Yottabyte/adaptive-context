from pathlib import Path
from unittest.mock import Mock

import chromadb
import pytest

from retrieval import CodeChunk
from semantic_retrieval import ChromaRetriever


class KeywordEmbedding:
    """Small deterministic embedding used instead of downloading a model."""

    vocabulary = ("checkout", "discount", "email", "permission", "configuration")

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, *(float(text.lower().count(term)) for term in self.vocabulary)]
            for text in texts
        ]


@pytest.fixture
def chroma_client():
    return chromadb.EphemeralClient()


@pytest.fixture
def retriever(chroma_client) -> ChromaRetriever:
    return ChromaRetriever(
        client=chroma_client,
        embedding_function=KeywordEmbedding(),
    )


def test_repository_chunks_are_indexed_with_source_metadata(
    tmp_path: Path,
    chroma_client,
    retriever: ChromaRetriever,
) -> None:
    chunks = [
        CodeChunk("payment.py", "def checkout(): return total", 1, 1),
        CodeChunk("tests/test_payment.py", "def test_discount(): pass", 4, 4),
    ]

    retriever.retrieve("checkout discount", chunks, 2, tmp_path)

    collection = chroma_client.get_collection(
        retriever.collection_name(tmp_path),
        embedding_function=None,
    )
    indexed = collection.get(include=["documents", "metadatas"])
    assert collection.count() == 2
    assert set(indexed["documents"]) == {chunk.content for chunk in chunks}
    metadata_by_path = {
        metadata["source_path"]: metadata for metadata in indexed["metadatas"]
    }
    assert metadata_by_path["payment.py"] == {
        "chunk_id": metadata_by_path["payment.py"]["chunk_id"],
        "chunk_index": 0,
        "end_line": 1,
        "file_type": "py",
        "source_path": "payment.py",
        "start_line": 1,
    }
    assert len(metadata_by_path["payment.py"]["chunk_id"]) == 64
    assert metadata_by_path["tests/test_payment.py"]["chunk_index"] == 1


def test_persistent_client_writes_to_configured_local_directory(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    persist_directory = tmp_path / "chroma-index"
    retriever = ChromaRetriever(
        persist_directory=persist_directory,
        embedding_function=KeywordEmbedding(),
    )

    results = retriever.retrieve(
        "checkout",
        [CodeChunk("payment.py", "def checkout(): return total", 1, 1)],
        1,
        repository,
    )

    assert persist_directory.is_dir()
    assert [result.chunk.source_path for result in results] == ["payment.py"]


def test_close_releases_client_once_and_clears_reference() -> None:
    client = Mock()
    retriever = ChromaRetriever(
        client=client,
        embedding_function=KeywordEmbedding(),
    )

    retriever.close()
    retriever.close()

    client.close.assert_called_once_with()
    assert retriever._client is None


def test_semantic_query_returns_relevant_file_and_fixed_top_k(
    tmp_path: Path,
    retriever: ChromaRetriever,
) -> None:
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

    results = retriever.retrieve(
        "checkout ignores the discount",
        chunks,
        2,
        tmp_path,
    )

    assert len(results) == 2
    assert results[0].chunk.source_path == "payment.py"
    assert results[0].score > results[1].score


def test_empty_and_very_small_repositories_are_handled(
    tmp_path: Path,
    retriever: ChromaRetriever,
) -> None:
    assert retriever.retrieve("anything", [], 3, tmp_path) == []

    one_result = retriever.retrieve(
        "configuration",
        [CodeChunk("settings.py", "configuration = {}", 1, 1)],
        5,
        tmp_path,
    )

    assert len(one_result) == 1
    assert one_result[0].chunk.source_path == "settings.py"


def test_repeated_runs_replace_stale_collection_content(
    tmp_path: Path,
    chroma_client,
    retriever: ChromaRetriever,
) -> None:
    retriever.retrieve(
        "checkout",
        [
            CodeChunk("old.py", "def checkout(): pass", 1, 1),
            CodeChunk("removed.py", "old content", 1, 1),
        ],
        2,
        tmp_path,
    )

    second_results = retriever.retrieve(
        "checkout",
        [CodeChunk("current.py", "def checkout(): return total", 1, 1)],
        2,
        tmp_path,
    )

    collection = chroma_client.get_collection(
        retriever.collection_name(tmp_path),
        embedding_function=None,
    )
    assert collection.count() == 1
    assert (
        collection.get(include=["metadatas"])["metadatas"][0]["source_path"]
        == "current.py"
    )
    assert [result.chunk.source_path for result in second_results] == ["current.py"]


def test_collections_are_isolated_by_repository_path(
    tmp_path: Path,
    chroma_client,
) -> None:
    first_repository = tmp_path / "first"
    second_repository = tmp_path / "second"
    first_repository.mkdir()
    second_repository.mkdir()
    retriever = ChromaRetriever(
        client=chroma_client,
        embedding_function=KeywordEmbedding(),
    )

    retriever.retrieve(
        "checkout",
        [CodeChunk("payment.py", "checkout", 1, 1)],
        1,
        first_repository,
    )
    retriever.retrieve(
        "permission",
        [CodeChunk("auth.py", "permission", 1, 1)],
        1,
        second_repository,
    )

    names = {collection.name for collection in chroma_client.list_collections()}
    assert retriever.collection_name(first_repository) in names
    assert retriever.collection_name(second_repository) in names
    assert retriever.collection_name(first_repository) != (
        retriever.collection_name(second_repository)
    )


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(
    tmp_path: Path,
    retriever: ChromaRetriever,
    top_k: int,
) -> None:
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.retrieve("query", [], top_k, tmp_path)


def test_invalid_repository_paths_are_rejected(
    tmp_path: Path,
    retriever: ChromaRetriever,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="Repository directory does not exist"):
        retriever.retrieve("query", [], 1, missing)

    regular_file = tmp_path / "repository.py"
    regular_file.write_text("value = 1\n", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="Repository path is not a directory"):
        retriever.retrieve("query", [], 1, regular_file)


def test_blank_query_is_rejected(
    tmp_path: Path,
    retriever: ChromaRetriever,
) -> None:
    with pytest.raises(ValueError, match="retrieval query must not be empty"):
        retriever.retrieve(" \n", [], 1, tmp_path)

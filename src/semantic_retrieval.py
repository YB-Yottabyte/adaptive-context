"""ChromaDB-backed one-pass semantic repository retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from retrieval import CodeChunk, RetrievedChunk

DEFAULT_CHROMA_DIRECTORY = Path(__file__).resolve().parent.parent / ".chroma"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_PREFIX = "context-debug"


class ChromaRetriever:
    """Index and query one repository through a local Chroma collection."""

    name = "chroma"
    display_name = "ChromaDB semantic search"

    def __init__(
        self,
        *,
        persist_directory: Path = DEFAULT_CHROMA_DIRECTORY,
        client: Any | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        self.persist_directory = persist_directory
        self._client = client
        self._embedding_function = (
            embedding_function
            if embedding_function is not None
            else DefaultEmbeddingFunction()
        )

    def _get_client(self) -> Any:
        if self._client is None:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        return self._client

    def _fresh_collection(self, repository_dir: Path) -> Any:
        """Create or empty this repository's isolated persistent collection."""
        collection = self._get_client().get_or_create_collection(
            name=self.collection_name(repository_dir),
            embedding_function=None,
            metadata={
                "repository_path": repository_dir.resolve().as_posix(),
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
            },
            configuration={"hnsw": {"space": "cosine"}},
        )
        existing_ids = collection.get(include=[])["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)
        return collection

    def retrieve(
        self,
        query: str,
        chunks: Sequence[CodeChunk],
        top_k: int,
        repository_dir: Path,
    ) -> list[RetrievedChunk]:
        """Reindex one repository and perform exactly one semantic query."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        repository_dir = repository_dir.resolve()
        if not repository_dir.exists():
            raise FileNotFoundError(
                f"Test case directory does not exist: {repository_dir}"
            )
        if not repository_dir.is_dir():
            raise NotADirectoryError(
                f"Test case path is not a directory: {repository_dir}"
            )
        if not query.strip():
            raise ValueError("retrieval query must not be empty")
        collection = self._fresh_collection(repository_dir)
        if not chunks:
            return []

        indexed_documents = [self.embedding_document(chunk) for chunk in chunks]
        document_embeddings = self._embedding_function(indexed_documents)
        collection.add(
            ids=[
                self.chunk_identifier(chunk, index)
                for index, chunk in enumerate(chunks)
            ],
            embeddings=document_embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "source_path": chunk.source_path,
                    "chunk_index": index,
                    "chunk_id": self.chunk_identifier(chunk, index),
                    "file_type": Path(chunk.source_path).suffix.lstrip(".")
                    or "unknown",
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                }
                for index, chunk in enumerate(chunks)
            ],
        )

        query_embedding = self._embedding_function([query])[0]
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(chunks)),
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []

        retrieved: list[RetrievedChunk] = []
        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            if document is None or metadata is None:
                continue
            retrieved.append(
                RetrievedChunk(
                    chunk=CodeChunk(
                        source_path=str(metadata["source_path"]),
                        content=document,
                        start_line=int(metadata["start_line"]),
                        end_line=int(metadata["end_line"]),
                    ),
                    score=1.0 - float(distance),
                )
            )
        return retrieved

    @staticmethod
    def collection_name(repository_dir: Path) -> str:
        """Return a stable, Chroma-safe name for one repository path."""
        normalized_path = repository_dir.resolve().as_posix()
        path_digest = sha256(normalized_path.encode("utf-8")).hexdigest()[:16]
        return f"{COLLECTION_PREFIX}-{path_digest}"

    @staticmethod
    def chunk_identifier(chunk: CodeChunk, chunk_index: int) -> str:
        """Create a deterministic identifier for a repository chunk."""
        identity = (
            f"{chunk.source_path}:{chunk.start_line}:{chunk.end_line}:{chunk_index}"
        )
        return sha256(identity.encode("utf-8")).hexdigest()

    @staticmethod
    def embedding_document(chunk: CodeChunk) -> str:
        """Include source metadata in text embedded for semantic search."""
        return (
            f"Repository file: {chunk.source_path}\n"
            f"Lines: {chunk.start_line}-{chunk.end_line}\n"
            f"{chunk.content}"
        )

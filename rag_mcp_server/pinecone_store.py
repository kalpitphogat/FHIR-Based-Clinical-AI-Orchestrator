from __future__ import annotations

import logging
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from .config import (
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_EMBED_MODEL,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
)

logger = logging.getLogger(__name__)


class PineconeVectorStore:
    """Encapsulates embedding, upsert, and semantic query operations."""

    _EMBED_DIMENSION = 1024

    def __init__(self) -> None:
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not configured.")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self._ensure_index()
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

    def _ensure_index(self) -> None:
        names = self.pc.list_indexes().names()
        if PINECONE_INDEX_NAME in names:
            return
        logger.info("Creating Pinecone index name=%s", PINECONE_INDEX_NAME)
        self.pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=self._EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        response = self.pc.inference.embed(
            model=PINECONE_EMBED_MODEL,
            inputs=texts,
            parameters={"input_type": input_type, "truncate": "END"},
        )
        vectors: list[list[float]] = []
        for item in response.data:
            vector = getattr(item, "values", None) or item.get("values", [])
            vectors.append(vector)
        return vectors

    def upsert_documents(self, docs: list[dict[str, Any]], namespace: str) -> int:
        if not docs:
            return 0
        values = self._embed([doc["text"] for doc in docs], input_type="passage")
        vectors = []
        for doc, vector in zip(docs, values):
            vectors.append(
                {
                    "id": doc["id"],
                    "values": vector,
                    "metadata": {
                        "text": doc["text"],
                        **doc.get("metadata", {}),
                    },
                }
            )
        self.index.upsert(vectors=vectors, namespace=namespace)
        return len(vectors)

    def query(self, query_text: str, top_k: int, namespace: str) -> list[dict[str, Any]]:
        vector = self._embed([query_text], input_type="query")[0]
        response = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
        )
        matches = []
        for match in response.matches:
            md = getattr(match, "metadata", None) or {}
            score = getattr(match, "score", 0.0)
            matches.append(
                {
                    "id": getattr(match, "id", ""),
                    "score": score,
                    "text": md.get("text", ""),
                    "metadata": md,
                }
            )
        return matches


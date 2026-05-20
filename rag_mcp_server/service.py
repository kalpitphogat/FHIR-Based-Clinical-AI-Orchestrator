from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from disease_identification_agent.tools import _query_nlm

from .config import PINECONE_NAMESPACE, PINECONE_TOP_K_DEFAULT, RAG_MATCH_THRESHOLD
from .pinecone_store import PineconeVectorStore

logger = logging.getLogger(__name__)


class RagContextService:
    """Coordinates Pinecone semantic retrieval and NLM fallback enrichment."""

    def __init__(self) -> None:
        self.store = PineconeVectorStore()

    @staticmethod
    def _doc_id(seed: str) -> str:
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _chunk_fhir_payload(fhir_payload: dict[str, Any]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for section, value in fhir_payload.items():
            text = f"{section}: {json.dumps(value, ensure_ascii=True)}"
            chunks.append(
                {
                    "id": RagContextService._doc_id(f"fhir::{section}::{text[:300]}"),
                    "text": text,
                    "metadata": {"source": "fhir", "section": section},
                }
            )
        return chunks

    def _cache_nlm_disease(self, disease: str) -> list[dict[str, Any]]:
        results = _query_nlm(disease)
        docs: list[dict[str, Any]] = []
        for idx, item in enumerate(results):
            text = (
                f"Disease topic: {item.get('title', '')}. "
                f"Alt titles: {', '.join(item.get('alt_titles', []))}. "
                f"Reference URL: {item.get('url', '')}"
            )
            docs.append(
                {
                    "id": self._doc_id(f"nlm::{disease}::{idx}::{text}"),
                    "text": text,
                    "metadata": {
                        "source": "nlm",
                        "disease_query": disease,
                        "url": item.get("url", ""),
                    },
                }
            )
        if docs:
            self.store.upsert_documents(docs, namespace=PINECONE_NAMESPACE)
        return docs

    def enrich_discharge_context(
        self,
        patient_id: str,
        user_query: str,
        fhir_payload: dict[str, Any],
        disease_terms: list[str] | None = None,
        top_k: int = PINECONE_TOP_K_DEFAULT,
    ) -> dict[str, Any]:
        disease_terms = [d.strip() for d in (disease_terms or []) if d.strip()]
        fhir_docs = self._chunk_fhir_payload(fhir_payload)
        fhir_namespace = f"patient-{patient_id}" if patient_id else PINECONE_NAMESPACE
        stored_count = self.store.upsert_documents(fhir_docs, namespace=fhir_namespace)
        semantic_hits = self.store.query(query_text=user_query, top_k=top_k, namespace=fhir_namespace)

        fetched_from_nlm: list[dict[str, Any]] = []
        cached_hits: list[dict[str, Any]] = []
        for term in disease_terms:
            hits = self.store.query(query_text=term, top_k=3, namespace=PINECONE_NAMESPACE)
            strong_hits = [h for h in hits if h["score"] >= RAG_MATCH_THRESHOLD]
            if strong_hits:
                cached_hits.extend(strong_hits)
                continue
            fetched_docs = self._cache_nlm_disease(term)
            fetched_from_nlm.extend(fetched_docs)

        return {
            "status": "success",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "patient_namespace": fhir_namespace,
            "stored_fhir_chunks": stored_count,
            "semantic_hits": semantic_hits,
            "cached_disease_hits": cached_hits,
            "fetched_disease_docs": fetched_from_nlm,
        }


_RAG_SERVICE: RagContextService | None = None


def get_rag_service() -> RagContextService:
    global _RAG_SERVICE
    if _RAG_SERVICE is None:
        _RAG_SERVICE = RagContextService()
    return _RAG_SERVICE


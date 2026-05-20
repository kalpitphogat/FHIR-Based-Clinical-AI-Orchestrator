from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .config import PINECONE_NAMESPACE, PINECONE_TOP_K_DEFAULT
from .service import get_rag_service

mcp = FastMCP("discharge-rag-mcp")


@mcp.tool()
def enrich_discharge_context(
    patient_id: str,
    user_query: str,
    fhir_payload: dict,
    disease_terms: list[str] | None = None,
    top_k: int = PINECONE_TOP_K_DEFAULT,
) -> dict:
    """Store FHIR context in Pinecone, run semantic search, and use NLM fallback."""
    return get_rag_service().enrich_discharge_context(
        patient_id=patient_id,
        user_query=user_query,
        fhir_payload=fhir_payload,
        disease_terms=disease_terms,
        top_k=top_k,
    )


@mcp.tool()
def semantic_search(query: str, namespace: str = PINECONE_NAMESPACE, top_k: int = 5) -> dict:
    """Perform semantic retrieval directly against a namespace."""
    hits = get_rag_service().store.query(query_text=query, top_k=top_k, namespace=namespace)
    return {"status": "success", "hits": hits, "namespace": namespace}


if __name__ == "__main__":
    mcp.run()


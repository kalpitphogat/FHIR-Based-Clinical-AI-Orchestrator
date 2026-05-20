"""Discharge tools that enrich context via Pinecone-backed RAG."""

from __future__ import annotations

from google.adk.tools import ToolContext

from rag_mcp_server.service import get_rag_service
from shared.tools.fhir import (
    get_active_conditions,
    get_active_medications,
    get_patient_demographics,
    get_recent_observations,
)


def get_discharge_rag_context(
    user_query: str,
    disease_terms: str,
    tool_context: ToolContext,
) -> dict:
    """
    Build a discharge-focused context bundle using FHIR + Pinecone semantic search.

    Args:
        user_query: The current clinical question.
        disease_terms: Comma-separated disease/symptom terms for fallback fetch.
    """
    patient_id = tool_context.state.get("patient_id", "")
    fhir_payload = {
        "demographics": get_patient_demographics(tool_context),
        "active_conditions": get_active_conditions(tool_context),
        "active_medications": get_active_medications(tool_context),
        "recent_vitals": get_recent_observations("vital-signs", tool_context),
    }

    terms = [term.strip() for term in (disease_terms or "").split(",") if term.strip()]
    return get_rag_service().enrich_discharge_context(
        patient_id=patient_id,
        user_query=user_query,
        fhir_payload=fhir_payload,
        disease_terms=terms,
    )


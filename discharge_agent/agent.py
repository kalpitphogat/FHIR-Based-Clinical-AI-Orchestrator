from google.adk.agents import Agent
from shared.tools import (
    get_patient_demographics,
    get_active_medications,
    get_active_conditions,
    get_recent_observations,
)
from .tools import get_discharge_rag_context

from shared.fhir_hook import extract_fhir_context
from shared.debug_trail import log_before_model, log_after_model, log_tool_call, log_tool_result



discharge_agent = Agent(
    name="discharge_agent",
    model="gemini-2.5-flash",
    description="Evaluates if patient is fit for discharge",
    instruction="""
    Assess if the patient is ready for discharge.

    You MUST:
    - Call get_discharge_rag_context first to retrieve semantic and fallback context from the RAG MCP layer.
    - Use get_recent_observations("vital-signs") to check vitals
    - Use get_active_conditions to check ongoing issues
    - Use get_active_medications if needed

    Only make decisions based on actual data retrieved from tools.

    structure of reason:
    - If vitals are unstable, mention which ones and why it matters.
    - If there are active conditions that require inpatient care, mention them and why.
    - If medications indicate need for monitoring, mention them and why.
    - If something is missing that is critical for decision, say that clearly.

    Structure of Summary:
    - provide bullet point of things that doctor needs to address before discharge (e.g. stabilize BP, control blood sugar, etc.)
    - If patient seems ready for discharge, say that clearly but also mention any follow-up needed.

    Return:
    - Decision
    - Reasons
    - Risks
    - Summary
    """,
    tools=[
        get_discharge_rag_context,
        get_patient_demographics,
        get_active_medications,
        get_active_conditions,
        get_recent_observations,
    ],
    before_model_callback=extract_fhir_context,
    after_model_callback=log_after_model,
    before_tool_callback=log_tool_call,
    after_tool_callback=log_tool_result
)
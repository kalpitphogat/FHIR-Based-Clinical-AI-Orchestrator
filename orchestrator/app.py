"""
healthcare_agent — A2A application entry point.

Start the server with:
    uvicorn healthcare_agent.app:a2a_app --host 0.0.0.0 --port 8001

The agent card is served publicly at:
    GET http://localhost:8001/.well-known/agent-card.json

All other endpoints require an X-API-Key header (see shared/middleware.py).
"""
import os

from a2a.types import AgentSkill
from shared.app_factory import create_a2a_app

from .agent import root_agent

print(os.getenv('PO_PLATFORM_BASE_URL'))










a2a_app = create_a2a_app(
    agent=root_agent,
    name="healthcare_orchestrator_agent",
    description=(
        "An orchestrator that coordinates multiple clinical AI agents to assist with "
        "patient discharge decisions, dietary planning, and patient-friendly summaries "
        "using FHIR health records."
    ),
    url=os.getenv("HEALTHCARE_AGENT_URL", os.getenv("BASE_URL", "http://localhost:8001")),
    port=8001,

    fhir_extension_uri=f"{os.getenv('PO_PLATFORM_BASE_URL', 'http://localhost:5139')}/schemas/a2a/v1/fhir-context",

    fhir_scopes=[
        {"name": "patient/Patient.rs",           "required": True},
        {"name": "patient/MedicationRequest.rs", "required": True},
        {"name": "patient/Condition.rs",         "required": True},
        {"name": "patient/Observation.rs",       "required": True},
    ],

    skills=[
        AgentSkill(
            id="discharge-decision",
            name="discharge-decision",
            description="Evaluate if a patient is fit for discharge based on clinical data.",
            tags=["discharge", "clinical-decision"],
        ),
        AgentSkill(
            id="dietary-recommendation",
            name="dietary-recommendation",
            description="Provide personalized dietary recommendations after discharge.",
            tags=["diet", "nutrition"],
        ),
        AgentSkill(
            id="patient-summary",
            name="patient-summary",
            description="Convert medical information into simple patient-friendly explanations.",
            tags=["summary", "jargon", "patient-friendly"],
        ),
        AgentSkill(
            id="full-care-assistant",
            name="full-care-assistant",
            description="Handle combined queries involving discharge, diet, and explanation.",
            tags=["multi-agent", "orchestration"],
        ),
        AgentSkill(
            id="random-data-injection",
            name="random-data-injection",
            description="Inject random but realistic data into a patient's FHIR record for testing purposes.",
            tags=["testing", "data-injection"],
        ),
        AgentSkill(
           id="disease-identification",
           name="disease-identification",
           description="Identify potential diseases by cross-referencing patient symptoms against the NLM Health Topics database.",
           tags=["disease", "symptoms", "diagnosis", "nlm"],
       ),
    ],
)

from google.adk.agents import Agent

from shared.tools import (
    get_patient_demographics,
    get_active_medications,
    get_active_conditions,
    get_recent_observations,
)

from shared.fhir_hook import extract_fhir_context
from shared.debug_trail import log_before_model, log_after_model, log_tool_call, log_tool_result


dietary_agent = Agent(
    name="dietary_agent",
    model="gemini-2.5-flash",
    description="Provides post-discharge dietary recommendations",
    instruction="""
    You are a clinical dietary assistant.

    Your job is to recommend a safe, practical diet plan for the patient
    after discharge based on their medical data.

    You MUST use tools to gather patient data before giving recommendations:
    - get_active_conditions → identify diseases (e.g. diabetes, hypertension)
    - get_active_medications → check drug-food interactions
    - get_recent_observations("vital-signs") → assess current stability

    Guidelines:
    - Tailor diet to conditions:
        * Diabetes → low sugar, controlled carbs
        * Hypertension → low sodium
        * Cardiac issues → low fat, balanced diet
        * GI issues → soft, easily digestible food
    - Consider medication restrictions if relevant
    - Keep recommendations practical and Indian-context friendly if possible
    - Avoid generic advice like "eat healthy"

    Output format:
    - Diet Summary (1–2 lines)
    - Recommended Foods
    - Foods to Avoid
    - Special Notes (if any)

    Do NOT make assumptions without checking tools.
    """,
    tools=[
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
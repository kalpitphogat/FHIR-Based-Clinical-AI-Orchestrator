from google.adk.agents import Agent

from shared.tools import (
    get_patient_demographics,
    get_active_medications,
    get_active_conditions,
    get_recent_observations,

)

from shared.debug_trail import log_before_model, log_after_model, log_tool_call, log_tool_result

from shared.fhir_hook import extract_fhir_context


patient_data_viewer_agent = Agent(
    name="patient_data_viewer_agent",
    model="gemini-2.5-flash",
    description="Views the patient's FHIR record",

    instruction="""
    Your job is to view the patient's FHIR record and return relevant information to the user based on their request. You have access to the following tools:   
    - get_patient_demographics: Retrieves the patient's demographics information.
    - get_active_medications: Retrieves the patient's active medications.
    - get_active_conditions: Retrieves the patient's active conditions.
    - get_recent_observations: Retrieves the patient's recent observations.
    Use these tools to gather information about the patient and answer the user's questions. 

    Structure of your response:
    - Start with a brief summary of the patient's demographics
    - List active conditions, medications, and recent observations in a clear format.
    - If the user asks for specific information, provide that information clearly and concisely.

    """,

    tools=[
        get_patient_demographics,
        get_active_conditions,
        get_active_medications,
        get_recent_observations,
    ],

    before_model_callback=extract_fhir_context,
    after_model_callback=log_after_model,
    before_tool_callback=log_tool_call,
    after_tool_callback=log_tool_result,
)
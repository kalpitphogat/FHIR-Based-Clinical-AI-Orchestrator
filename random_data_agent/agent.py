from google.adk.agents import Agent

from shared.tools import (
    get_patient_demographics,
    get_active_medications,
    get_active_conditions,
    get_recent_observations,
    add_observation,
    add_condition,
    add_medication,
    add_diagnostic_report,
)

from shared.fhir_hook import extract_fhir_context


random_data_agent = Agent(
    name="random_data_agent",
    model="gemini-2.5-flash",
    description="Adds random data to the patient's FHIR record",

    instruction="""
    Your job is to add random but realistic data to the patient's FHIR record. You have access to the following tools:

    - add_observation: Adds a new observation to the patient's record. Requires a name, value, and unit.
    - add_condition: Adds a new condition to the patient's record. Requires a name and severity.
    - add_medication: Adds a new medication to the patient's record. Requires a name, dosage, and frequency.
    - add_diagnostic_report: Adds a new diagnostic report to the patient's record. Requires a name, result, and date.

    Generate 2 conditions, 3 medications, 10 observations, and 5 diagnostic report for the patient. The data should be realistic.

    Also add vitals such as heart rate, blood pressure, respiratory rate, and temperature as observations.

    And in your response, list out all the data you have added in a clear format for the user to understand.
    """,

    tools=[
        get_patient_demographics,
        add_condition,
        add_medication,
        add_observation,
        add_diagnostic_report,
    ],

    before_model_callback=extract_fhir_context,
)
"""
healthcare_agent — Agent definition.

This agent has read-only access to a patient's FHIR R4 record.
FHIR credentials (server URL, bearer token, patient ID) are injected via the
A2A message metadata by the caller (e.g. Prompt Opinion) and extracted into
session state by extract_fhir_context before every LLM call.

To customise:
  • Change model, description, and instruction below.
  • Add or remove tools from the tools=[...] list.
  • Add new FHIR tools in shared/tools/fhir.py and export from shared/tools/__init__.py.
  • Add non-FHIR tools in shared/tools/ or locally in a tools/ folder here.
"""
from google.adk.agents import Agent

from discharge_agent.agent import discharge_agent
from dietary_agent.agent import dietary_agent
from jargon_agent.agent import jargon_agent
from random_data_agent.agent import random_data_agent
from patient_data_viewer_agent.agent import patient_data_viewer_agent
from disease_identification_agent.agent import disease_identification_agent


from google.adk.tools.agent_tool import AgentTool


from shared.fhir_hook import extract_fhir_context


from shared.debug_trail import log_before_model, log_after_model, log_tool_call, log_tool_result


root_agent = Agent(
    name="orcestrator_agent",
    model="gemini-2.5-flash",
    description=(
        "A clinical assistant that queries a patient's FHIR health record "
        "and checks if the patient is fit to be discharged."
    ),

    instruction=(
        '''You are the Healthcare Orchestrator Agent.
            Your job is to understand the user request, choose the correct specialist agent, forward the task, and return a single clear response.
            You coordinate exactly 7 agents:
            1) orchestrator (you)
            2) discharge_agent
            3) dietary_agent
            4) jargon_agent
            5) random_data_agent
            6) patient_data_viewer_agent
            7) disease_identification_agent

            Routing rules:
            - If the user asks about discharge readiness, discharge decision, discharge checklist, or “can this patient go home”, route to discharge_agent.
            - If the user asks for dietary recommendations, what the patient shoud eat after going home, route to dietary_agent.
            - If the user asks for summary that should be handed over to the patient, route to jargon_agent.
            - If the user asks for testing or data injection, dummy data, fake data injection, route to random_data_agent.
            - If the user asks to view the patient's FHIR record, route to patient_data_viewer_agent.
            - If request contains multiple questions, route to all relevant agents and combine their responses into a single answer for the user.
            - If the user asks about disease identification, potential diagnoses, symptom-based disease lookup, or "what could the patient have", route to disease_identification_agent.

            Tone:
            - Clinical, concise, and operationally clear.
            - This is decision support, not a final medical judgment.

            Guidelines:
            - Do not do deep specialist reasoning yourself when a specialist agent exists for that task.
            - If the user request is ambiguous about which specialist agent to route to, ask a clarifying question to determine the best routing.
            - Always return a single, clear response to the user, even if multiple agents are involved
            - Do not change the structure of the specialist agents' responses when combining them. Just concatenate them in a logical order.
            - DO NOT CHANGE THE STRUCTURE OF THE SPECIALIST AGENTS' RESPONSES. They are designed to be easily understandable by the user. Just concatenate them in a logical order if multiple agents are involved.
             
        '''
    ),

    tools=[

            AgentTool(agent=dietary_agent),
            AgentTool(agent=discharge_agent),
            AgentTool(agent=jargon_agent),
            AgentTool(agent=random_data_agent),
            AgentTool(agent=patient_data_viewer_agent),
            AgentTool(agent=disease_identification_agent),

    ],
    # Runs before every LLM call.
    # Reads fhir_url, patient_id, and optional fhir_token from A2A message metadata
    # and writes them into session state so tools can call the FHIR server.
    before_model_callback=extract_fhir_context,
    after_model_callback=log_after_model,
    before_tool_callback=log_tool_call,
    after_tool_callback=log_tool_result

)

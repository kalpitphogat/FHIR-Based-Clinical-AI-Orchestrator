import json
import logging
import os

from shared.logging_utils import safe_pretty_json, serialize_for_log, token_fingerprint

logger = logging.getLogger(__name__)

LOG_HOOK_RAW_OBJECTS = os.getenv("LOG_HOOK_RAW_OBJECTS", "false").lower() == "true"

FHIR_CONTEXT_KEY = "fhir-context"


def _first_non_empty(*values):
    for v in values:
        if v not in (None, ""):
            return v
    return None


def _safe_correlation_ids(callback_context, llm_request) -> dict:
    return {
        "task_id":    _first_non_empty(getattr(llm_request, "task_id", None),    getattr(callback_context, "task_id", None)),
        "context_id": _first_non_empty(getattr(llm_request, "context_id", None),  getattr(callback_context, "context_id", None)),
        "message_id": _first_non_empty(getattr(llm_request, "message_id", None),  getattr(callback_context, "message_id", None)),
    }


def _coerce_fhir_data(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _extract_metadata_sources(callback_context, llm_request) -> list:
    callback_metadata = getattr(callback_context, "metadata", None)

    run_config      = getattr(callback_context, "run_config", None)
    custom_metadata = getattr(run_config, "custom_metadata", None) if run_config else None
    a2a_metadata    = custom_metadata.get("a2a_metadata") if isinstance(custom_metadata, dict) else None

    llm_payload      = serialize_for_log(llm_request)
    contents         = llm_payload.get("contents", []) if isinstance(llm_payload, dict) else []
    content_metadata = None
    if contents and isinstance(contents, list):
        last = contents[-1]
        if isinstance(last, dict):
            content_metadata = last.get("metadata")

    return [
        ("callback_context.metadata",                                  callback_metadata),
        ("callback_context.run_config.custom_metadata.a2a_metadata",   a2a_metadata),
        ("llm_request.contents[-1].metadata",                          content_metadata),
    ]


def extract_fhir_from_payload(payload: dict):
    if not isinstance(payload, dict):
        return None, None
    params = payload.get("params")
    if not isinstance(params, dict):
        return None, None

    for metadata in (params.get("metadata"), (params.get("message") or {}).get("metadata")):
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if FHIR_CONTEXT_KEY in str(key):
                    return key, _coerce_fhir_data(value)

    return None, None


def extract_fhir_context(callback_context, llm_request):
    correlation      = _safe_correlation_ids(callback_context, llm_request)
    metadata_sources = _extract_metadata_sources(callback_context, llm_request)

    selected_source = "none"
    metadata        = {}
    for source_name, candidate in metadata_sources:
        if isinstance(candidate, dict) and candidate:
            metadata        = candidate
            selected_source = source_name
            break

    metadata_keys = list(metadata.keys())

    logger.info(
        "hook_called_enter task_id=%s context_id=%s message_id=%s metadata_source=%s metadata_keys=%s",
        correlation["task_id"], correlation["context_id"], correlation["message_id"],
        selected_source, metadata_keys,
    )

    if not metadata:
        logger.info(
            "hook_called_no_metadata task_id=%s context_id=%s message_id=%s",
            correlation["task_id"], correlation["context_id"], correlation["message_id"],
        )
        return None

    if not isinstance(metadata, dict):
        logger.warning(
            "hook_called_metadata_invalid_shape task_id=%s context_id=%s message_id=%s metadata_type=%s",
            correlation["task_id"], correlation["context_id"], correlation["message_id"],
            type(metadata).__name__,
        )
        return None

    fhir_data = None
    for key, value in metadata.items():
        if FHIR_CONTEXT_KEY in str(key):
            fhir_data = _coerce_fhir_data(value)
            break

    if fhir_data:
        callback_context.state["fhir_url"]   = fhir_data.get("fhirUrl", "")
        callback_context.state["patient_id"] = fhir_data.get("patientId", "")

        token = fhir_data.get("fhirToken")
        callback_context.state["fhir_token"] = token

        logger.info("FHIR_URL_FOUND value=%s", callback_context.state["fhir_url"] or "[EMPTY]")
        logger.info("FHIR_PATIENT_FOUND value=%s", callback_context.state["patient_id"] or "[EMPTY]")

        if token:
            logger.info("FHIR_TOKEN_FOUND fingerprint=%s", token_fingerprint(token))
        else:
            logger.info("FHIR_TOKEN_NOT_PROVIDED (continuing)")

        logger.info(
            "hook_called_fhir_found task_id=%s context_id=%s message_id=%s patient_id=%s fhir_url_set=%s",
            correlation["task_id"], correlation["context_id"], correlation["message_id"],
            callback_context.state["patient_id"],
            bool(callback_context.state["fhir_url"]),
        )
    else:
        logger.info(
            "hook_called_fhir_not_found task_id=%s context_id=%s message_id=%s metadata_keys=%s",
            correlation["task_id"], correlation["context_id"], correlation["message_id"],
            metadata_keys,
        )

    logger.info(
        "hook_called_exit task_id=%s context_id=%s message_id=%s patient_id=%s",
        correlation["task_id"], correlation["context_id"], correlation["message_id"],
        callback_context.state.get("patient_id", ""),
    )

    return None
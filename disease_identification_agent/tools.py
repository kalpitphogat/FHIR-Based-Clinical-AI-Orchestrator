"""
NLM Health Topics search tools — query the NLM MedlinePlus web-service
to find diseases/conditions associated with patient symptoms.

API endpoint:
    https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term=<symptom>&retmax=500

Each tool reads symptom strings, calls the NLM API, parses the XML response,
and extracts disease names from <content name="title"> and <content name="altTitle">
elements.

Usage:
    The disease_identification_agent registers these tools so the LLM can
    search symptoms individually or in batch.
"""
import logging
import re
import xml.etree.ElementTree as ET
from collections import Counter

import httpx
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

NLM_BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"
NLM_TIMEOUT = 20  # seconds


# ── Private helpers ────────────────────────────────────────────────────────────

def _strip_html_tags(text: str) -> str:
    """Remove HTML / XML tags and decode common entities from NLM text."""
    # The NLM API HTML-encodes angle brackets inside content values,
    # e.g. &lt;span class="qt0"&gt;Headache&lt;/span&gt;
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _query_nlm(symptom: str) -> list[dict]:
    """
    Call the NLM Health Topics search API for a single symptom.

    Returns a list of dicts, each with:
        { "title": str, "alt_titles": list[str], "url": str }
    """
    params = {
        "db": "healthTopics",
        "term": symptom,
        "retmax": "500",
    }
    try:
        response = httpx.get(NLM_BASE_URL, params=params, timeout=NLM_TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "NLM API HTTP error for symptom=%s status=%s", symptom, exc.response.status_code
        )
        return []
    except Exception as exc:
        logger.warning("NLM API connection error for symptom=%s error=%s", symptom, exc)
        return []

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        logger.warning("NLM XML parse error for symptom=%s error=%s", symptom, exc)
        return []

    results = []
    for doc in root.iter("document"):
        url = doc.get("url", "")
        title = ""
        alt_titles = []

        for content_elem in doc.iter("content"):
            name = content_elem.get("name", "")
            raw_text = content_elem.text or ""
            clean_text = _strip_html_tags(raw_text)

            if name == "title" and clean_text:
                title = clean_text
            elif name == "altTitle" and clean_text:
                alt_titles.append(clean_text)

        if title:
            results.append({
                "title": title,
                "alt_titles": alt_titles,
                "url": url,
            })

    return results


def _collect_disease_names(nlm_results: list[dict]) -> list[str]:
    """Extract all unique disease names (title + altTitles) from NLM results."""
    names = set()
    for entry in nlm_results:
        names.add(entry["title"])
        for alt in entry.get("alt_titles", []):
            names.add(alt)
    return sorted(names)


# ── Tool: search diseases for a single symptom ────────────────────────────────

def search_diseases_for_symptom(symptom: str, tool_context: ToolContext) -> dict:
    """
    Searches the NLM MedlinePlus Health Topics API for a single symptom
    and returns a list of diseases/conditions associated with it.

    Args:
        symptom: A single symptom to search for, e.g. "headache", "fever",
                 "shortness of breath".

    Returns a dict with:
        - status: "success" or "error"
        - symptom: the searched symptom
        - disease_count: number of unique diseases found
        - diseases: list of disease name strings (from title + altTitle)
        - details: list of dicts with title, alt_titles, and url for each result
    """
    symptom = symptom.strip()
    if not symptom:
        return {"status": "error", "error_message": "Symptom cannot be empty."}

    logger.info("tool_search_diseases_for_symptom symptom=%s", symptom)

    nlm_results = _query_nlm(symptom)
    disease_names = _collect_disease_names(nlm_results)

    return {
        "status": "success",
        "symptom": symptom,
        "disease_count": len(disease_names),
        "diseases": disease_names,
        "details": nlm_results,
    }


# ── Tool: batch search + aggregation ──────────────────────────────────────────

def search_and_aggregate_diseases(symptoms: str, tool_context: ToolContext) -> dict:
    """
    Searches the NLM MedlinePlus Health Topics API for EACH symptom in a
    comma-separated list and aggregates the results.

    For every symptom, the tool queries the API, collects disease names
    (from both title and altTitle), and counts how often each disease
    appears across ALL symptom searches.

    Args:
        symptoms: A comma-separated string of symptoms, e.g.
                  "headache, fever, nausea, shortness of breath".

    Returns a dict with:
        - status: "success" or "error"
        - symptoms_searched: list of symptom strings that were searched
        - per_symptom_diseases: dict mapping each symptom to its list of diseases
        - disease_frequency: dict mapping each disease name to the number of
          symptoms it appeared in (sorted descending by count)
        - total_unique_diseases: total number of unique disease names found
    """
    if not symptoms or not symptoms.strip():
        return {"status": "error", "error_message": "Symptoms string cannot be empty."}

    symptom_list = [s.strip() for s in symptoms.split(",") if s.strip()]
    if not symptom_list:
        return {"status": "error", "error_message": "No valid symptoms found after parsing."}

    logger.info(
        "tool_search_and_aggregate_diseases symptoms=%s count=%d",
        symptom_list, len(symptom_list),
    )

    per_symptom: dict[str, list[str]] = {}
    frequency: Counter = Counter()

    for symptom in symptom_list:
        nlm_results = _query_nlm(symptom)
        disease_names = _collect_disease_names(nlm_results)
        per_symptom[symptom] = disease_names
        # Each disease counts once per symptom (not per NLM result)
        frequency.update(set(disease_names))

    # Sort by frequency descending
    sorted_freq = dict(frequency.most_common())

    return {
        "status": "success",
        "symptoms_searched": symptom_list,
        "per_symptom_diseases": per_symptom,
        "disease_frequency": sorted_freq,
        "total_unique_diseases": len(sorted_freq),
    }

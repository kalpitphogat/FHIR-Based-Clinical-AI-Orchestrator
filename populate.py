

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx


# https://app.promptopinion.ai/api/workspaces/019da4f4-206a-776d-940d-f545a1aa9264/fhir
# THIS IS MY FHIR URL, DO NOT USE THIS, USE YOURS.




WORKSPACE_FHIR_URL = ""   # PLEASE PUT YOUR FHIR URL HERE 
FHIR_TOKEN         = ""         
DATA_FOLDER        = Path(__file__).parent / "populate_data"
MAX_WORKERS        = 5          


FHIR_URL = WORKSPACE_FHIR_URL.rstrip("/")
TIMEOUT  = 15

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)

def ok(tag, msg):   tprint(f"  {GREEN}✔{RESET} [{tag}] {msg}")
def warn(tag, msg): tprint(f"  {YELLOW}⚠{RESET} [{tag}] {msg}")
def info(tag, msg): tprint(f"  {CYAN}→{RESET} [{tag}] {msg}")
def err(tag, msg):  tprint(f"  {RED}✘{RESET} [{tag}] {msg}")


@dataclass
class PatientResult:
    file:        str
    patient_id:  str  = ""
    name:        str  = ""
    success:     bool = False
    errors:      list = field(default_factory=list)
    counts: dict      = field(default_factory=lambda: {
        "medications": 0,
        "conditions":  0,
        "observations": 0,
        "reports":     0,
    })


def _headers(content_type: bool = False) -> dict:
    h = {"Accept": "application/fhir+json"}
    if content_type:
        h["Content-Type"] = "application/fhir+json"
    if FHIR_TOKEN:
        h["Authorization"] = f"Bearer {FHIR_TOKEN}"
    return h

def fhir_post(resource_type: str, body: dict) -> dict:
    r = httpx.post(
        f"{FHIR_URL}/{resource_type}",
        json=body,
        headers=_headers(content_type=True),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()

def fhir_get(path: str, params: dict | None = None) -> dict:
    r = httpx.get(
        f"{FHIR_URL}/{path}",
        params=params,
        headers=_headers(),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def _post_patient(patient_data: dict, tag: str) -> tuple[str, str]:
    
    result = fhir_post("Patient", patient_data)
    patient_id = result.get("id")
    if not patient_id:
        raise ValueError("Server returned no id for Patient resource")

    fetched = fhir_get(f"Patient/{patient_id}")
    names    = fetched.get("name", [])
    official = next((n for n in names if n.get("use") == "official"), names[0] if names else {})
    full_name = (" ".join(official.get("given", [])) + " " + official.get("family", "")).strip()
    ok(tag, f"Patient created → {BOLD}{full_name}{RESET}  (id={patient_id})")
    return patient_id, full_name


def _post_medications(patient_id: str, medications: list, tag: str) -> int:
    count = 0
    for med in medications:
        resource = {
            "resourceType": "MedicationRequest",
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {"text": med["medication_name"]},
            "subject":    {"reference": f"Patient/{patient_id}"},
            "authoredOn": med.get("authored_on", datetime.today().strftime("%Y-%m-%d")),
            "dosageInstruction": [{"text": med["dosage"]}],
        }
        try:
            fhir_post("MedicationRequest", resource)
            count += 1
        except Exception as e:
            warn(tag, f"Medication '{med['medication_name']}' failed: {e}")
    ok(tag, f"{count}/{len(medications)} medications added")
    return count


def _post_conditions(patient_id: str, conditions: list, tag: str) -> int:
    count = 0
    for cond in conditions:
        resource = {
            "resourceType": "Condition",
            "clinicalStatus": {
                "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]
            },
            "verificationStatus": {
                "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]
            },
            "code":    {"text": cond["condition_name"]},
            "subject": {"reference": f"Patient/{patient_id}"},
            "severity": {"text": cond["severity"]},
            "onsetDateTime": cond.get("onset"),
        }
        try:
            fhir_post("Condition", resource)
            count += 1
        except Exception as e:
            warn(tag, f"Condition '{cond['condition_name']}' failed: {e}")
    ok(tag, f"{count}/{len(conditions)} conditions added")
    return count


def _post_observations(patient_id: str, observations: list, bp: dict | None, tag: str) -> int:
    count = 0

    for obs in observations:
        resource = {
            "resourceType": "Observation",
            "status": "final",
            "category": [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": obs["category"],
            }]}],
            "code":    {"text": obs["name"]},
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": obs.get("effective_date", datetime.utcnow().isoformat() + "Z"),
            "valueQuantity": {"value": obs["value"], "unit": obs["unit"]},
        }
        try:
            fhir_post("Observation", resource)
            count += 1
        except Exception as e:
            warn(tag, f"Observation '{obs['name']}' failed: {e}")

    if bp:
        bp_resource = {
            "resourceType": "Observation",
            "status": "final",
            "category": [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "vital-signs",
            }]}],
            "code": {
                "coding": [{"system": "http://loinc.org", "code": "85354-9"}],
                "text": "Blood Pressure",
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": bp.get("effective_date", datetime.utcnow().isoformat() + "Z"),
            "component": [
                {
                    "code": {
                        "coding": [{"system": "http://loinc.org", "code": "8480-6"}],
                        "text": "Systolic Blood Pressure",
                    },
                    "valueQuantity": {"value": bp["systolic"]["value"], "unit": bp["systolic"]["unit"]},
                },
                {
                    "code": {
                        "coding": [{"system": "http://loinc.org", "code": "8462-4"}],
                        "text": "Diastolic Blood Pressure",
                    },
                    "valueQuantity": {"value": bp["diastolic"]["value"], "unit": bp["diastolic"]["unit"]},
                },
            ],
        }
        try:
            fhir_post("Observation", bp_resource)
            count += 1
        except Exception as e:
            warn(tag, f"Blood Pressure failed: {e}")

    ok(tag, f"{count} observation(s) added")
    return count


def _post_diagnostic_reports(patient_id: str, reports: list, tag: str) -> int:
    count = 0
    for rpt in reports:
        resource = {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "code":    {"text": rpt["report_name"]},
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": datetime.today().strftime("%Y-%m-%d"),
            "conclusion": rpt["conclusion"],
        }
        try:
            fhir_post("DiagnosticReport", resource)
            count += 1
        except Exception as e:
            warn(tag, f"DiagnosticReport '{rpt['report_name']}' failed: {e}")
    ok(tag, f"{count}/{len(reports)} diagnostic reports added")
    return count


def process_file(json_file: Path) -> PatientResult:
    tag    = json_file.stem          # e.g. "demo1", "demo2"
    result = PatientResult(file=json_file.name)

    info(tag, f"Loading {json_file.name} …")

    try:
        with json_file.open() as f:
            data = json.load(f)
    except Exception as e:
        result.errors.append(f"JSON parse error: {e}")
        err(tag, f"Could not load {json_file.name}: {e}")
        return result

    try:
        patient_id, full_name = _post_patient(data["patient"], tag)
        result.patient_id = patient_id
        result.name       = full_name
    except Exception as e:
        result.errors.append(f"Patient creation failed: {e}")
        err(tag, f"Patient creation failed: {e}")
        return result   # can't continue without a patient id

    try:
        result.counts["medications"] = _post_medications(patient_id, data.get("medications", []), tag)
    except Exception as e:
        result.errors.append(f"Medications: {e}")

    try:
        result.counts["conditions"] = _post_conditions(patient_id, data.get("conditions", []), tag)
    except Exception as e:
        result.errors.append(f"Conditions: {e}")

    try:
        result.counts["observations"] = _post_observations(
            patient_id,
            data.get("observations", []),
            data.get("blood_pressure"),
            tag,
        )
    except Exception as e:
        result.errors.append(f"Observations: {e}")

    try:
        result.counts["reports"] = _post_diagnostic_reports(patient_id, data.get("diagnostic_reports", []), tag)
    except Exception as e:
        result.errors.append(f"DiagnosticReports: {e}")

    result.success = len(result.errors) == 0
    return result


def print_summary(results: list[PatientResult]):
    with _print_lock:
        print(f"\n{BOLD}{'─'*72}{RESET}")
        print(f"{BOLD}  FINAL SUMMARY{RESET}")
        print(f"{BOLD}{'─'*72}{RESET}")
        print(f"  {'FILE':<14} {'PATIENT':<28} {'MEDS':>4} {'COND':>4} {'OBS':>4} {'RPT':>4}  STATUS")
        print(f"  {'─'*13} {'─'*27} {'─'*4} {'─'*4} {'─'*4} {'─'*4}  {'─'*10}")

        passed = failed = 0
        for r in sorted(results, key=lambda x: x.file):
            status = f"{GREEN}OK{RESET}" if r.success else f"{RED}ERRORS{RESET}"
            name   = r.name[:27] if r.name else "(failed)"
            meds   = str(r.counts["medications"])  if r.patient_id else "-"
            conds  = str(r.counts["conditions"])   if r.patient_id else "-"
            obs    = str(r.counts["observations"]) if r.patient_id else "-"
            rpts   = str(r.counts["reports"])      if r.patient_id else "-"
            print(f"  {r.file:<14} {name:<28} {meds:>4} {conds:>4} {obs:>4} {rpts:>4}  {status}")

            if r.errors:
                for e in r.errors:
                    print(f"    {YELLOW}↳ {e}{RESET}")

            if r.success:
                passed += 1
            else:
                failed += 1

        print(f"{BOLD}{'─'*72}{RESET}")
        print(f"  {GREEN}{passed} succeeded{RESET}  ·  {RED}{failed} failed{RESET}  ·  "
              f"{passed + failed} total\n")


def main():
    print(f"\n{BOLD}FHIR Bulk Populator{RESET}  (parallel, {MAX_WORKERS} workers)")
    print(f"  Workspace : {CYAN}{WORKSPACE_FHIR_URL}{RESET}")
    print(f"  Data dir  : {CYAN}{DATA_FOLDER}{RESET}")

    if not DATA_FOLDER.exists():
        print(f"{RED}✘{RESET}  Data folder not found: {DATA_FOLDER}")
        sys.exit(1)

    json_files = sorted(DATA_FOLDER.glob("*.json"))
    if not json_files:
        print(f"{YELLOW}⚠{RESET}  No *.json files found in {DATA_FOLDER}")
        sys.exit(0)

    print(f"  Files     : {CYAN}{len(json_files)} found{RESET}  "
          f"({', '.join(f.name for f in json_files)})\n")

    print(f"{CYAN}→{RESET}  Checking FHIR server …")
    try:
        meta = fhir_get("metadata")
        sv   = meta.get("software", {}).get("name", "FHIR server")
        fv   = meta.get("fhirVersion", "?")
        print(f"{GREEN}✔{RESET}  {sv} is up  (FHIR {fv})\n")
    except Exception as e:
        print(f"{RED}✘{RESET}  Cannot reach {FHIR_URL}/metadata\n  {e}")
        sys.exit(1)

    workers = min(MAX_WORKERS, len(json_files))
    print(f"{CYAN}→{RESET}  Launching {workers} parallel worker(s) …\n")

    results: list[PatientResult] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, f): f for f in json_files}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                fname = futures[future].name
                err(fname, f"Unexpected worker error: {exc}")
                results.append(PatientResult(file=fname, errors=[str(exc)]))

    print_summary(results)


if __name__ == "__main__":
    main()
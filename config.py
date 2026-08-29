"""
config.py — shared constants and configuration/secret resolution.

No other module should hardcode a Synthea filename, an expected column list,
or a LOINC lab code — everything lives here so the rest of the codebase stays
in sync with a single source of truth.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set

# ------------------------------------------------------------------------- #
# Synthea file inventory
# ------------------------------------------------------------------------- #

SYNTHEA_FILES: List[str] = [
    "allergies.csv", "careplans.csv", "conditions.csv", "devices.csv",
    "encounters.csv", "imaging_studies.csv", "immunizations.csv",
    "medications.csv", "observations.csv", "organizations.csv",
    "patients.csv", "payer_transitions.csv", "payers.csv",
    "procedures.csv", "providers.csv", "supplies.csv",
]

# Minimal expected-column schema per file. Used to build an empty, correctly
# shaped DataFrame when a file is missing or completely empty, so downstream
# joins never KeyError.
FILE_SCHEMAS: Dict[str, List[str]] = {
    "patients.csv": [
        "Id", "BIRTHDATE", "DEATHDATE", "FIRST", "LAST", "GENDER", "RACE",
        "ETHNICITY", "MARITAL", "ADDRESS", "CITY", "STATE", "ZIP",
    ],
    "encounters.csv": [
        "Id", "START", "STOP", "PATIENT", "ORGANIZATION", "PROVIDER", "PAYER",
        "ENCOUNTERCLASS", "CODE", "DESCRIPTION", "REASONCODE", "REASONDESCRIPTION",
    ],
    "allergies.csv": [
        "START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "SYSTEM", "DESCRIPTION",
        "TYPE", "CATEGORY", "REACTION1", "DESCRIPTION1", "SEVERITY1",
        "REACTION2", "DESCRIPTION2", "SEVERITY2",
    ],
    "careplans.csv": [
        "Id", "START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION",
        "REASONCODE", "REASONDESCRIPTION",
    ],
    "conditions.csv": ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"],
    "devices.csv": ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "UDI"],
    "imaging_studies.csv": [
        "Id", "DATE", "PATIENT", "ENCOUNTER", "BODYSITE_DESCRIPTION",
        "MODALITY_DESCRIPTION", "SOP_DESCRIPTION", "PROCEDURE_CODE",
    ],
    "immunizations.csv": ["DATE", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "BASE_COST"],
    "medications.csv": [
        "START", "STOP", "PATIENT", "PAYER", "ENCOUNTER", "CODE", "DESCRIPTION",
        "REASONCODE", "REASONDESCRIPTION",
    ],
    "observations.csv": ["DATE", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "VALUE", "UNITS", "TYPE"],
    "organizations.csv": ["Id", "NAME", "ADDRESS", "CITY", "STATE", "ZIP", "PHONE"],
    "payer_transitions.csv": ["PATIENT", "START_YEAR", "END_YEAR", "PAYER", "OWNERSHIP"],
    "payers.csv": ["Id", "NAME", "ADDRESS", "CITY", "STATE_HEADQUARTERED", "ZIP", "PHONE"],
    "procedures.csv": [
        "START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION",
        "REASONCODE", "REASONDESCRIPTION",
    ],
    "providers.csv": [
        "Id", "ORGANIZATION", "NAME", "GENDER", "SPECIALITY", "ADDRESS",
        "CITY", "STATE", "ZIP",
    ],
    "supplies.csv": ["DATE", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "QUANTITY"],
}

# ------------------------------------------------------------------------- #
# Clinical constants
# ------------------------------------------------------------------------- #

# LOINC codes for the "recent critical lab panels" the spec calls out.
CRITICAL_LAB_CODES: Dict[str, Set[str]] = {
    "eGFR": {"48642-3", "48643-1", "33914-3"},
    "HbA1c": {"4548-4"},
    "Systolic_BP": {"8480-6"},
    "Diastolic_BP": {"8462-4"},
}

CLINICAL_SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant operating strictly within a "
    "human-in-the-loop workflow. You are given a U-shaped context: authoritative "
    "guidelines at the top, general patient history in the middle, and hard "
    "safety constraints (allergies, organ dysfunction) pinned at the bottom. "
    "The bottom safety section always takes precedence over anything above it. "
    "Produce a concise clinical summary and flag any medication or treatment "
    "conflicts with the pinned safety constraints. You do not make a final "
    "treatment decision — a licensed clinician must review and approve your "
    "output before it is acted on."
)


# ------------------------------------------------------------------------- #
# Secret / config resolution (Streamlit secrets -> environment -> default)
# ------------------------------------------------------------------------- #

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read a config value from Streamlit secrets first, then environment
    variables. Never raises — returns `default` if not found anywhere.
    Safe to call even outside a running Streamlit app (e.g. from tests)."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:  # noqa: BLE001 - streamlit may not be installed/running in test contexts
        pass
    return os.environ.get(key, default)

"""
clinical_aggregation.py — per-patient clinical data aggregation.

Pulls matching line items from allergies.csv, medications.csv,
observations.csv, and every other clinical table into a single
PatientClinicalBundle for one patient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from config import CRITICAL_LAB_CODES
from data_loader import ConsolidatedSyntheaData


@dataclass
class PatientClinicalBundle:
    patient_id: str
    demographics: Dict[str, str]
    allergies: List[Dict[str, str]]
    active_medications: List[Dict[str, str]]
    critical_labs: Dict[str, Dict[str, str]]
    conditions: List[Dict[str, str]]
    procedures: List[Dict[str, str]]
    careplans: List[Dict[str, str]]
    immunizations: List[Dict[str, str]]
    devices: List[Dict[str, str]]
    imaging_studies: List[Dict[str, str]]
    supplies: List[Dict[str, str]]
    encounter_count: int


def _filter_patient(df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    if df.empty or "PATIENT" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df[df["PATIENT"] == patient_id]


def _safe_records(df: pd.DataFrame, columns: List[str]) -> List[Dict[str, str]]:
    present_cols = [c for c in columns if c in df.columns]
    if df.empty or not present_cols:
        return []
    return df[present_cols].drop_duplicates().to_dict("records")


def get_allergy_profile(allergies_df: pd.DataFrame, patient_id: str) -> List[Dict[str, str]]:
    pdf = _filter_patient(allergies_df, patient_id)
    return _safe_records(pdf, ["DESCRIPTION", "CATEGORY", "REACTION1", "DESCRIPTION1", "SEVERITY1"])


def get_active_medications(medications_df: pd.DataFrame, patient_id: str) -> List[Dict[str, str]]:
    pdf = _filter_patient(medications_df, patient_id)
    if pdf.empty:
        return []
    if "STOP" in pdf.columns:
        active = pdf[pdf["STOP"] == ""]
        pdf = active if not active.empty else pdf
    return _safe_records(pdf, ["DESCRIPTION", "START", "STOP", "REASONDESCRIPTION"])


def get_recent_critical_labs(observations_df: pd.DataFrame, patient_id: str) -> Dict[str, Dict[str, str]]:
    pdf = _filter_patient(observations_df, patient_id)
    if pdf.empty or "CODE" not in pdf.columns:
        return {}

    pdf = pdf.copy()
    pdf["_DATE_PARSED"] = pd.to_datetime(pdf.get("DATE", ""), errors="coerce")

    results: Dict[str, Dict[str, str]] = {}
    for label, codes in CRITICAL_LAB_CODES.items():
        subset = pdf[pdf["CODE"].isin(codes)]
        if subset.empty:
            continue
        subset = subset.sort_values("_DATE_PARSED", ascending=False, na_position="last")
        top = subset.iloc[0]
        results[label] = {
            "value": str(top.get("VALUE", "")),
            "units": str(top.get("UNITS", "")),
            "date": str(top.get("DATE", "")),
        }
    return results


def build_patient_clinical_bundle(consolidated: ConsolidatedSyntheaData, patient_id: str) -> PatientClinicalBundle:
    """Aggregate every table's rows for one patient into a single bundle."""
    patients_df = consolidated.table("patients")
    demo_row = patients_df[patients_df["Id"] == patient_id] if "Id" in patients_df.columns else pd.DataFrame()
    demographics = demo_row.iloc[0].to_dict() if not demo_row.empty else {"Id": patient_id}

    encounters_pdf = _filter_patient(consolidated.table("encounters"), patient_id)

    return PatientClinicalBundle(
        patient_id=patient_id,
        demographics=demographics,
        allergies=get_allergy_profile(consolidated.table("allergies"), patient_id),
        active_medications=get_active_medications(consolidated.table("medications"), patient_id),
        critical_labs=get_recent_critical_labs(consolidated.table("observations"), patient_id),
        conditions=_safe_records(_filter_patient(consolidated.table("conditions"), patient_id),
                                  ["DESCRIPTION", "START", "STOP"]),
        procedures=_safe_records(_filter_patient(consolidated.table("procedures"), patient_id),
                                  ["DESCRIPTION", "START", "REASONDESCRIPTION"]),
        careplans=_safe_records(_filter_patient(consolidated.table("careplans"), patient_id),
                                 ["DESCRIPTION", "START", "STOP", "REASONDESCRIPTION"]),
        immunizations=_safe_records(_filter_patient(consolidated.table("immunizations"), patient_id),
                                     ["DESCRIPTION", "DATE"]),
        devices=_safe_records(_filter_patient(consolidated.table("devices"), patient_id),
                               ["DESCRIPTION", "START", "STOP"]),
        imaging_studies=_safe_records(_filter_patient(consolidated.table("imaging_studies"), patient_id),
                                       ["BODYSITE_DESCRIPTION", "MODALITY_DESCRIPTION", "DATE"]),
        supplies=_safe_records(_filter_patient(consolidated.table("supplies"), patient_id),
                                ["DESCRIPTION", "DATE", "QUANTITY"]),
        encounter_count=int(len(encounters_pdf)),
    )

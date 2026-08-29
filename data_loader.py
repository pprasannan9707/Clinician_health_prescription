"""
data_loader.py — DYNAMIC SYNTHEA FILE INGESTION.

Reads the 16 raw Synthea CSVs defensively and builds the derived Provider
Access Control List (patient -> set of provider IDs) by joining
patients.csv <-> encounters.csv.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Set

import pandas as pd

from config import FILE_SCHEMAS, SYNTHEA_FILES

logger = logging.getLogger("clinical_rag.data_loader")


def _empty_schema_frame(filename: str) -> pd.DataFrame:
    """Return a correctly-columned, empty DataFrame for a missing/blank file."""
    return pd.DataFrame(columns=FILE_SCHEMAS.get(filename, []))


def safe_read_csv(data_dir: str, filename: str) -> pd.DataFrame:
    """
    Read a single Synthea CSV defensively.

    Handles: missing file, zero-byte file, header-only file, stray whitespace
    in column names, and mixed-type ID columns (everything is read as string
    so UUID-style Synthea IDs never get corrupted by pandas type inference).
    """
    path = os.path.join(data_dir, filename)

    if not os.path.isfile(path):
        logger.warning("Synthea file not found, substituting empty frame: %s", path)
        return _empty_schema_frame(filename)

    if os.path.getsize(path) == 0:
        logger.warning("Synthea file is zero-byte, substituting empty frame: %s", path)
        return _empty_schema_frame(filename)

    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    except pd.errors.EmptyDataError:
        logger.warning("Synthea file has no parseable rows: %s", path)
        return _empty_schema_frame(filename)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the pipeline
        logger.error("Failed to read %s: %s", path, exc)
        return _empty_schema_frame(filename)

    # Normalize column names (strip stray whitespace some Synthea exports include).
    df.columns = [str(c).strip() for c in df.columns]

    # Uniform empty-string representation instead of NaN, since we do a lot of
    # string equality / membership testing on clinical fields downstream.
    df = df.where(pd.notnull(df), "")

    return df


@dataclass
class ConsolidatedSyntheaData:
    """Container for every parsed Synthea table plus derived indices."""

    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    provider_acl: Dict[str, Set[str]] = field(default_factory=dict)   # PATIENT id -> {PROVIDER ids}
    load_warnings: List[str] = field(default_factory=list)

    def table(self, name: str) -> pd.DataFrame:
        return self.tables.get(name, pd.DataFrame())


def load_and_consolidate_synthea(data_dir: str) -> ConsolidatedSyntheaData:
    """
    Load all 16 raw Synthea CSVs from `data_dir`, and build the derived
    structures the rest of the app depends on:

      1. tables       -> dict[filename_stem] = DataFrame for all 16 files
      2. provider_acl -> dict[patient_id] = set(provider_id) built by joining
                          patients.csv <-> encounters.csv. This is the
                          ground-truth multi-tenant ACL: a doctor may only
                          access a patient's record if their Provider ID
                          appears in that patient's actual encounter history.

    Every file is read defensively via `safe_read_csv`, so a missing or
    empty file degrades to an empty (but correctly shaped) table rather than
    raising.
    """
    warnings: List[str] = []
    tables: Dict[str, pd.DataFrame] = {}

    for filename in SYNTHEA_FILES:
        stem = filename.replace(".csv", "")
        df = safe_read_csv(data_dir, filename)
        if df.empty:
            warnings.append(f"'{filename}' produced 0 rows (missing, empty, or unreadable).")
        tables[stem] = df

    patients_df = tables["patients"]
    encounters_df = tables["encounters"]

    # --- Build the Provider Access Control List -----------------------------
    # This is the security-critical join: it is the *only* source of truth
    # for "has this doctor ever treated this patient".
    provider_acl: Dict[str, Set[str]] = {}
    if not encounters_df.empty and "PATIENT" in encounters_df.columns:
        # Sanity-join against patients.csv so we only build ACL entries for
        # patients that actually exist in the roster (defends against
        # orphaned/corrupt encounter rows).
        valid_patient_ids: Set[str] = set(patients_df["Id"]) if "Id" in patients_df.columns else set()

        joinable = encounters_df.copy()
        if valid_patient_ids:
            joinable = joinable[joinable["PATIENT"].isin(valid_patient_ids)]

        provider_col = joinable["PROVIDER"] if "PROVIDER" in joinable.columns else pd.Series(dtype=str)
        joinable = joinable.assign(PROVIDER=provider_col)

        for patient_id, group in joinable.groupby("PATIENT"):
            providers = {p for p in group["PROVIDER"].tolist() if p}
            if patient_id:
                provider_acl[patient_id] = providers
    else:
        warnings.append("encounters.csv missing/empty — Provider ACL could not be built; ALL access will be denied.")

    return ConsolidatedSyntheaData(tables=tables, provider_acl=provider_acl, load_warnings=warnings)

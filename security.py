"""
security.py — STRICT MULTI-TENANCY INTERCEPTION.

A doctor may only access a patient's record if their Provider ID appears in
that patient's Provider ACL (built in data_loader.py from real encounter
history). This check must run before any retrieval, embedding, or generation
logic — nothing in this module performs RAG work; it is a pure, deterministic
gate.
"""

from __future__ import annotations

from data_loader import ConsolidatedSyntheaData


class MultiTenancyAccessError(Exception):
    """Raised when the active doctor session has no encounter history with the
    requested patient."""

    def __init__(self, doctor_id: str, patient_id: str, reason: str):
        self.doctor_id = doctor_id
        self.patient_id = patient_id
        self.reason = reason
        super().__init__(f"ACCESS DENIED — doctor='{doctor_id}' patient='{patient_id}' :: {reason}")


def check_multi_tenant_access(doctor_id: str, patient_id: str, consolidated: ConsolidatedSyntheaData) -> None:
    """
    Strict tenancy gate. Raises MultiTenancyAccessError if the doctor is not
    present in the patient's encounter-derived Provider ACL. This function
    performs NO retrieval, embedding, or generation — it is a pure,
    deterministic security check and must run before any of that logic.
    """
    if not doctor_id:
        raise MultiTenancyAccessError(doctor_id, patient_id, "No active doctor session (unauthenticated).")
    if not patient_id:
        raise MultiTenancyAccessError(doctor_id, patient_id, "No patient selected.")

    authorized_providers = consolidated.provider_acl.get(patient_id)

    if authorized_providers is None:
        raise MultiTenancyAccessError(
            doctor_id, patient_id,
            "Patient has no encounter history in the ACL index (unknown or orphaned record)."
        )

    if doctor_id not in authorized_providers:
        raise MultiTenancyAccessError(
            doctor_id, patient_id,
            f"Doctor is not among the {len(authorized_providers)} provider(s) with a recorded "
            f"encounter for this patient."
        )
    # Access granted — falls through silently.

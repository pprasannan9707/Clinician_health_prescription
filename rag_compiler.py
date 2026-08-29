"""
rag_compiler.py — DUAL-RAG PROMPT COMPILER WITH U-SHAPED RE-RANKING.

Turns a PatientClinicalBundle into de-duplicated text chunks, then arranges
them into a strict U-shaped layout to counter "Lost in the Middle":

    TOP     -> external clinical guidelines
    MIDDLE  -> general patient clinical history
    BOTTOM  -> critical drug-allergy / organ-dysfunction safety flags (pinned)
"""

from __future__ import annotations

import hashlib
from typing import List, Set

from clinical_aggregation import PatientClinicalBundle
from config import CLINICAL_SYSTEM_PROMPT  # re-exported for convenience

__all__ = [
    "sha256_hash", "dedupe_chunks", "bundle_to_history_chunks",
    "bundle_to_safety_chunks", "compile_u_shaped_prompt", "CLINICAL_SYSTEM_PROMPT",
]


def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def dedupe_chunks(chunks: List[str]) -> List[str]:
    """Collapse whitespace, hash each chunk with SHA-256, and drop exact
    repeats while preserving first-seen order."""
    seen: Set[str] = set()
    deduped: List[str] = []
    for raw in chunks:
        if raw is None:
            continue
        normalized = " ".join(str(raw).split())
        if not normalized:
            continue
        h = sha256_hash(normalized)
        if h in seen:
            continue
        seen.add(h)
        deduped.append(normalized)
    return deduped


def bundle_to_history_chunks(bundle: PatientClinicalBundle) -> List[str]:
    """Turn the general (non-safety-critical) clinical history into flat text
    chunks suitable for the MIDDLE of the U-shaped prompt."""
    chunks: List[str] = []

    for c in bundle.conditions:
        chunks.append(f"Condition: {c.get('DESCRIPTION','')} (onset {c.get('START','unknown')}, "
                       f"resolved {c.get('STOP','ongoing') or 'ongoing'}).")

    for p in bundle.procedures:
        chunks.append(f"Procedure: {p.get('DESCRIPTION','')} on {p.get('START','unknown date')}; "
                       f"reason: {p.get('REASONDESCRIPTION','') or 'not specified'}.")

    for cp in bundle.careplans:
        chunks.append(f"Care plan: {cp.get('DESCRIPTION','')} started {cp.get('START','unknown')}, "
                       f"reason: {cp.get('REASONDESCRIPTION','') or 'not specified'}.")

    for i in bundle.immunizations:
        chunks.append(f"Immunization: {i.get('DESCRIPTION','')} administered {i.get('DATE','unknown date')}.")

    for d in bundle.devices:
        chunks.append(f"Device: {d.get('DESCRIPTION','')} (implanted/fitted {d.get('START','unknown')}).")

    for im in bundle.imaging_studies:
        chunks.append(f"Imaging study: {im.get('MODALITY_DESCRIPTION','')} of "
                       f"{im.get('BODYSITE_DESCRIPTION','unspecified site')} on {im.get('DATE','unknown date')}.")

    for s in bundle.supplies:
        chunks.append(f"Supply used: {s.get('DESCRIPTION','')} x{s.get('QUANTITY','1')} on "
                       f"{s.get('DATE','unknown date')}.")

    for med in bundle.active_medications:
        chunks.append(f"Medication history: {med.get('DESCRIPTION','')} started {med.get('START','unknown')}, "
                       f"reason: {med.get('REASONDESCRIPTION','') or 'not specified'}.")

    for label, lab in bundle.critical_labs.items():
        chunks.append(f"Lab result — {label}: {lab.get('value','')} {lab.get('units','')} "
                       f"(recorded {lab.get('date','unknown date')}).")

    chunks.append(f"Total recorded encounters on file: {bundle.encounter_count}.")

    return chunks


def bundle_to_safety_chunks(bundle: PatientClinicalBundle) -> List[str]:
    """Safety-critical flags (allergies + organ-dysfunction-relevant labs)
    that must be PINNED to the BOTTOM of the U-shaped context, since
    recency/position bias makes the end of a long context the most reliably
    attended-to region for an LLM, alongside the very start."""
    chunks: List[str] = []

    if not bundle.allergies:
        chunks.append("NO KNOWN DRUG OR ENVIRONMENTAL ALLERGIES RECORDED.")
    for a in bundle.allergies:
        severity = a.get("SEVERITY1", "") or "severity unspecified"
        reaction = a.get("REACTION1", "") or a.get("DESCRIPTION1", "") or "reaction unspecified"
        chunks.append(f"*** ALLERGY FLAG *** {a.get('DESCRIPTION','Unknown allergen')} "
                       f"— severity: {severity}; reaction: {reaction}.")

    egfr = bundle.critical_labs.get("eGFR")
    if egfr:
        try:
            egfr_val = float(egfr["value"])
            flag = " (CKD-RANGE — DOSE-ADJUST RENALLY-CLEARED DRUGS)" if egfr_val < 60 else ""
        except (ValueError, TypeError):
            flag = ""
        chunks.append(f"*** ORGAN FUNCTION FLAG *** Most recent eGFR: {egfr['value']} {egfr['units']} "
                       f"(as of {egfr['date']}){flag}.")

    hba1c = bundle.critical_labs.get("HbA1c")
    if hba1c:
        chunks.append(f"*** METABOLIC FLAG *** Most recent HbA1c: {hba1c['value']} {hba1c['units']} "
                       f"(as of {hba1c['date']}).")

    sbp = bundle.critical_labs.get("Systolic_BP")
    dbp = bundle.critical_labs.get("Diastolic_BP")
    if sbp or dbp:
        chunks.append(f"*** HEMODYNAMIC FLAG *** Most recent BP: "
                       f"{sbp.get('value','?') if sbp else '?'}/{dbp.get('value','?') if dbp else '?'} mmHg.")

    return chunks


def compile_u_shaped_prompt(guideline_chunks: List[str], history_chunks: List[str],
                             safety_chunks: List[str]) -> str:
    """
    Build the final RAG context using a strict U-shaped layout:

        [ TOP ]    External clinical guidelines           <- primacy-anchored
        [ MID ]    General patient clinical history        <- attention-weak zone
        [ BOTTOM ] Allergies & organ dysfunction flags     <- recency-anchored

    Each of the three sections is independently SHA-256 de-duplicated so
    repeated high-recall retrieval hits don't waste context budget or bias
    the model toward over-represented facts.
    """
    guidelines = dedupe_chunks(guideline_chunks)
    history = dedupe_chunks(history_chunks)
    safety = dedupe_chunks(safety_chunks)

    lines: List[str] = []
    lines.append("################  TOP — EXTERNAL CLINICAL PRACTICE GUIDELINES  ################")
    lines.extend(f"- {c}" for c in (guidelines or ["No external guideline context supplied for this query."]))
    lines.append("")
    lines.append("################  MIDDLE — PATIENT CLINICAL HISTORY  ################")
    lines.extend(f"- {c}" for c in (history or ["No additional clinical history on file."]))
    lines.append("")
    lines.append("################  BOTTOM (PINNED) — CRITICAL SAFETY FLAGS  ################")
    lines.append("### Allergy and organ-dysfunction flags below OVERRIDE any conflicting")
    lines.append("### guidance above. Treat every flag as a hard constraint.")
    lines.extend(f"- {c}" for c in (safety or ["No allergy or organ-dysfunction flags recorded."]))

    return "\n".join(lines)

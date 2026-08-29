"""
app.py — Streamlit UI entrypoint.

This file contains ONLY interface wiring. All pipeline logic lives in the
sibling modules:
    config.py                -> constants, secret resolution
    data_loader.py            -> DYNAMIC SYNTHEA FILE INGESTION
    clinical_aggregation.py   -> per-patient safety profile / labs / meds
    security.py                -> STRICT MULTI-TENANCY INTERCEPTION
    rag_compiler.py            -> SHA-256 dedup + U-SHAPED prompt compiler
    azure_integration.py       -> Azure OpenAI + Blob Storage
    audit.py                    -> asynchronous forensic audit trail

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy to Azure: see DEPLOYMENT.md.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
import streamlit as st

from audit import AsyncForensicAuditLogger, make_audit_event
from azure_integration import (
    build_azure_openai_client,
    build_blob_container_client,
    download_synthea_from_blob,
    generate_clinical_summary,
)
from clinical_aggregation import build_patient_clinical_bundle
from config import get_secret
from data_loader import load_and_consolidate_synthea
from rag_compiler import (
    bundle_to_history_chunks,
    bundle_to_safety_chunks,
    compile_u_shaped_prompt,
    dedupe_chunks,
    sha256_hash,
)
from security import MultiTenancyAccessError, check_multi_tenant_access

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clinical_rag.app")

st.set_page_config(page_title="Clinical RAG Copilot — Synthea / Azure", layout="wide")

# Cache the expensive ingestion step at the UI layer (keeps data_loader.py
# framework-agnostic and unit-testable without a Streamlit runtime).
cached_load_and_consolidate_synthea = st.cache_data(
    show_spinner="Reading and consolidating raw Synthea EHR export..."
)(load_and_consolidate_synthea)


def _init_session_state() -> None:
    defaults = {
        "audit_logger": None,
        "compiled_context": None,
        "generated_summary": None,
        "context_hash": None,
        "hitl_decision_made": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_audit_logger(blob_container_client) -> AsyncForensicAuditLogger:
    if st.session_state.audit_logger is None:
        st.session_state.audit_logger = AsyncForensicAuditLogger(
            local_path="audit_log.jsonl", blob_container_client=blob_container_client
        )
    return st.session_state.audit_logger


def _patient_label(row: pd.Series) -> str:
    first = row.get("FIRST", "") or "Unknown"
    last = row.get("LAST", "") or "Patient"
    pid = row.get("Id", "")
    short_id = pid[:8] if pid else "no-id"
    return f"{first} {last}  —  {short_id}"


def _provider_label(row: pd.Series) -> str:
    name = row.get("NAME", "") or "Unnamed Provider"
    speciality = row.get("SPECIALITY", "") or "General"
    pid = row.get("Id", "")
    short_id = pid[:8] if pid else "no-id"
    return f"Dr. {name} — {speciality} ({short_id})"


def main() -> None:
    _init_session_state()

    st.title("🏥 Clinical RAG Copilot — Synthea EHR / Azure OpenAI")
    st.caption(
        "Prototype only — synthetic Synthea data. Every access is authorized against a "
        "deterministic Provider ACL; every decision is written to an asynchronous forensic "
        "audit trail; every generated summary requires human-in-the-loop sign-off before use."
    )

    # -------------------------------------------------------------- SIDEBAR
    with st.sidebar:
        st.header("⚙️ Data Source")
        source_mode = st.radio("Synthea CSV source", ["Local directory", "Azure Blob Storage"], index=0)

        data_dir = "./synthea_data"
        blob_container_client = None

        if source_mode == "Local directory":
            data_dir = st.text_input("Local Synthea export directory", value="./synthea_data")
        else:
            account_url = st.text_input("Storage account URL",
                                         value=get_secret("AZURE_STORAGE_ACCOUNT_URL", "") or "")
            container_name = st.text_input("Synthea container name",
                                            value=get_secret("AZURE_SYNTHEA_CONTAINER", "") or "")
            data_dir = st.text_input("Local cache directory", value="./synthea_data_cache")
            if st.button("⬇️ Pull CSVs from Blob Storage"):
                try:
                    files = download_synthea_from_blob(account_url, container_name, data_dir)
                    st.success(f"Downloaded {len(files)} CSV file(s) into {data_dir}.")
                    cached_load_and_consolidate_synthea.clear()  # bust the cache
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Blob download failed: {exc}")

        audit_account_url = get_secret("AZURE_STORAGE_ACCOUNT_URL", "")
        audit_container_name = get_secret("AZURE_AUDIT_CONTAINER", "")
        if audit_account_url and audit_container_name:
            blob_container_client = build_blob_container_client(audit_account_url, audit_container_name)

        st.divider()
        st.header("🤖 Model Backend")
        client = build_azure_openai_client()
        if client is None:
            st.warning("Azure OpenAI not configured — running in **MOCK MODE**. "
                       "Set AZURE_OPENAI_ENDPOINT (+ AZURE_OPENAI_API_KEY or Managed Identity) to enable live generation.")
        else:
            st.success("Azure OpenAI client ready.")

        st.divider()
        st.header("📖 External Guideline Context")
        guideline_text = st.text_area(
            "Paste retrieved treatment-guideline text for this query "
            "(placed at the TOP of the U-shaped context).",
            height=140,
            placeholder="e.g. ADA 2026 Standards of Care — metformin first-line unless eGFR < 30 ...",
        )

    audit_logger = _get_audit_logger(blob_container_client)

    if not os.path.isdir(data_dir):
        st.info(f"Directory '{data_dir}' does not exist yet. Create it and drop in the 16 Synthea "
                f"CSV files, or use the Azure Blob Storage pull option in the sidebar.")
        os.makedirs(data_dir, exist_ok=True)

    consolidated = cached_load_and_consolidate_synthea(data_dir)

    if consolidated.load_warnings:
        with st.expander(f"⚠️ {len(consolidated.load_warnings)} data-load warning(s)", expanded=False):
            for w in consolidated.load_warnings:
                st.write(f"- {w}")

    patients_df = consolidated.table("patients")
    providers_df = consolidated.table("providers")

    if patients_df.empty or providers_df.empty:
        st.warning("patients.csv and/or providers.csv produced no rows. Add the raw Synthea export "
                   "to the configured data source to populate the selectors below.")
        return

    # -------------------------------------------------------- SESSION SETUP
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("👨‍⚕️ Active Doctor Session")
        provider_options = providers_df.to_dict("records")
        provider_choice = st.selectbox(
            "Logged-in Provider ID (simulated auth session)",
            options=provider_options,
            format_func=_provider_label,
        )
        active_doctor_id = provider_choice.get("Id", "") if provider_choice else ""

    with col_b:
        st.subheader("🧑‍🦽 Patient Record")
        patient_options = patients_df.to_dict("records")
        patient_choice = st.selectbox(
            "Patient to access",
            options=patient_options,
            format_func=_patient_label,
        )
        active_patient_id = patient_choice.get("Id", "") if patient_choice else ""

    st.divider()

    if st.button("🔐 Run Secure Clinical RAG Pipeline", type="primary"):
        st.session_state.hitl_decision_made = False

        # ---- STEP 1: STRICT MULTI-TENANCY GATE (must run before ANY RAG logic)
        try:
            check_multi_tenant_access(active_doctor_id, active_patient_id, consolidated)
        except MultiTenancyAccessError as denial:
            st.error(
                f"### ⛔ ACCESS DENIED — Multi-Tenant Security Violation\n\n"
                f"**Doctor ID:** `{active_doctor_id}`\n\n"
                f"**Requested Patient ID:** `{active_patient_id}`\n\n"
                f"**Reason:** {denial.reason}\n\n"
                f"No retrieval, embedding, or generation logic was executed for this session."
            )
            audit_logger.log_event(make_audit_event(
                action="ACCESS_DENIED",
                doctor_id=active_doctor_id,
                patient_id=active_patient_id,
                detail={"reason": denial.reason},
            ))
            st.session_state.compiled_context = None
            st.session_state.generated_summary = None
            st.stop()

        audit_logger.log_event(make_audit_event(
            action="ACCESS_GRANTED", doctor_id=active_doctor_id, patient_id=active_patient_id,
        ))
        st.success(f"✅ Access granted — Dr. `{active_doctor_id[:8]}` has a verified encounter history "
                   f"with patient `{active_patient_id[:8]}`.")

        # ---- STEP 2: AGGREGATE CLINICAL BUNDLE
        bundle = build_patient_clinical_bundle(consolidated, active_patient_id)

        # ---- STEP 3: DUAL-RAG COMPILE (dedupe + U-shape)
        guideline_chunks = [guideline_text] if guideline_text.strip() else []
        history_chunks = bundle_to_history_chunks(bundle)
        safety_chunks = bundle_to_safety_chunks(bundle)

        compiled_context = compile_u_shaped_prompt(guideline_chunks, history_chunks, safety_chunks)
        context_hash = sha256_hash(compiled_context)

        st.session_state.compiled_context = compiled_context
        st.session_state.context_hash = context_hash

        # ---- STEP 4: GENERATE
        with st.spinner("Generating clinical summary via Azure OpenAI..."):
            summary = generate_clinical_summary(client, compiled_context)
        st.session_state.generated_summary = summary

        audit_logger.log_event(make_audit_event(
            action="RAG_GENERATED",
            doctor_id=active_doctor_id,
            patient_id=active_patient_id,
            context_hash=context_hash,
            detail={
                "guideline_chunks": len(dedupe_chunks(guideline_chunks)),
                "history_chunks": len(dedupe_chunks(history_chunks)),
                "safety_chunks": len(dedupe_chunks(safety_chunks)),
            },
        ))

    # -------------------------------------------------------- RESULTS + HITL
    if st.session_state.compiled_context:
        with st.expander("🧩 Compiled U-Shaped RAG Context", expanded=False):
            st.code(st.session_state.compiled_context, language="text")
            st.caption(f"SHA-256 context fingerprint: `{st.session_state.context_hash}`")

        st.subheader("📝 Generated Clinical Summary (draft — requires sign-off)")
        st.info(st.session_state.generated_summary)

        st.subheader("✅ Human-in-the-Loop Approval")
        with st.form("hitl_approval_form", clear_on_submit=False):
            decision = st.radio("Clinician decision", ["Approve", "Reject", "Approve with edits"], horizontal=True)
            edited_text = st.text_area("Edited summary (only used if 'Approve with edits')",
                                        value=st.session_state.generated_summary or "", height=150)
            comment = st.text_area("Reviewer comment / rationale", height=80)
            submitted = st.form_submit_button("Submit HITL Decision")

            if submitted:
                action_map = {
                    "Approve": "HITL_APPROVED",
                    "Reject": "HITL_REJECTED",
                    "Approve with edits": "HITL_APPROVED_WITH_EDITS",
                }
                audit_logger.log_event(make_audit_event(
                    action=action_map[decision],
                    doctor_id=active_doctor_id,
                    patient_id=active_patient_id,
                    context_hash=st.session_state.context_hash,
                    detail={
                        "reviewer_comment": comment,
                        "final_text": edited_text if decision == "Approve with edits" else st.session_state.generated_summary,
                    },
                ))
                st.session_state.hitl_decision_made = True
                st.success(f"Decision '{decision}' recorded to the forensic audit trail.")

    # -------------------------------------------------------------- AUDIT UI
    st.divider()
    st.subheader("🧾 Forensic Audit Trail (most recent)")
    recent_events = audit_logger.read_recent(limit=100)
    if recent_events:
        audit_df = pd.DataFrame(recent_events)
        st.dataframe(audit_df, use_container_width=True, height=280)
    else:
        st.caption("No audit events recorded yet in this environment.")


if __name__ == "__main__":
    main()

"""
circuit_breaker_function/__init__.py — PHASE 7: Orchestration & Resilience.

HTTP-triggered Azure Function that is the single entry point the Streamlit
app (via APIM) calls. It wires together every phase in order, and wraps the
three network-dependent calls (AI Search, cross-encoder, Azure OpenAI) in a
circuit breaker so a degraded downstream dependency fails fast with a clear
503 instead of the whole app hanging or cascading retries into an outage.

Deploy as an Azure Functions Python app (Consumption or Premium plan).
"""

from __future__ import annotations

import json
import logging
import os
import sys

import azure.functions as func
import pybreaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # reach the pandas pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))         # reach azure_rag_v2 modules

from audit import make_audit_event  # noqa: E402
from azure_integration import build_azure_openai_client  # noqa: E402
from context_assembler import assemble_u_shaped_context, sha256_hash  # noqa: E402
from data_loader import load_and_consolidate_synthea  # noqa: E402
from generation_guardrail import generate_summary, verify_output_safety  # noqa: E402
from redis_memory import append_turn, get_cached_response, set_cached_response  # noqa: E402
from retrieval_rerank_dedup import (  # noqa: E402
    dedupe_by_parent,
    rerank,
    retrieve_guideline_children,
    retrieve_patient_children,
)
from security import MultiTenancyAccessError, check_multi_tenant_access  # noqa: E402
from servicebus_feedback import send_hitl_feedback  # noqa: E402

logger = logging.getLogger("clinical_rag.orchestrator")

# Three independent breakers — AI Search, cross-encoder, and Azure OpenAI fail
# independently and should trip independently. fail_max/reset_timeout are
# starting points; tune against your actual p99 latency and error budget.
search_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=30, name="ai_search")
rerank_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=30, name="cross_encoder")
generation_breaker = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=60, name="azure_openai")

DATA_DIR = os.environ.get("SYNTHEA_DATA_DIR", "/home/site/wwwroot/synthea_data")


def _embed_query(azure_openai_client, text: str):
    embedding_deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    resp = azure_openai_client.embeddings.create(model=embedding_deployment, input=[text])
    return resp.data[0].embedding


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse(json.dumps({"error": "Invalid JSON body"}), status_code=400,
                                  mimetype="application/json")

    doctor_id = body.get("doctor_id", "")
    patient_id = body.get("patient_id", "")
    question = body.get("question", "Summarize this patient's current clinical status.")
    disease_filter = body.get("disease_filter")  # optional, for guideline retrieval
    session_id = body.get("session_id", f"{doctor_id}:{patient_id}")

    # ---- STEP 1: STRICT MULTI-TENANCY GATE (runs before anything else) -----
    consolidated = load_and_consolidate_synthea(DATA_DIR)
    try:
        check_multi_tenant_access(doctor_id, patient_id, consolidated)
    except MultiTenancyAccessError as denial:
        send_hitl_feedback(make_audit_event("ACCESS_DENIED", doctor_id, patient_id,
                                             detail={"reason": denial.reason}))
        return func.HttpResponse(
            json.dumps({"error": "ACCESS_DENIED", "reason": denial.reason}),
            status_code=403, mimetype="application/json",
        )

    azure_openai_client = build_azure_openai_client()
    if azure_openai_client is None:
        return func.HttpResponse(json.dumps({"error": "Azure OpenAI not configured"}),
                                  status_code=503, mimetype="application/json")

    try:
        query_vector = _embed_query(azure_openai_client, question)

        # ---- STEP 2: RETRIEVE (circuit-broken) ------------------------------
        patient_children = search_breaker.call(
            retrieve_patient_children, query_vector, question, doctor_id, patient_id, 40
        )
        guideline_children = []
        if disease_filter:
            guideline_children = search_breaker.call(
                retrieve_guideline_children, query_vector, question, disease_filter, 40
            )

        # ---- STEP 3: RERANK (circuit-broken) --------------------------------
        patient_top_k = rerank_breaker.call(rerank, question, patient_children, 8)
        guideline_top_k = rerank_breaker.call(rerank, question, guideline_children, 5) if guideline_children else []

        # ---- STEP 4: DEDUP (Redis-backed parent dedup) -----------------------
        patient_deduped = dedupe_by_parent(session_id, patient_top_k)
        guideline_deduped = dedupe_by_parent(session_id, guideline_top_k)

        # ---- STEP 5: ASSEMBLE U-SHAPED CONTEXT ---------------------------------
        compiled_context = assemble_u_shaped_context(guideline_deduped, patient_deduped)
        context_hash = sha256_hash(compiled_context)

        # ---- STEP 6: RESPONSE CACHE CHECK --------------------------------------
        cached = get_cached_response(context_hash)
        if cached:
            summary = cached
            cache_hit = True
        else:
            summary = generation_breaker.call(generate_summary, azure_openai_client, compiled_context)
            set_cached_response(context_hash, summary)
            cache_hit = False

        # ---- STEP 7: DETERMINISTIC SAFETY GUARDRAIL ----------------------------
        safety_result = verify_output_safety(compiled_context, summary)

        # ---- STEP 8: MEMORY + AUDIT ---------------------------------------------
        append_turn(doctor_id, patient_id, "assistant", summary)
        send_hitl_feedback(make_audit_event(
            "RAG_GENERATED", doctor_id, patient_id, context_hash=context_hash,
            detail={"cache_hit": cache_hit, "safety_passed": safety_result.passed,
                    "safety_conflicts": safety_result.conflicts},
        ))

        return func.HttpResponse(
            json.dumps({
                "compiled_context": compiled_context,
                "context_sha256": context_hash,
                "summary": summary,
                "cache_hit": cache_hit,
                "safety_verification": {"passed": safety_result.passed, "conflicts": safety_result.conflicts},
            }),
            status_code=200, mimetype="application/json",
        )

    except pybreaker.CircuitBreakerError as breaker_err:
        logger.error("Circuit open: %s", breaker_err)
        return func.HttpResponse(
            json.dumps({"error": "UPSTREAM_DEGRADED",
                        "detail": "A downstream dependency is failing repeatedly; circuit breaker is open. "
                                  "Retry shortly."}),
            status_code=503, mimetype="application/json",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled orchestrator error")
        return func.HttpResponse(json.dumps({"error": "INTERNAL_ERROR", "detail": str(exc)}),
                                  status_code=500, mimetype="application/json")

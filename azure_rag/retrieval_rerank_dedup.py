"""
retrieval_rerank_dedup.py — PHASES 3, 4 & 5:

  3. Filtered child-chunk retrieval from Azure AI Search
     (patient-index filtered by DoctorID, medical-knowledge-index by Disease)
  4. Cross-encoder reranking of the high-recall child-chunk pull down to top-K
  5. Redis-backed de-duplication of PARENT chunks (a single parent can surface
     via multiple overlapping children — we only want it once in the final
     context, and Redis lets every concurrent request share one dedup cache
     with a short TTL instead of recomputing per-request)
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional

import redis
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

AI_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
AI_SEARCH_API_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY")

REDIS_HOST = os.environ.get("AZURE_REDIS_HOST")
REDIS_PORT = int(os.environ.get("AZURE_REDIS_PORT", "6380"))
REDIS_PASSWORD = os.environ.get("AZURE_REDIS_PASSWORD")
DEDUP_TTL_SECONDS = 600  # short-lived: dedup is per-query-session, not permanent state


@dataclass
class RetrievedChild:
    child_id: str
    parent_id: str
    parent_text: str
    child_text: str
    search_score: float
    rerank_score: Optional[float] = None
    metadata: Optional[dict] = None


def _search_client(index_name: str) -> SearchClient:
    credential = AzureKeyCredential(AI_SEARCH_API_KEY) if AI_SEARCH_API_KEY else DefaultAzureCredential()
    return SearchClient(endpoint=AI_SEARCH_ENDPOINT, index_name=index_name, credential=credential)


def _redis_client() -> Optional["redis.Redis"]:
    if not REDIS_HOST:
        return None
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, ssl=True, decode_responses=True)


# --------------------------------------------------------------------------- #
# PHASE 3 — Filtered retrieval (high recall pull)
# --------------------------------------------------------------------------- #

def retrieve_patient_children(query_vector: List[float], query_text: str, doctor_id: str,
                               patient_id: Optional[str] = None, top_k: int = 40) -> List[RetrievedChild]:
    """
    Multi-tenant filter: `search.in(doctor_ids, '<doctor_id>')` — a child chunk
    is only retrievable if the requesting doctor is in its `doctor_ids`
    collection, which was populated straight from the real Provider ACL at
    ingestion time. This is the AI-Search-layer mirror of security.py's
    check_multi_tenant_access — belt-and-suspenders: the app-layer gate still
    runs too, this just means an unauthorized query can't even surface chunks.
    """
    client = _search_client("patient-index")
    filter_parts = [f"doctor_ids/any(d: d eq '{doctor_id}')"]
    if patient_id:
        filter_parts.append(f"patient_id eq '{patient_id}'")
    odata_filter = " and ".join(filter_parts)

    results = client.search(
        search_text=query_text,
        vector_queries=[VectorizedQuery(vector=query_vector, k_nearest_neighbors=top_k, fields="child_vector")],
        filter=odata_filter,
        top=top_k,
        query_type="semantic",
        semantic_configuration_name="patient-semantic-config",
    )
    return [
        RetrievedChild(
            child_id=r["child_id"], parent_id=r["parent_id"], parent_text=r["parent_text"],
            child_text=r["child_text"], search_score=r["@search.score"],
            metadata={"section": r.get("section"), "event_date": r.get("event_date")},
        )
        for r in results
    ]


def retrieve_guideline_children(query_vector: List[float], query_text: str, disease: str,
                                 top_k: int = 40) -> List[RetrievedChild]:
    client = _search_client("medical-knowledge-index")
    results = client.search(
        search_text=query_text,
        vector_queries=[VectorizedQuery(vector=query_vector, k_nearest_neighbors=top_k, fields="child_vector")],
        filter=f"disease eq '{disease}'",
        top=top_k,
        query_type="semantic",
        semantic_configuration_name="knowledge-semantic-config",
    )
    return [
        RetrievedChild(
            child_id=r["child_id"], parent_id=r["parent_id"], parent_text=r["parent_text"],
            child_text=r["child_text"], search_score=r["@search.score"],
            metadata={"source_name": r.get("source_name")},
        )
        for r in results
    ]


# --------------------------------------------------------------------------- #
# PHASE 4 — Cross-encoder reranking
# --------------------------------------------------------------------------- #

_cross_encoder = None


def _get_cross_encoder():
    """
    Lazy-loaded singleton. Swap the model name for whatever you've approved
    for clinical use / whatever you can self-host in an Azure ML endpoint —
    ms-marco-MiniLM is a reasonable general-purpose default, but for a
    medical-grade deployment consider fine-tuning on clinical query/passage
    pairs, or hosting a domain cross-encoder behind Azure ML instead of
    downloading from the public hub at runtime.
    """
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank(query_text: str, candidates: List[RetrievedChild], top_k: int = 8) -> List[RetrievedChild]:
    """Cross-encoder scores (query, child_text) pairs directly — much higher
    precision than the bi-encoder vector search alone, at the cost of being
    too slow to run over the whole index (hence: vector search for recall,
    cross-encoder for precision, on just the ~40 candidates it returned)."""
    if not candidates:
        return []
    encoder = _get_cross_encoder()
    pairs = [(query_text, c.child_text) for c in candidates]
    scores = encoder.predict(pairs)
    for c, s in zip(candidates, scores):
        c.rerank_score = float(s)
    candidates.sort(key=lambda c: c.rerank_score, reverse=True)
    return candidates[:top_k]


# --------------------------------------------------------------------------- #
# PHASE 5 — Redis-backed parent-chunk de-duplication
# --------------------------------------------------------------------------- #

def _dedup_key(session_id: str, parent_id: str) -> str:
    digest = hashlib.sha256(parent_id.encode("utf-8")).hexdigest()[:16]
    return f"dedup:{session_id}:{digest}"


def dedupe_by_parent(session_id: str, reranked: List[RetrievedChild]) -> List[RetrievedChild]:
    """
    Multiple child chunks from the SAME parent can both make the reranked
    top-K (e.g. two sentences from the same encounter note). We only want the
    parent represented once in the final context. Redis SETNX gives every
    concurrent request in the same session a shared, atomic "have we already
    included this parent" check with a short TTL, instead of every request
    recomputing a local set (which breaks across Function App instances).
    Falls back to an in-process set if Redis isn't configured (e.g. local dev).
    """
    r = _redis_client()
    deduped: List[RetrievedChild] = []

    if r is None:
        seen_local = set()
        for c in reranked:
            if c.parent_id in seen_local:
                continue
            seen_local.add(c.parent_id)
            deduped.append(c)
        return deduped

    for c in reranked:
        key = _dedup_key(session_id, c.parent_id)
        # SET ... NX EX=TTL returns True only the first time this key is set.
        first_seen = r.set(key, "1", nx=True, ex=DEDUP_TTL_SECONDS)
        if first_seen:
            deduped.append(c)
    return deduped

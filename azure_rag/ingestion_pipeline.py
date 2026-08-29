"""
ingestion_pipeline.py — PHASE 2b: embed child chunks and push to Azure AI Search.

Wires chunking.py -> Azure OpenAI embeddings -> Azure AI Search upload, for
both indexes (patient-index and medical-knowledge-index).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

from chunking import ChildChunk, chunk_guideline_document, chunk_patient_bundle

AI_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
AI_SEARCH_API_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")


def _search_client(index_name: str) -> SearchClient:
    credential = AzureKeyCredential(AI_SEARCH_API_KEY) if AI_SEARCH_API_KEY else DefaultAzureCredential()
    return SearchClient(endpoint=AI_SEARCH_ENDPOINT, index_name=index_name, credential=credential)


def embed_texts(azure_openai_client, texts: List[str]) -> List[List[float]]:
    """Batch-embed child chunk texts. Azure OpenAI embeddings endpoint accepts
    up to ~2048 inputs per call; batch defensively at 100 to stay well under
    request-size limits for long clinical text."""
    vectors: List[List[float]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = azure_openai_client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=batch)
        vectors.extend([item.embedding for item in response.data])
    return vectors


def ingest_patient_chunks(azure_openai_client, patient_id: str, doctor_ids: List[str],
                           section_texts: Dict[str, str], event_dates: Optional[Dict[str, str]] = None) -> int:
    """
    section_texts: {"allergies": "...", "medications": "...", ...} — the same
    flattened text your pandas pipeline already produces per section.
    doctor_ids: every Provider ID with a real encounter for this patient
                (straight from data_loader.ConsolidatedSyntheaData.provider_acl)
                — this is what makes the multi-tenant filter work at query time.
    """
    children: List[ChildChunk] = chunk_patient_bundle(patient_id, section_texts)
    if not children:
        return 0

    vectors = embed_texts(azure_openai_client, [c.child_text for c in children])

    docs = []
    for child, vector in zip(children, vectors):
        section = child.metadata.get("section", "")
        docs.append({
            "child_id": child.child_id,
            "parent_id": child.parent_id,
            "parent_text": child.parent_text,
            "child_text": child.child_text,
            "patient_id": patient_id,
            "doctor_ids": doctor_ids,
            "section": section,
            "event_date": (event_dates or {}).get(section),
            "child_vector": vector,
        })

    client = _search_client("patient-index")
    result = client.upload_documents(documents=docs)
    failed = [r for r in result if not r.succeeded]
    if failed:
        raise RuntimeError(f"{len(failed)} patient chunk(s) failed to index: {failed[:3]}")
    return len(docs)


def ingest_guideline_document(azure_openai_client, doc_id: str, disease: str, source_name: str,
                               full_text: str, publication_year: Optional[int] = None) -> int:
    children = chunk_guideline_document(doc_id, disease, source_name, full_text, publication_year)
    if not children:
        return 0

    vectors = embed_texts(azure_openai_client, [c.child_text for c in children])

    docs = []
    for child, vector in zip(children, vectors):
        docs.append({
            "child_id": child.child_id,
            "parent_id": child.parent_id,
            "parent_text": child.parent_text,
            "child_text": child.child_text,
            "disease": disease,
            "source_name": source_name,
            "publication_year": publication_year or 0,
            "child_vector": vector,
        })

    client = _search_client("medical-knowledge-index")
    result = client.upload_documents(documents=docs)
    failed = [r for r in result if not r.succeeded]
    if failed:
        raise RuntimeError(f"{len(failed)} guideline chunk(s) failed to index: {failed[:3]}")
    return len(docs)

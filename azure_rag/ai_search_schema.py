"""
ai_search_schema.py — PHASE 1: Azure AI Search index design.

Creates two indexes matching your diagram:

  1. patient-index   — per-patient clinical chunks, filterable by DoctorID
                        (multi-tenant: query-time filter, not a separate
                        index per tenant — cheaper and matches your ACL model)
  2. medical-knowledge-index — external guideline chunks, filterable by Disease

Both indexes use a PARENT/CHILD chunk model in a single flat index:
  - "child" documents are small (~200-400 token) retrieval units with their
    own vector embedding — these are what similarity search actually matches.
  - each child carries a `parent_id` + the FULL `parent_text` duplicated onto
    it, so that once we've picked our top-K *children*, we can immediately
    recover the *parent* context (the whole note/section) without a second
    round-trip to a document store. This is what powers the "child chunks
    retrieved -> reranked -> deduped by parent -> parent+child reassembled
    into the U-shaped prompt" flow in context_assembler.py.

Run:
    python ai_search_schema.py --create-patient-index
    python ai_search_schema.py --create-knowledge-index
"""

from __future__ import annotations

import argparse
import os

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

AI_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "https://<your-search-service>.search.windows.net")
AI_SEARCH_API_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY")  # omit to use Managed Identity instead
EMBEDDING_DIMENSIONS = 3072  # text-embedding-3-large; use 1536 for text-embedding-3-small


def _get_index_client() -> SearchIndexClient:
    credential = AzureKeyCredential(AI_SEARCH_API_KEY) if AI_SEARCH_API_KEY else DefaultAzureCredential()
    return SearchIndexClient(endpoint=AI_SEARCH_ENDPOINT, credential=credential)


def _vector_search_config() -> VectorSearch:
    return VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                parameters=HnswParameters(m=4, ef_construction=400, ef_search=500, metric="cosine"),
            )
        ],
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-config")],
    )


def build_patient_index() -> SearchIndex:
    """
    Multi-tenant clinical chunk index. `doctor_ids` is a Collection(Edm.String)
    so a single child chunk can be visible to every provider who has a
    recorded encounter for that patient (mirrors the Provider ACL from the
    pandas pipeline) — filtered at query time with
    `search.in(doctor_ids, '<doctor_id>')`, never by a separate per-tenant index.
    """
    fields = [
        SimpleField(name="child_id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="parent_text", type=SearchFieldDataType.String, searchable=False, retrievable=True),
        SearchField(name="child_text", type=SearchFieldDataType.String, searchable=True, retrievable=True),
        SimpleField(name="patient_id", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="doctor_ids",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
        ),
        SimpleField(name="section", type=SearchFieldDataType.String, filterable=True),  # allergy|medication|condition|lab|encounter
        SimpleField(name="event_date", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
        SearchField(
            name="child_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="hnsw-profile",
        ),
    ]
    semantic_config = SemanticConfiguration(
        name="patient-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="child_text")]
        ),
    )
    return SearchIndex(
        name="patient-index",
        fields=fields,
        vector_search=_vector_search_config(),
        semantic_search=SemanticSearch(configurations=[semantic_config]),
    )


def build_knowledge_index() -> SearchIndex:
    """External clinical guideline chunk index, filterable by Disease/topic."""
    fields = [
        SimpleField(name="child_id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="parent_text", type=SearchFieldDataType.String, searchable=False, retrievable=True),
        SearchField(name="child_text", type=SearchFieldDataType.String, searchable=True, retrievable=True),
        SimpleField(name="disease", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="source_name", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="publication_year", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SearchField(
            name="child_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="hnsw-profile",
        ),
    ]
    semantic_config = SemanticConfiguration(
        name="knowledge-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="child_text")]
        ),
    )
    return SearchIndex(
        name="medical-knowledge-index",
        fields=fields,
        vector_search=_vector_search_config(),
        semantic_search=SemanticSearch(configurations=[semantic_config]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-patient-index", action="store_true")
    parser.add_argument("--create-knowledge-index", action="store_true")
    args = parser.parse_args()

    client = _get_index_client()

    if args.create_patient_index:
        client.create_or_update_index(build_patient_index())
        print("patient-index created/updated.")

    if args.create_knowledge_index:
        client.create_or_update_index(build_knowledge_index())
        print("medical-knowledge-index created/updated.")


if __name__ == "__main__":
    main()

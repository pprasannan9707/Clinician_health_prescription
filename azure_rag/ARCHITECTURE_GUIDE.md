# Azure Clinical RAG v2 — Phased Build Guide

This matches your architecture diagram exactly. Fourteen phases, each with
the Azure resources to provision and the code module that implements it.
Do them roughly in this order — later phases depend on earlier ones being
live.

```
Streamlit UI ──► APIM (JWT/RBAC) ──► Azure Function (circuit breaker)
                                            │
                        ┌───────────────────┼────────────────────┐
                        ▼                   ▼                    ▼
                Azure AI Search      Azure Cache for Redis   Azure OpenAI
              (patient / knowledge)   (memory + dedup keys)     (GPT-4o)
                        │
              cross-encoder rerank ──► parent dedup ──► U-shape assembly
                                                              │
                                                    safety guardrail + RAGAS
                                                              │
                                                  HITL sign-off ──► Service Bus
                                                              │
                                                    prescription .docx
```

---

## Phase 0 — Provision core resources

```bash
RG=rg-clinical-rag-v2
LOCATION=eastus2
az group create -n $RG -l $LOCATION

# Azure AI Search
az search service create --name search-clinical-rag --resource-group $RG \
  --sku standard --location $LOCATION --partition-count 1 --replica-count 1

# Azure Cache for Redis (Standard tier — Basic has no SLA, avoid for clinical workloads)
az redis create --name redis-clinical-rag --resource-group $RG --location $LOCATION \
  --sku Standard --vm-size C1

# Azure Service Bus (Standard tier — needed for dead-letter + topics if you expand later)
az servicebus namespace create --name sb-clinical-rag --resource-group $RG \
  --location $LOCATION --sku Standard
az servicebus queue create --namespace-name sb-clinical-rag --resource-group $RG \
  --name hitl-feedback --max-delivery-count 5 --enable-dead-lettering-on-message-expiration true

# Storage (raw Synthea, prescriptions, audit trail — reuse from v1 or create fresh)
STORAGE_ACCOUNT=stclinicalragv2$RANDOM
az storage account create --name $STORAGE_ACCOUNT --resource-group $RG --location $LOCATION \
  --sku Standard_LRS --min-tls-version TLS1_2 --allow-blob-public-access false
for c in synthea-raw audit-trail prescriptions; do
  az storage container create --account-name $STORAGE_ACCOUNT --name $c --auth-mode login
done

# Azure OpenAI (chat + embeddings)
az cognitiveservices account create --name aoai-clinical-rag --resource-group $RG \
  --kind OpenAI --sku S0 --location $LOCATION --custom-domain aoai-clinical-rag
az cognitiveservices account deployment create --name aoai-clinical-rag --resource-group $RG \
  --deployment-name gpt-4o --model-name gpt-4o --model-version "2024-08-06" \
  --model-format OpenAI --sku-capacity 10 --sku-name Standard
az cognitiveservices account deployment create --name aoai-clinical-rag --resource-group $RG \
  --deployment-name text-embedding-3-large --model-name text-embedding-3-large \
  --model-version "1" --model-format OpenAI --sku-capacity 10 --sku-name Standard
```

---

## Phase 1 — Azure AI Search: parent-child index design

Two indexes, matching your diagram's "Patient Index" (filter: DoctorID) and
"Medical Knowledge" (filter: Disease). Both use the parent/child chunk model
described in `ai_search_schema.py`: every retrievable "child" document also
carries the full `parent_text`, so one search hit returns both the precise
match and its full context in a single round trip.

```bash
export AZURE_SEARCH_ENDPOINT=https://search-clinical-rag.search.windows.net
python ai_search_schema.py --create-patient-index --create-knowledge-index
```

**Why parent/child instead of one flat chunk size:** small chunks (child)
maximize retrieval precision (a 300-token slice matches a specific query
far better than a 3000-token note); large chunks (parent) maximize the
*usefulness* of what the LLM actually reads. Storing both and only
embedding the child gets you both without a second fetch.

---

## Phase 2 — Ingestion: parent-child chunking + embedding + push

`chunking.py` implements the greedy sentence-packing chunker (300-token
target, 40-token overlap). `ingestion_pipeline.py` embeds every child via
`text-embedding-3-large` and bulk-uploads to the matching index.

```python
from openai import AzureOpenAI
from ingestion_pipeline import ingest_patient_chunks, ingest_guideline_document

client = AzureOpenAI(azure_endpoint=..., api_key=..., api_version="2024-10-21")

# One call per patient, using the section text your existing pandas pipeline
# (clinical_aggregation.py / rag_compiler.py) already produces:
ingest_patient_chunks(
    client,
    patient_id="p1",
    doctor_ids=list(consolidated.provider_acl["p1"]),   # <- straight from data_loader ACL
    section_texts={
        "allergies": "\n".join(bundle_to_safety_chunks(bundle)),
        "conditions": "\n".join(c for c in history_chunks if c.startswith("Condition:")),
        "medications": "\n".join(c for c in history_chunks if c.startswith("Medication")),
        "organ_dysfunction_labs": "\n".join(c for c in bundle_to_safety_chunks(bundle) if "FLAG" in c),
    },
)

# Guideline documents, once, offline (not per-patient):
ingest_guideline_document(client, doc_id="ada-2026-t2dm", disease="Type 2 Diabetes",
                           source_name="ADA Standards of Care 2026", full_text=guideline_pdf_text,
                           publication_year=2026)
```

Run patient ingestion as a nightly batch job (Azure Function timer trigger
or Data Factory pipeline) reading straight from your existing
`load_and_consolidate_synthea()` output — re-ingest the whole patient
whenever their record changes; `child_id` is a fresh UUID each run so stale
chunks from a prior version need an explicit delete-by-`patient_id` filter
before re-upload if you want to avoid duplicate parents accumulating.

---

## Phase 3 — Retrieval with multi-tenant / disease filters

`retrieval_rerank_dedup.py::retrieve_patient_children()` issues a hybrid
(vector + semantic) query against `patient-index`, filtered by
`doctor_ids/any(d: d eq '<doctor_id>')` — the AI-Search-layer mirror of your
pandas `security.py` gate. `retrieve_guideline_children()` does the same
against `medical-knowledge-index` filtered by `disease eq '<disease>'`.

This is a **second, independent** enforcement layer: even if a bug ever let
an unauthorized request reach the retrieval step, the filter means no
chunks for that patient could physically come back. Keep both layers — the
app-layer `check_multi_tenant_access()` gate always runs *first* and never
gets removed just because the index also filters.

---

## Phase 4 — Cross-encoder reranking

High-recall vector search returns ~40 candidates; `rerank()` scores every
`(query, child_text)` pair with a cross-encoder and keeps the top 8 (patient)
/ top 5 (guideline). This is deliberately a two-stage retrieve-then-rerank
design: cross-encoders are too slow to run over the whole index but far more
precise than the bi-encoder vector search alone on a short candidate list.

For a real clinical deployment, don't ship the public
`ms-marco-MiniLM-L-6-v2` as-is — either fine-tune it on clinical
query/passage pairs, or host a domain-appropriate cross-encoder behind an
**Azure ML managed online endpoint** and swap `_get_cross_encoder()` for an
HTTP call to that endpoint instead of loading weights in-process.

---

## Phase 5 — Redis-backed parent-chunk de-duplication

Multiple reranked children can point at the same parent (two sentences from
one note). `dedupe_by_parent()` uses `SET key NX EX=600` in Redis so every
concurrent Function App instance shares one atomic "have we already
included this parent in this session" check — critical once you scale the
Function App beyond one instance, where an in-process Python `set()` would
silently diverge per instance.

```bash
az redis show --name redis-clinical-rag --resource-group $RG --query hostName -o tsv
# set AZURE_REDIS_HOST / AZURE_REDIS_PASSWORD (from `az redis list-keys`) as Function App settings
```

---

## Phase 6 — U-shaped context assembly

`context_assembler.py::assemble_u_shaped_context()` takes the deduped
guideline + patient children, splits patient parents into
`safety` (allergy / organ-dysfunction sections) vs. everything else, and
lays them out exactly like the pandas-only pipeline: **guidelines top,
general history middle, safety flags pinned bottom** — with a second,
text-level SHA-256 dedup pass in case two different parents produced
near-identical text.

---

## Phase 7 — Orchestration & Resilience (Azure Functions + circuit breaker)

`circuit_breaker_function/` is the single HTTP entry point. It runs phases
1–6 in order, wraps AI Search, the cross-encoder, and Azure OpenAI each in
their own `pybreaker.CircuitBreaker` (independent failure domains — one
degraded dependency shouldn't trip the others), and returns `503
UPSTREAM_DEGRADED` fast instead of hanging when a breaker is open.

```bash
FUNC_APP=func-clinical-rag-orchestrator
az functionapp create --resource-group $RG --consumption-plan-location $LOCATION \
  --runtime python --runtime-version 3.11 --functions-version 4 \
  --name $FUNC_APP --storage-account $STORAGE_ACCOUNT --os-type linux

az functionapp identity assign --name $FUNC_APP --resource-group $RG

az functionapp config appsettings set --name $FUNC_APP --resource-group $RG --settings \
  AZURE_SEARCH_ENDPOINT=https://search-clinical-rag.search.windows.net \
  AZURE_OPENAI_ENDPOINT=https://aoai-clinical-rag.openai.azure.com/ \
  AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o \
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large \
  AZURE_REDIS_HOST=redis-clinical-rag.redis.cache.windows.net \
  AZURE_SERVICEBUS_NAMESPACE=sb-clinical-rag.servicebus.windows.net \
  SYNTHEA_DATA_DIR=/home/site/wwwroot/synthea_data

func azure functionapp publish $FUNC_APP
```

Grant the Function App's managed identity: `Search Index Data Reader` on
the search service, `Cognitive Services OpenAI User` on the OpenAI
resource, `Azure Service Bus Data Sender` on the namespace, and configure
Redis with Entra ID auth enabled (`az redis identity assign`) so no
connection-string secret is needed there either.

---

## Phase 8 — Ongoing memory layer (Redis)

`redis_memory.py` gives you session turns (`append_turn` /
`get_session_turns`, 30-min TTL, last 10 turns) and a response cache
(`get_cached_response` / `set_cached_response`, 15-min TTL, keyed by the
compiled context's SHA-256 — so re-asking the same question against
unchanged clinical data skips a full Azure OpenAI call). Keep the cache TTL
short: clinical data changes, and a stale cached summary is worse than a
cheap regeneration.

---

## Phase 9 — Generation & deterministic safety guardrail

`generation_guardrail.py::generate_summary()` is the actual GPT-4o call.
`verify_output_safety()` runs immediately after, and is intentionally **not
another LLM call** — plain regex against the pinned allergy/CKD flags in
the compiled context, so the guardrail itself can never hallucinate. A
guardrail failure never silently blocks the response (a clinician must
always see it); it attaches a hard warning banner and gets written to the
audit trail via `send_hitl_feedback`.

Extend `verify_output_safety()` with your organization's real
drug-interaction rule set (or call out to a licensed interaction-checking
API) before this goes anywhere near production — the shipped version is a
minimal, auditable starting point, not a substitute for a pharmacology
engine.

---

## Phase 10 — Custom RAGAS evals

`ragas_eval.py` re-implements the four core RAGAS metrics
(faithfulness, context precision, context recall, answer relevance) as
direct Azure OpenAI judge calls with `response_format={"type":
"json_object"}`, rather than depending on the `ragas` package's own
non-Azure OpenAI client wiring.

Run this as a nightly Azure DevOps / GitHub Actions pipeline step over a
held-out eval set — reuse the 300-case clinical fact retrieval suite from
`tests/test_suite.json` as your ground-truth answers:

```python
from ragas_eval import evaluate

result = evaluate(
    azure_openai_client=client,
    question="What allergies does this patient have?",
    answer=generated_summary,
    retrieved_chunks=[c.parent_text for c in patient_deduped],
    ground_truth_answer="Penicillin — moderate severity, rash reaction.",  # from test_suite.json
)
print(result.overall, result.faithfulness, result.context_precision,
      result.context_recall, result.answer_relevance)
```

Alert if `overall` drops below your baseline threshold on any nightly run
— this catches retrieval regressions (bad chunking, index staleness) and
generation regressions (prompt drift) before a clinician sees them.

---

## Phase 11 — Prescription output as .docx

`prescription_docx.py` builds the signed prescription with `python-docx` —
runs only *after* HITL approval, never before. The Function App (or a
follow-up Function triggered off the Service Bus `hitl-feedback` queue once
a message has `action == "HITL_APPROVED..."`) generates the file, uploads
it to the `prescriptions` Blob container, and returns a short-lived
user-delegation SAS URL to the Streamlit UI:

```python
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
import datetime

sas = generate_blob_sas(
    account_name=STORAGE_ACCOUNT, container_name="prescriptions", blob_name=blob_name,
    user_delegation_key=udk, permission=BlobSasPermissions(read=True),
    expiry=datetime.datetime.utcnow() + datetime.timedelta(minutes=15),
)
```

---

## Phase 12 — HITL interface + async feedback queue

`streamlit_app_v2.py` renders the approval form; on submit it calls
`servicebus_feedback.send_hitl_feedback()` rather than writing to storage
directly. `receive_and_process_feedback()` is meant to run as a **Service
Bus-triggered Azure Function** (separate from the orchestrator — decouples
"handle this HTTP request" from "process this feedback event"), fanning out
to: the audit-trail writer, the prescription generator, and a RAGAS
eval sampler that occasionally re-scores approved/rejected pairs.

```bash
az servicebus queue create --namespace-name sb-clinical-rag --resource-group $RG \
  --name hitl-feedback-dlq-monitor  # optional: alert on DLQ depth > 0 via Azure Monitor
```

---

## Phase 13 — Azure API Management (JWT / RBAC gateway)

`apim_policy.xml` validates the clinician's Entra ID JWT
(`validate-jwt`), requires a `ClinicalRAG.Provider` or `ClinicalRAG.Admin`
app role claim, force-overwrites `doctor_id` in the request body from a
server-side Entra-ObjectId → Provider-Id mapping (so a clinician can never
spoof a different `doctor_id` from the browser), rate-limits per token, and
injects the Function host key server-side so it's never in the Streamlit
client.

```bash
az apim create --name apim-clinical-rag --resource-group $RG --location $LOCATION \
  --publisher-name "Your Org" --publisher-email you@example.com --sku-name Developer

az apim api create --resource-group $RG --service-name apim-clinical-rag \
  --api-id clinical-rag --path clinical-rag --display-name "Clinical RAG API" \
  --service-url https://$FUNC_APP.azurewebsites.net/api

az apim api policy create --resource-group $RG --service-name apim-clinical-rag \
  --api-id clinical-rag --policy-format xml --value @apim_policy.xml
```

Register an Entra ID app registration for the API (`api://clinical-rag-copilot`),
expose the `ClinicalRAG.Provider` / `ClinicalRAG.Admin` app roles on it, and
assign them to your clinician user accounts / groups.

---

## Phase 14 — Deploy the Streamlit front end

```bash
az acr build --registry $ACR_NAME --image clinical-rag-ui-v2:latest .
az containerapp create --name clinical-rag-ui-v2 --resource-group $RG \
  --environment $APP_ENV --image "$ACR_NAME.azurecr.io/clinical-rag-ui-v2:latest" \
  --target-port 8501 --ingress external --registry-server "$ACR_NAME.azurecr.io" \
  --env-vars APIM_BASE_URL=https://apim-clinical-rag.azure-api.net/clinical-rag \
             APIM_SUBSCRIPTION_KEY=secretref:apim-key
```

Use Azure AD auth on the Container App itself (Easy Auth / `az containerapp auth`)
so the "Bearer token" field in the sidebar is actually populated by a real
sign-in flow rather than pasted manually — manual pasting is fine for a demo,
not for anything touching real clinical data.

---

## What changed vs. the earlier pandas-only prototype

| | v1 (pandas-only) | v2 (this guide) |
|---|---|---|
| Vector store | none — direct pandas filtering | Azure AI Search, 2 indexes |
| Chunking | flat per-section strings | parent/child, embedded children |
| Re-ranking | none | cross-encoder top-K |
| Dedup | SHA-256 on final text | Redis parent-id dedup + SHA-256 text dedup |
| Multi-tenancy | app-layer ACL check only | app-layer check **+** AI Search filter |
| Orchestration | in-process Streamlit | Azure Function, circuit-breaker isolated |
| Memory | none | Redis session + response cache |
| Evals | none | custom RAGAS (4 metrics) |
| Gateway | none | APIM JWT/RBAC |
| Feedback | synchronous audit write | Service Bus async queue + DLQ |
| Prescription | n/a | signed .docx via python-docx |

The v1 modules (`data_loader.py`, `clinical_aggregation.py`, `security.py`,
`rag_compiler.py`) aren't thrown away — `security.py`'s
`check_multi_tenant_access` still runs as the first, authoritative gate in
the Function orchestrator, and `data_loader.py`'s `provider_acl` is exactly
what feeds `doctor_ids` at ingestion time in Phase 2.

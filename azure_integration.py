"""
azure_integration.py — Azure OpenAI (chat generation) and Azure Blob Storage
(raw Synthea pull + forensic audit mirror) integration.

Prefers Managed Identity / Azure AD auth over API keys wherever possible.
Every function degrades gracefully (returns None / a mock response) if Azure
isn't configured, so local dev and tests never hard-fail on missing cloud
credentials.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from config import CLINICAL_SYSTEM_PROMPT, get_secret

logger = logging.getLogger("clinical_rag.azure_integration")


def build_azure_openai_client():
    """
    Build an Azure OpenAI client. Prefers Azure AD / Managed Identity
    (recommended for medical-grade deployments — no long-lived API key to
    leak) and falls back to an API key if one is configured. Returns None
    (mock mode) if no endpoint is configured at all.
    """
    endpoint = get_secret("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        return None

    api_version = get_secret("AZURE_OPENAI_API_VERSION", "2024-10-21")
    api_key = get_secret("AZURE_OPENAI_API_KEY")

    try:
        from openai import AzureOpenAI
    except ImportError:
        logger.error("openai package not installed — running in mock mode.")
        return None

    try:
        if api_key:
            return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

        # Keyless, Managed-Identity / AAD path — preferred for production.
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        return AzureOpenAI(azure_endpoint=endpoint, azure_ad_token_provider=token_provider,
                            api_version=api_version)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to construct Azure OpenAI client, falling back to mock mode: %s", exc)
        return None


def generate_clinical_summary(client, compiled_prompt: str) -> str:
    deployment = get_secret("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

    if client is None:
        return (
            "[MOCK MODE — no AZURE_OPENAI_ENDPOINT configured]\n\n"
            "The U-shaped RAG context below was compiled successfully and is ready "
            "for clinician review. Configure Azure OpenAI environment variables to "
            "enable live model generation."
        )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
                {"role": "user", "content": compiled_prompt},
            ],
            temperature=0.1,
            max_tokens=900,
        )
        return response.choices[0].message.content or "(empty response from model)"
    except Exception as exc:  # noqa: BLE001
        logger.error("Azure OpenAI generation failed: %s", exc)
        return f"[Azure OpenAI generation error — see server logs] {exc}"


def build_blob_container_client(account_url: Optional[str], container_name: Optional[str]):
    """Build an Azure Blob container client via Managed Identity / AAD.
    Returns None if not configured or the SDK isn't installed."""
    if not account_url or not container_name:
        return None
    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        logger.warning("azure-storage-blob / azure-identity not installed — Blob features disabled.")
        return None
    try:
        credential = DefaultAzureCredential()
        service_client = BlobServiceClient(account_url=account_url, credential=credential)
        container_client = service_client.get_container_client(container_name)
        if not container_client.exists():
            container_client.create_container()
        return container_client
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not build Blob container client: %s", exc)
        return None


def download_synthea_from_blob(account_url: str, container_name: str, local_dir: str) -> List[str]:
    """Pull all *.csv blobs from the configured container down to `local_dir`
    so `load_and_consolidate_synthea` can read them exactly as if they were
    a local raw Synthea export."""
    container_client = build_blob_container_client(account_url, container_name)
    if container_client is None:
        raise RuntimeError("Could not connect to Azure Blob Storage with the given configuration.")

    os.makedirs(local_dir, exist_ok=True)
    downloaded: List[str] = []
    for blob in container_client.list_blobs():
        if not blob.name.lower().endswith(".csv"):
            continue
        dest_path = os.path.join(local_dir, os.path.basename(blob.name))
        with open(dest_path, "wb") as f:
            f.write(container_client.download_blob(blob.name).readall())
        downloaded.append(dest_path)
    return downloaded

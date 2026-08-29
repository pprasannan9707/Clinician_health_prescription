# Azure Deployment Guide — Clinical RAG Copilot

This deploys the app as a **keyless, Managed-Identity-authenticated** service:
no Azure OpenAI API key or Storage key ever needs to sit in an app setting or
`.env` file. Everything below uses `az cli`. Adjust names/regions/SKUs as needed.

> **Note on data**: Synthea output is synthetic. It is still recommended you
> treat this deployment as if it processed real PHI (private VNet, Key Vault,
> Microsoft Defender for Cloud, audit retention policy) so the architecture is
> directly promotable to a real clinical pilot without redesign.

## 0. Variables

```bash
RG=rg-clinical-rag
LOCATION=eastus2
ACR_NAME=acrclinicalrag$RANDOM
APP_ENV=cae-clinical-rag
APP_NAME=clinical-rag-app
STORAGE_ACCOUNT=stclinicalrag$RANDOM
OPENAI_RESOURCE=aoai-clinical-rag
OPENAI_DEPLOYMENT=gpt-4o
KEYVAULT_NAME=kv-clinical-rag-$RANDOM

az group create -n $RG -l $LOCATION
```

## 1. Azure OpenAI resource + model deployment

```bash
az cognitiveservices account create \
  --name $OPENAI_RESOURCE \
  --resource-group $RG \
  --kind OpenAI \
  --sku S0 \
  --location $LOCATION \
  --custom-domain $OPENAI_RESOURCE

az cognitiveservices account deployment create \
  --name $OPENAI_RESOURCE \
  --resource-group $RG \
  --deployment-name $OPENAI_DEPLOYMENT \
  --model-name gpt-4o \
  --model-version "2024-08-06" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name Standard

OPENAI_ENDPOINT=$(az cognitiveservices account show \
  --name $OPENAI_RESOURCE --resource-group $RG --query properties.endpoint -o tsv)
```

## 2. Storage account for raw Synthea CSVs + forensic audit trail

```bash
az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RG \
  --location $LOCATION \
  --sku Standard_LRS \
  --min-tls-version TLS1_2 \
  --allow-blob-public-access false

az storage container create --account-name $STORAGE_ACCOUNT --name synthea-raw --auth-mode login
az storage container create --account-name $STORAGE_ACCOUNT --name audit-trail --auth-mode login

# Upload your raw Synthea export
az storage blob upload-batch \
  --account-name $STORAGE_ACCOUNT \
  --destination synthea-raw \
  --source ./output/csv \
  --auth-mode login
```

Enable immutability / WORM policy on `audit-trail` for compliance-grade
tamper evidence (optional but recommended):

```bash
az storage container immutability-policy create \
  --account-name $STORAGE_ACCOUNT \
  --container-name audit-trail \
  --period 365
```

## 3. Container Registry — build & push the image

```bash
az acr create --resource-group $RG --name $ACR_NAME --sku Basic
az acr build --registry $ACR_NAME --image clinical-rag-app:latest .
```

## 4. Azure Container Apps environment + app, with a system-assigned Managed Identity

```bash
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App

az containerapp env create \
  --name $APP_ENV \
  --resource-group $RG \
  --location $LOCATION

az containerapp create \
  --name $APP_NAME \
  --resource-group $RG \
  --environment $APP_ENV \
  --image "$ACR_NAME.azurecr.io/clinical-rag-app:latest" \
  --target-port 8501 \
  --ingress external \
  --registry-server "$ACR_NAME.azurecr.io" \
  --system-assigned \
  --min-replicas 1 --max-replicas 3 \
  --cpu 1.0 --memory 2.0Gi \
  --env-vars \
      AZURE_OPENAI_ENDPOINT="$OPENAI_ENDPOINT" \
      AZURE_OPENAI_CHAT_DEPLOYMENT="$OPENAI_DEPLOYMENT" \
      AZURE_OPENAI_API_VERSION="2024-10-21" \
      AZURE_STORAGE_ACCOUNT_URL="https://$STORAGE_ACCOUNT.blob.core.windows.net" \
      AZURE_SYNTHEA_CONTAINER="synthea-raw" \
      AZURE_AUDIT_CONTAINER="audit-trail"
```

Note: no `AZURE_OPENAI_API_KEY` is set — the app falls through to
`DefaultAzureCredential`, which resolves to the Container App's managed
identity automatically.

## 5. Grant the Managed Identity least-privilege RBAC (no keys, ever)

```bash
PRINCIPAL_ID=$(az containerapp show -n $APP_NAME -g $RG --query identity.principalId -o tsv)
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

# Azure OpenAI — "Cognitive Services OpenAI User"
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Cognitive Services OpenAI User" \
  --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RG/providers/Microsoft.CognitiveServices/accounts/$OPENAI_RESOURCE"

# Blob Storage — "Storage Blob Data Contributor" (read Synthea CSVs, append audit blobs)
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Storage Blob Data Contributor" \
  --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RG/providers/Microsoft.Storage/storageAccounts/$STORAGE_ACCOUNT"
```

## 6. (Optional) Key Vault for any remaining secrets

If you must use API-key auth instead of Managed Identity (e.g. cross-tenant
Azure OpenAI access), store the key in Key Vault and grant the Container
App's identity `Key Vault Secrets User`, then reference it as a Container
Apps **secret** rather than a plaintext env var:

```bash
az keyvault create --name $KEYVAULT_NAME --resource-group $RG --location $LOCATION --enable-rbac-authorization true
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Key Vault Secrets User" \
  --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RG/providers/Microsoft.KeyVault/vaults/$KEYVAULT_NAME"

az keyvault secret set --vault-name $KEYVAULT_NAME --name AzureOpenAIApiKey --value "<key>"

az containerapp secret set -n $APP_NAME -g $RG \
  --secrets azure-openai-api-key=keyvaultref:https://$KEYVAULT_NAME.vault.azure.net/secrets/AzureOpenAIApiKey,identityref:system

az containerapp update -n $APP_NAME -g $RG \
  --set-env-vars AZURE_OPENAI_API_KEY=secretref:azure-openai-api-key
```

## 7. Network hardening (recommended for a real clinical pilot)

- Put the Container Apps environment on a private VNet (`--infrastructure-subnet-resource-id`)
  and disable public ingress on the Storage account (`--default-action Deny` +
  private endpoint).
- Enable Microsoft Defender for Cloud on the resource group.
- Turn on diagnostic settings on the Container App, Storage account, and
  Azure OpenAI resource, streaming to a Log Analytics workspace, in addition
  to the application-level forensic audit trail already written by the app.
- Set a data-retention / legal-hold policy on the `audit-trail` container to
  match your compliance requirement (e.g. HIPAA 6-year minimum if this is
  ever pointed at real PHI instead of synthetic Synthea data).

## 8. Verify

```bash
az containerapp show -n $APP_NAME -g $RG --query properties.configuration.ingress.fqdn -o tsv
```

Open the returned URL — you should see the Streamlit UI, with the sidebar
already reporting "Azure OpenAI client ready" (no `.env` file, no key ever
touched disk).

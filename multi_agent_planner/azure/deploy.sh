#!/bin/bash
set -euo pipefail
RESOURCE_GROUP="multi-agent-planner-rg"
LOCATION="eastus"
ENVIRONMENT="prod"

echo "==> Azure login"
az login

echo "==> Creating resource group"
az group create --name $RESOURCE_GROUP --location $LOCATION

echo "==> Deploying infrastructure (Bicep)"
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --template-file azure/main.bicep \
  --parameters environment=$ENVIRONMENT \
  --query "properties.outputs" -o json

ACR=$(az deployment group show -g $RESOURCE_GROUP -n main \
  --query "properties.outputs.acrServer.value" -o tsv)

echo "==> Building API image → $ACR"
az acr build --registry $ACR --image multi-agent-planner:latest -f Dockerfile .

echo "==> Building Streamlit image → $ACR"
az acr build --registry $ACR --image multi-agent-planner-ui:latest -f streamlit_app/Dockerfile .

echo "==> Done. URLs:"
az deployment group show -g $RESOURCE_GROUP -n main --query "properties.outputs" -o table

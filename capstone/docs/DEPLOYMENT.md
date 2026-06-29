# Deployment Guide

## Prerequisites
- Azure subscription with AKS cluster
- Azure Container Registry (ACR)
- kubectl + helm installed

## Steps

### 1. Build and push images
```bash
make build
docker tag day-planner-api:latest <ACR>.azurecr.io/day-planner-api:latest
docker push <ACR>.azurecr.io/day-planner-api:latest
```

### 2. Create K8s secrets
```bash
kubectl create secret generic day-planner-secrets \
  --from-env-file=.env \
  --namespace day-planner
```

### 3. Deploy with Helm
```bash
make helm-install
```

### 4. Verify
```bash
kubectl get pods -n day-planner
kubectl logs -f deployment/day-planner-api -n day-planner
```

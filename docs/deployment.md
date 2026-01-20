# Deployment Guide

## Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Kubernetes 1.24+ (for production deployment)
- Python 3.10+ (for local development)
- PostgreSQL 14+ with TimescaleDB extension
- Redis 6.2+
- Prometheus 2.40+

## Local Development

### 1. Clone the Repository

```bash
git clone https://github.com/ChitrangiBhatnagar/agentic-observability-platform.git
cd agentic-observability-platform
```

### 2. Install Dependencies

Using Poetry (recommended):

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

Using pip:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

Create `.env` file:

```bash
# Environment
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=postgresql://observability:observability@localhost:5432/observability
REDIS_URL=redis://localhost:6379/0

# Prometheus
PROMETHEUS_URL=http://localhost:9090

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# ML Settings
ANOMALY_THRESHOLD=0.7
ENSEMBLE_STRATEGY=adaptive
AUTO_RETRAIN=true

# Agent Settings
CORRELATION_WINDOW=300
AUTO_REMEDIATE=false
```

### 4. Run Database Migrations

```bash
# Start PostgreSQL with TimescaleDB
docker run -d \
  --name timescaledb \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=observability \
  timescale/timescaledb:latest-pg14

# Initialize schema
psql -h localhost -U postgres -f docker/init-db.sql
```

### 5. Start the Application

```bash
# Run API server
python main.py

# Or with uvicorn directly
uvicorn src.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000
```

### 6. Run Demo Data Generator

In a separate terminal:

```bash
python scripts/generate_demo_data.py --interval 15
```

### 7. Access the Application

- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/health

## Docker Deployment

### Single Container

```bash
# Build image
docker build -t agentic-observability:latest -f docker/Dockerfile .

# Run container
docker run -d \
  --name observability-platform \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e REDIS_URL=redis://host:6379/0 \
  agentic-observability:latest
```

### Docker Compose (Full Stack)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down

# Clean up volumes
docker-compose down -v
```

Services available:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- PostgreSQL: localhost:5432

### Docker Compose with Demo Data

```bash
# Start with demo generator
docker-compose --profile demo up -d

# View demo generator logs
docker-compose logs -f demo-generator
```

## Kubernetes Deployment

### 1. Prerequisites

```bash
# Verify cluster access
kubectl cluster-info

# Create namespace
kubectl create namespace observability
```

### 2. Deploy Dependencies

#### PostgreSQL (TimescaleDB)

```bash
# Using Helm
helm repo add timescale https://charts.timescale.com
helm install timescaledb timescale/timescaledb-single \
  --namespace observability \
  --set persistentVolumes.data.size=20Gi \
  --set persistentVolumes.wal.size=5Gi
```

#### Redis

```bash
# Using Helm
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install redis bitnami/redis \
  --namespace observability \
  --set auth.enabled=false \
  --set master.persistence.size=5Gi
```

#### Prometheus

```bash
# Using Helm
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/prometheus \
  --namespace observability \
  --set server.persistentVolume.size=20Gi
```

### 3. Configure Secrets

```bash
# Create database credentials
kubectl create secret generic observability-secrets \
  --namespace observability \
  --from-literal=database-url='postgresql://user:pass@timescaledb:5432/observability' \
  --from-literal=redis-url='redis://redis-master:6379/0'

# Or apply from kubernetes/configmap.yaml after editing
kubectl apply -f kubernetes/configmap.yaml
```

### 4. Deploy Application

```bash
# Apply all manifests
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/ingress.yaml

# Verify deployment
kubectl get all -n observability

# Check pods
kubectl get pods -n observability -w

# View logs
kubectl logs -f deployment/observability-platform -n observability
```

### 5. Configure Ingress

Update ingress hostname in `kubernetes/ingress.yaml`:

```yaml
spec:
  rules:
    - host: observability.yourdomain.com  # Change this
```

Then apply:

```bash
kubectl apply -f kubernetes/ingress.yaml

# Install cert-manager for TLS (if not already installed)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### 6. Scaling

#### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment observability-platform \
  --namespace observability \
  --replicas=5
```

#### Horizontal Pod Autoscaler (HPA)

HPA is configured in `kubernetes/deployment.yaml`:

```yaml
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

View HPA status:

```bash
kubectl get hpa -n observability
kubectl describe hpa observability-platform-hpa -n observability
```

### 7. Monitoring

```bash
# Port-forward Prometheus
kubectl port-forward -n observability svc/prometheus-server 9090:80

# Port-forward Grafana (if installed)
kubectl port-forward -n observability svc/grafana 3000:80

# View metrics
curl http://localhost:9090/api/v1/query?query=up
```

### 8. Troubleshooting

```bash
# Check pod status
kubectl describe pod <pod-name> -n observability

# View logs
kubectl logs -f <pod-name> -n observability

# Execute commands in pod
kubectl exec -it <pod-name> -n observability -- /bin/bash

# Check events
kubectl get events -n observability --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n observability
kubectl top nodes
```

## Production Considerations

### 1. High Availability

- **Multiple replicas**: Minimum 3 for API service
- **Pod Disruption Budget**: Ensure at least 1 replica always available
- **Anti-affinity rules**: Spread pods across nodes
- **Database replication**: TimescaleDB with streaming replication
- **Redis Sentinel**: For Redis high availability

### 2. Security

```bash
# Network Policies
kubectl apply -f kubernetes/ingress.yaml  # Includes NetworkPolicy

# RBAC
kubectl create serviceaccount observability-sa -n observability
kubectl create rolebinding observability-rb \
  --clusterrole=view \
  --serviceaccount=observability:observability-sa \
  -n observability

# Pod Security Standards
kubectl label namespace observability \
  pod-security.kubernetes.io/enforce=restricted
```

### 3. Resource Limits

Update in `kubernetes/deployment.yaml`:

```yaml
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"
```

### 4. Persistent Storage

```yaml
# For model storage
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: fast-ssd  # Use appropriate storage class
```

### 5. Backup Strategy

```bash
# Database backups
kubectl exec -it timescaledb-0 -n observability -- \
  pg_dump -U postgres observability > backup-$(date +%Y%m%d).sql

# Redis snapshots
kubectl exec -it redis-master-0 -n observability -- \
  redis-cli BGSAVE

# Kubernetes manifests
kubectl get all -n observability -o yaml > k8s-backup.yaml
```

### 6. Monitoring & Alerting

Configure Prometheus alerts:

```yaml
# prometheus-rules.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-alerts
data:
  alerts.yml: |
    groups:
      - name: observability
        rules:
          - alert: HighErrorRate
            expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "High error rate detected"
```

### 7. Performance Tuning

#### Database

```sql
-- Adjust TimescaleDB settings
ALTER DATABASE observability SET timescaledb.max_background_workers = 8;
ALTER DATABASE observability SET shared_preload_libraries = 'timescaledb';

-- Create appropriate indexes
CREATE INDEX CONCURRENTLY ON metrics (timestamp DESC, metric_name);
CREATE INDEX CONCURRENTLY ON anomalies (severity, timestamp DESC);
```

#### Application

```bash
# Environment variables for performance
API_WORKERS=8  # CPU cores * 2
WORKER_CONNECTIONS=1000
KEEPALIVE_TIMEOUT=5
```

### 8. Upgrading

```bash
# Rolling update
kubectl set image deployment/observability-platform \
  app=agentic-observability:v1.1.0 \
  -n observability

# Check rollout status
kubectl rollout status deployment/observability-platform -n observability

# Rollback if needed
kubectl rollout undo deployment/observability-platform -n observability
```

## Health Checks

### Liveness Probe

```bash
curl http://localhost:8000/api/v1/health/live
```

Expected response:
```json
{
  "status": "alive",
  "timestamp": "2024-01-19T10:30:00Z"
}
```

### Readiness Probe

```bash
curl http://localhost:8000/api/v1/health/ready
```

Expected response:
```json
{
  "status": "ready",
  "checks": {
    "database": "ok",
    "redis": "ok",
    "agents": "ok"
  }
}
```

## Metrics

View platform metrics:

```bash
curl http://localhost:8000/metrics
```

Available metrics:
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency
- `anomalies_detected_total`: Total anomalies detected
- `model_inference_duration_seconds`: Model inference time
- `agent_messages_processed_total`: Agent message count

## Support & Troubleshooting

Common issues:

1. **Database connection errors**: Check DATABASE_URL and TimescaleDB status
2. **High memory usage**: Reduce model count or adjust batch sizes
3. **Slow detection**: Increase API workers or optimize model parameters
4. **Agent not responding**: Check agent health endpoint

For more help, check:
- Logs: `kubectl logs -f deployment/observability-platform`
- Health: http://localhost:8000/api/v1/health/agents
- Metrics: http://localhost:8000/metrics

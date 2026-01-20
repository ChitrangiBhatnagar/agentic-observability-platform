# Quick Start Guide

This guide will help you get the Agentic AI-Driven Observability Platform up and running quickly.

## Option 1: Docker Compose (Recommended for Testing)

The fastest way to try the platform is using Docker Compose:

```bash
# Clone the repository
git clone https://github.com/ChitrangiBhatnagar/agentic-observability-platform.git
cd agentic-observability-platform

# Start the entire stack
docker-compose up -d

# Wait for services to be ready (about 30 seconds)
docker-compose ps

# View logs
docker-compose logs -f app
```

Access the services:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Start Demo Data Generator

```bash
# Start with demo data generator
docker-compose --profile demo up -d

# View demo generator logs
docker-compose logs -f demo-generator
```

### Test the API

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Ingest a metric
curl -X POST http://localhost:8000/api/v1/anomalies/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "metrics": [{
      "metric_name": "cpu_usage",
      "value": 95.5,
      "timestamp": "2024-01-19T10:00:00Z",
      "labels": {"service": "api", "env": "prod"}
    }]
  }'

# Get recent anomalies
curl http://localhost:8000/api/v1/anomalies/recent?limit=10
```

## Option 2: Local Development

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ with TimescaleDB
- Redis 6.2+
- Prometheus 2.40+

### Setup Steps

1. **Clone and Install**

```bash
git clone https://github.com/ChitrangiBhatnagar/agentic-observability-platform.git
cd agentic-observability-platform

# Install dependencies
pip install -r requirements.txt

# Or for development
pip install -r requirements.txt -r requirements-dev.txt
```

2. **Configure Environment**

Create `.env` file:

```bash
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://observability:observability@localhost:5432/observability
REDIS_URL=redis://localhost:6379/0
PROMETHEUS_URL=http://localhost:9090
API_HOST=0.0.0.0
API_PORT=8000
```

3. **Initialize Database**

```bash
# Start PostgreSQL with TimescaleDB
docker run -d \
  --name timescaledb \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=observability \
  -e POSTGRES_USER=observability \
  -e POSTGRES_DB=observability \
  timescale/timescaledb:latest-pg14

# Initialize schema
psql -h localhost -U observability -d observability -f docker/init-db.sql
```

4. **Start Services**

```bash
# Terminal 1: Start API
python main.py

# Terminal 2: Start demo generator (optional)
python scripts/generate_demo_data.py --interval 15
```

5. **Access the Application**

- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health

## Option 3: Kubernetes (Production)

### Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3+ (for dependencies)

### Quick Deploy

```bash
# Create namespace
kubectl create namespace observability

# Deploy dependencies (PostgreSQL, Redis, Prometheus)
helm install timescaledb timescale/timescaledb-single -n observability
helm install redis bitnami/redis -n observability
helm install prometheus prometheus-community/prometheus -n observability

# Configure secrets
kubectl create secret generic observability-secrets \
  --namespace observability \
  --from-literal=database-url='postgresql://user:pass@timescaledb:5432/observability' \
  --from-literal=redis-url='redis://redis-master:6379/0'

# Deploy application
kubectl apply -f kubernetes/

# Check status
kubectl get all -n observability

# View logs
kubectl logs -f deployment/observability-platform -n observability
```

### Access via Port Forward

```bash
# API
kubectl port-forward -n observability svc/observability-platform 8000:8000

# Prometheus
kubectl port-forward -n observability svc/prometheus-server 9090:80

# Grafana
kubectl port-forward -n observability svc/grafana 3000:80
```

## Testing the Platform

### 1. Basic Anomaly Detection

```python
import httpx
import asyncio

async def test_detection():
    async with httpx.AsyncClient() as client:
        # Detect anomalies in time series
        response = await client.post(
            "http://localhost:8000/api/v1/anomalies/detect",
            json={
                "metric_name": "cpu_usage",
                "data_points": [
                    {"timestamp": "2024-01-19T10:00:00Z", "value": 45.2},
                    {"timestamp": "2024-01-19T10:01:00Z", "value": 95.5},  # Spike
                    {"timestamp": "2024-01-19T10:02:00Z", "value": 46.1},
                ],
                "labels": {"service": "api", "env": "prod"}
            }
        )
        print(response.json())

asyncio.run(test_detection())
```

### 2. View Active Incidents

```bash
curl http://localhost:8000/api/v1/incidents/active
```

### 3. Submit Feedback

```bash
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "anomaly_id": "anomaly-123",
    "feedback_type": "true_positive",
    "comment": "Confirmed CPU spike during deployment"
  }'
```

### 4. Get Recommendations

```bash
curl http://localhost:8000/api/v1/incidents/{incident_id}/recommendations
```

## Using the Demo Notebook

Open the demo notebook to explore ML models interactively:

```bash
# Install Jupyter
pip install jupyter matplotlib seaborn

# Start Jupyter
jupyter notebook notebooks/demo.md
```

The notebook demonstrates:
- Time series generation
- Multiple ML models
- Feature engineering
- Agent system
- Explainability

## Monitoring the Platform

### Application Metrics

```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# Agent health
curl http://localhost:8000/api/v1/health/agents

# Model performance
curl http://localhost:8000/api/v1/feedback/model-performance
```

### Grafana Dashboards

1. Access Grafana: http://localhost:3000
2. Login: admin/admin
3. Navigate to Dashboards
4. Explore pre-configured dashboards

## Common Issues

### Database Connection Error

```bash
# Check database is running
docker ps | grep timescaledb

# Test connection
psql -h localhost -U observability -d observability -c "SELECT 1"
```

### Redis Connection Error

```bash
# Check Redis is running
docker ps | grep redis

# Test connection
redis-cli -h localhost ping
```

### High Memory Usage

Reduce model count or batch sizes in `.env`:

```bash
ENSEMBLE_MODELS=zscore,stl_esd,isolation_forest  # Instead of all models
BATCH_SIZE=32  # Reduce from default 64
```

## Next Steps

1. **Configure Alerts**: Set up Prometheus alerts and Grafana notifications
2. **Integrate Services**: Connect your Prometheus instance
3. **Customize Models**: Adjust model parameters in configuration
4. **Add Service Topology**: Define service dependencies for better root cause analysis
5. **Set Up CI/CD**: Use the included Kubernetes manifests

## Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: https://github.com/ChitrangiBhatnagar/agentic-observability-platform/issues
- **API Reference**: http://localhost:8000/docs

## Clean Up

### Docker Compose

```bash
# Stop services
docker-compose down

# Remove volumes
docker-compose down -v
```

### Kubernetes

```bash
# Delete deployment
kubectl delete -f kubernetes/

# Delete namespace
kubectl delete namespace observability
```

### Local Development

```bash
# Stop services
# Ctrl+C in terminal windows

# Remove database
docker stop timescaledb
docker rm timescaledb
```

---

**Congratulations!** You now have the Agentic AI-Driven Observability Platform running. Check out the full documentation in the `docs/` directory for advanced configuration and deployment options.

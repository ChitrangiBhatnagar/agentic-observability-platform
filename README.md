# Agentic AI-Driven Observability & Anomaly Detection Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, AI-driven observability platform featuring a **multi-agent architecture** for intelligent anomaly detection, root cause analysis, and automated remediation recommendations.

## 🌟 Key Features

### Multi-Agent Intelligence
- **Detection Agent**: Contextual model selection based on metric characteristics
- **Correlation Agent**: Links anomalies across services using topology-aware algorithms
- **Root Cause Agent**: Causal inference for ranking probable causes
- **Recommendation Agent**: Risk-aware remediation suggestions from playbooks
- **Feedback Agent**: Continuous learning from operator feedback

### Advanced ML Models
- **Statistical**: Z-Score, STL+ESD for seasonal decomposition
- **Isolation Forest**: Unsupervised outlier detection
- **One-Class SVM**: Boundary-based anomaly detection
- **Deep Learning**: LSTM and Transformer Autoencoders
- **Adaptive Ensemble**: Weighted voting with feedback adaptation

### Explainability
- SHAP-based feature importance
- Natural language explanations
- Incident timeline reconstruction

### Production-Ready Infrastructure
- FastAPI with async support
- TimescaleDB for time-series storage
- Redis for caching and pub/sub
- Prometheus metrics integration
- Grafana dashboards
- Kubernetes-ready deployment

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         API Layer (FastAPI)                         │
├─────────────────────────────────────────────────────────────────────┤
│                      Agent Orchestrator                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Detection │ │Correlation│ │Root Cause│ │Recommend │ │ Feedback │  │
│  │  Agent   │→│  Agent   │→│  Agent   │→│  Agent   │→│  Agent   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                    ML Models & Explainability                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐   │
│  │ Z-Score │ │   STL   │ │Isolation│ │  LSTM   │ │ Transformer │   │
│  │         │ │   ESD   │ │ Forest  │ │Autoenc. │ │  Autoenc.   │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│              Feature Engineering & Data Ingestion                   │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────────┐ │
│  │  Prometheus   │ │    Stream     │ │   Feature Transformers    │ │
│  │    Client     │ │   Processor   │ │  (Stats, Seasonal, Lag)   │ │
│  └───────────────┘ └───────────────┘ └───────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                      Storage Layer                                  │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────────┐ │
│  │  TimescaleDB  │ │     Redis     │ │       Prometheus          │ │
│  └───────────────┘ └───────────────┘ └───────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
agentic-observability-platform/
├── config/                    # Configuration management
│   ├── __init__.py
│   └── settings.py           # Pydantic settings
├── src/
│   ├── types.py              # Core type definitions
│   ├── utils/                # Utilities (logging, helpers)
│   ├── ingestion/            # Data ingestion layer
│   │   ├── prometheus_client.py
│   │   ├── metrics_collector.py
│   │   └── stream_processor.py
│   ├── features/             # Feature engineering
│   │   ├── transformers.py
│   │   ├── extractor.py
│   │   └── pipeline.py
│   ├── models/               # ML models
│   │   ├── statistical.py
│   │   ├── isolation_forest.py
│   │   ├── one_class_svm.py
│   │   ├── autoencoder.py
│   │   └── ensemble.py
│   ├── agents/               # Multi-agent system
│   │   ├── detection_agent.py
│   │   ├── correlation_agent.py
│   │   ├── root_cause_agent.py
│   │   ├── recommendation_agent.py
│   │   ├── feedback_agent.py
│   │   └── orchestrator.py
│   ├── explainability/       # Explainability module
│   │   ├── shap_explainer.py
│   │   ├── natural_language.py
│   │   └── timeline.py
│   └── api/                  # FastAPI application
│       ├── app.py
│       └── routes/
├── docker/                   # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── prometheus.yml
├── kubernetes/               # Kubernetes manifests
│   ├── namespace.yaml
│   ├── deployment.yaml
│   └── ingress.yaml
├── grafana/                  # Grafana dashboards
│   ├── provisioning/
│   └── dashboards/
├── tests/                    # Test suite
├── notebooks/                # Jupyter notebooks
├── docs/                     # Documentation
└── pyproject.toml           # Project dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- (Optional) Kubernetes cluster

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/agentic-observability-platform.git
   cd agentic-observability-platform
   ```

2. **Install dependencies**
   ```bash
   pip install poetry
   poetry install
   ```

3. **Start infrastructure**
   ```bash
   cd docker
   docker-compose up -d postgres redis prometheus grafana
   ```

4. **Run the application**
   ```bash
   poetry run uvicorn src.api.app:app --reload
   ```

5. **Access the platform**
   - API Docs: http://localhost:8000/docs
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

### Docker Deployment

```bash
cd docker
docker-compose up -d
```

### Kubernetes Deployment

```bash
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/ingress.yaml
```

## 📊 API Endpoints

### Anomaly Detection
- `POST /api/v1/anomalies/detect` - Detect anomalies in metric data
- `GET /api/v1/anomalies/recent` - Get recent anomalies
- `GET /api/v1/anomalies/{id}/explain` - Get anomaly explanation

### Incidents
- `GET /api/v1/incidents` - List incidents
- `GET /api/v1/incidents/{id}` - Get incident details
- `GET /api/v1/incidents/{id}/root-causes` - Get root causes
- `GET /api/v1/incidents/{id}/recommendations` - Get recommendations
- `POST /api/v1/incidents/{id}/recommendations/{rec_id}/approve` - Approve action

### Feedback
- `POST /api/v1/feedback` - Submit anomaly feedback
- `GET /api/v1/feedback/model-performance` - Get model metrics

### Health
- `GET /api/v1/health` - Health check
- `GET /api/v1/health/agents` - Agent health status

## 🧠 How It Works

### 1. Data Ingestion
Metrics are collected from Prometheus or ingested directly via API. The stream processor handles backpressure and batching.

### 2. Feature Engineering
Raw metrics are transformed into rich feature vectors:
- Statistical features (mean, std, percentiles)
- Seasonal decomposition (STL)
- Trend analysis (linear regression, EMA)
- Change point detection (CUSUM)
- Rolling and lag features

### 3. Anomaly Detection
The Detection Agent selects optimal models based on metric characteristics:
- Seasonal metrics → STL+ESD
- High-dimensional → Isolation Forest
- Trending → One-Class SVM
- Complex patterns → Autoencoder

### 4. Correlation & Root Cause
The Correlation Agent links related anomalies using:
- Time proximity
- Service topology
- Metric similarity

The Root Cause Agent ranks causes using:
- Pattern matching
- Causal graph analysis
- Temporal ordering

### 5. Recommendations
The Recommendation Agent proposes actions from playbooks:
- Restart, Scale, Failover, Rollback
- Risk-aware ranking
- Historical effectiveness scoring

### 6. Continuous Learning
The Feedback Agent processes operator feedback to:
- Update model weights
- Trigger retraining
- Improve future predictions

## 🔧 Configuration

Configure via environment variables or `.env` file:

```env
ENVIRONMENT=development
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:pass@localhost:5432/observability
REDIS_URL=redis://localhost:6379/0
PROMETHEUS_URL=http://localhost:9090
ML_ANOMALY_THRESHOLD=0.7
ML_ENSEMBLE_MODELS=zscore,isolation_forest,autoencoder
```

## 📈 Metrics & Monitoring

The platform exports Prometheus metrics at `/metrics`:
- `anomalies_detected_total` - Total anomalies detected
- `agent_messages_routed_total` - Messages between agents
- `model_inference_duration_seconds` - Model latency
- `feedback_received_total` - Feedback counts by type

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test file
poetry run pytest tests/test_agents.py -v
```

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [Model Documentation](docs/models.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [SHAP](https://shap.readthedocs.io/) - Explainability library
- [TimescaleDB](https://www.timescale.com/) - Time-series database
- [Prometheus](https://prometheus.io/) - Monitoring system

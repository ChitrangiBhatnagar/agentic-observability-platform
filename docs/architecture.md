# Architecture Documentation

## System Overview

The Agentic AI-Driven Observability Platform is a production-grade anomaly detection system that combines multiple machine learning models with intelligent agents to provide comprehensive monitoring, correlation, root cause analysis, and automated remediation recommendations.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Sources                                │
│  (Prometheus, Kafka, Custom Exporters, Push Gateway)            │
└────────────┬────────────────────────────────────────────────────┘
             │
             v
┌─────────────────────────────────────────────────────────────────┐
│                   Data Ingestion Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  • PrometheusClient: Query metrics from Prometheus              │
│  • MetricsCollector: Continuous collection with auto-discovery  │
│  • StreamProcessor: Sliding windows & batch processing          │
└────────────┬────────────────────────────────────────────────────┘
             │
             v
┌─────────────────────────────────────────────────────────────────┐
│                Feature Engineering Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│  • StatisticalFeatures: Mean, std, percentiles                  │
│  • SeasonalDecomposer: STL decomposition                        │
│  • TrendAnalyzer: Linear regression, momentum                   │
│  • ChangePointDetector: CUSUM algorithm                         │
│  • RollingFeatures: Moving statistics                           │
│  • LagFeatures: Temporal dependencies                           │
└────────────┬────────────────────────────────────────────────────┘
             │
             v
┌─────────────────────────────────────────────────────────────────┐
│                   ML Model Portfolio                            │
├─────────────────────────────────────────────────────────────────┤
│  Statistical Models:                                            │
│    • Z-Score (with MAD variant)                                 │
│    • STL + Generalized ESD Test                                 │
│                                                                  │
│  Tree-Based Models:                                             │
│    • Isolation Forest                                           │
│    • One-Class SVM                                              │
│                                                                  │
│  Deep Learning Models:                                          │
│    • LSTM Autoencoder                                           │
│    • Transformer Autoencoder                                    │
│                                                                  │
│  Meta-Model:                                                    │
│    • Adaptive Ensemble (feedback-driven weighting)              │
└────────────┬────────────────────────────────────────────────────┘
             │
             v
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Agent Intelligence System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │ Detection Agent  │───>│ Correlation Agent│                  │
│  └──────────────────┘    └─────────┬────────┘                  │
│         │                           │                           │
│         │                           v                           │
│         │              ┌──────────────────────┐                 │
│         └─────────────>│  Root Cause Agent   │                 │
│                        └──────────┬───────────┘                 │
│                                   │                             │
│                                   v                             │
│                        ┌──────────────────────┐                 │
│                        │ Recommendation Agent │                 │
│                        └──────────┬───────────┘                 │
│                                   │                             │
│                                   v                             │
│                        ┌──────────────────────┐                 │
│                        │   Feedback Agent     │                 │
│                        └──────────────────────┘                 │
│                                   │                             │
│                                   │ (Feedback Loop)             │
│                                   └─────────────────────┐       │
│                                                         v       │
│                        ┌────────────────────────────────────┐   │
│                        │    Agent Orchestrator              │   │
│                        │  - Message routing                 │   │
│                        │  - Health monitoring               │   │
│                        │  - State management                │   │
│                        └────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
             │
             v
┌─────────────────────────────────────────────────────────────────┐
│                    Explainability Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  • SHAP Explainer: Feature importance                          │
│  • Natural Language Generator: Human-readable explanations      │
│  • Timeline Reconstructor: Event sequence analysis              │
└────────────┬────────────────────────────────────────────────────┘
             │
             v
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Application:                                           │
│    • Health & Metrics Endpoints                                 │
│    • Anomaly Detection & Ingestion                             │
│    • Incident Management                                        │
│    • Feedback & Retraining                                      │
│    • Explainability Endpoints                                   │
└────────────┬────────────────────────────────────────────────────┘
             │
             v
┌─────────────────────────────────────────────────────────────────┐
│                    Storage Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  • TimescaleDB: Metrics & time-series data                      │
│  • PostgreSQL: Incidents, feedback, decisions                   │
│  • Redis: Caching, session state                               │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Ingestion Layer

**Purpose**: Collect and preprocess metrics from various sources

**Components**:
- **PrometheusClient**: Async HTTP client for querying Prometheus
  - Supports instant queries and range queries
  - Exponential backoff retry logic
  - Connection pooling

- **MetricsCollector**: Orchestrates continuous metric collection
  - Auto-discovery of metrics
  - Configurable collection intervals
  - Callback-based extensibility
  - Health tracking

- **StreamProcessor**: Manages streaming data
  - Sliding window aggregation
  - Buffered batch processing
  - Backpressure handling

### 2. Feature Engineering Pipeline

**Purpose**: Transform raw metrics into ML-ready features

**Transformers**:

1. **StatisticalFeatures**
   - Basic statistics (mean, std, min, max)
   - Percentiles (configurable)
   - Coefficient of variation

2. **SeasonalDecomposer**
   - STL (Seasonal-Trend decomposition using Loess)
   - Seasonal strength calculation
   - Trend strength calculation

3. **TrendAnalyzer**
   - Linear regression trend
   - Exponential moving average
   - Momentum calculation

4. **ChangePointDetector**
   - CUSUM algorithm
   - Identifies structural breaks

5. **RollingFeatures**
   - Moving averages
   - Rolling standard deviation
   - Rolling min/max

6. **LagFeatures**
   - Temporal dependencies
   - Autocorrelation features

### 3. ML Model Portfolio

**Purpose**: Multi-model approach for robust detection

**Models**:

1. **Z-Score Detector**
   - Fast, interpretable
   - MAD (Median Absolute Deviation) variant
   - Online learning support
   - Best for: Gaussian distributions

2. **STL+ESD Detector**
   - Handles seasonality
   - Generalized ESD test for outliers
   - Best for: Seasonal time series

3. **Isolation Forest**
   - Tree-based anomaly detection
   - Feature importance
   - Best for: High-dimensional data

4. **One-Class SVM**
   - Kernel-based method
   - RBF kernel
   - Best for: Complex boundaries

5. **LSTM Autoencoder**
   - Deep learning approach
   - Captures temporal patterns
   - Reconstruction error-based
   - Best for: Sequential dependencies

6. **Transformer Autoencoder**
   - Attention mechanism
   - Multi-head attention
   - Best for: Long-range dependencies

7. **Adaptive Ensemble**
   - Combines all models
   - Feedback-driven weight adaptation
   - Dynamic model selection
   - Best for: Production deployment

### 4. Multi-Agent System

**Purpose**: Intelligent analysis and response orchestration

**Agents**:

1. **Detection Agent**
   - Manages model portfolio
   - Contextual model selection
   - Anomaly classification
   - Confidence scoring

2. **Correlation Agent**
   - Time-based correlation
   - Service topology analysis
   - Metric similarity (Jaccard)
   - Cluster management
   - Incident creation

3. **Root Cause Agent**
   - Pattern matching (6 built-in patterns)
   - Topology-based analysis
   - Temporal analysis
   - Causal link identification
   - Confidence scoring

4. **Recommendation Agent**
   - 9 action templates
   - Risk assessment
   - Impact estimation
   - Auto-approval for low-risk actions
   - Effectiveness tracking

5. **Feedback Agent**
   - Model performance tracking
   - Sample quality assessment
   - Auto-retrain triggers
   - Label management

**Agent Communication**:
- Async message passing
- Priority-based queues
- Shared memory (short/long-term)
- Event-driven architecture

### 5. Explainability Layer

**Purpose**: Make AI decisions interpretable

**Components**:

1. **SHAP Explainer**
   - Feature importance
   - Supports tree and kernel methods
   - Fallback perturbation-based
   - Ensemble explanations

2. **Natural Language Generator**
   - Template-based generation
   - Severity indicators
   - Multiple output formats (Slack, Email, PagerDuty)
   - Contextual descriptions

3. **Timeline Reconstructor**
   - Event sequence analysis
   - Phase identification
   - Summary statistics
   - Multiple formats (text, markdown, HTML)

### 6. API Layer

**Purpose**: RESTful interface for integration

**Endpoints**:

- **Health**: `/health`, `/health/live`, `/health/ready`
- **Anomalies**: `/anomalies/detect`, `/anomalies/ingest`, `/anomalies/recent`
- **Incidents**: `/incidents`, `/incidents/active`, `/incidents/{id}`
- **Feedback**: `/feedback`, `/model-performance`, `/retrain-status`

**Features**:
- Async handlers
- Pydantic validation
- CORS support
- Prometheus metrics
- OpenAPI documentation

### 7. Storage Layer

**Purpose**: Persistent data storage

**Databases**:

1. **TimescaleDB**
   - Hypertables for metrics
   - Automatic partitioning
   - Retention policies
   - Continuous aggregates

2. **PostgreSQL**
   - Incidents, anomalies
   - Feedback, decisions
   - Correlations
   - Labeled samples

3. **Redis**
   - Caching
   - Session state
   - Rate limiting
   - Pub/Sub messaging

## Deployment Architecture

### Docker Compose (Development)

```yaml
Services:
  - app (FastAPI)
  - postgres (TimescaleDB)
  - redis
  - prometheus
  - grafana
  - kafka (optional)
  - demo-generator
```

### Kubernetes (Production)

```yaml
Resources:
  - Deployment (3 replicas)
  - HPA (2-10 replicas, CPU/memory based)
  - PDB (minAvailable: 1)
  - Service (ClusterIP)
  - Ingress (TLS, rate limiting)
  - NetworkPolicy (restricted egress/ingress)
  - ConfigMap + Secret
  - PersistentVolumeClaim (models)
```

## Data Flow

1. **Metrics Collection**
   ```
   Prometheus → PrometheusClient → MetricsCollector → StreamProcessor
   ```

2. **Feature Extraction**
   ```
   StreamProcessor → FeaturePipeline → FeatureExtractor → Feature Vector
   ```

3. **Anomaly Detection**
   ```
   Feature Vector → Model Portfolio → Ensemble → Anomaly Score
   ```

4. **Agent Processing**
   ```
   Anomaly → Detection Agent → Correlation Agent → Root Cause Agent → 
   Recommendation Agent → Feedback Agent
   ```

5. **Storage & Response**
   ```
   Agents → Database Repositories → API Response
   ```

## Scalability Considerations

1. **Horizontal Scaling**
   - Stateless API design
   - Redis for shared state
   - Kubernetes HPA

2. **Vertical Scaling**
   - Configurable worker processes
   - Thread pool sizing
   - Database connection pooling

3. **Performance Optimization**
   - Feature caching
   - Model result caching
   - Batch processing
   - Async I/O throughout

4. **Resource Management**
   - Memory limits per model
   - GPU support for deep learning
   - Lazy model loading
   - Periodic cleanup

## Security

1. **Authentication**
   - API key support
   - JWT token validation
   - Role-based access control

2. **Network Security**
   - TLS/SSL encryption
   - Network policies
   - Ingress restrictions

3. **Data Security**
   - Secrets management
   - Environment variable encryption
   - Database encryption at rest

## Monitoring & Observability

1. **Application Metrics**
   - Request latency
   - Error rates
   - Model performance
   - Agent health

2. **Infrastructure Metrics**
   - CPU, memory usage
   - Database connections
   - Cache hit rates

3. **Business Metrics**
   - Anomalies detected
   - Incidents created
   - Feedback accuracy
   - Model retraining frequency

## Future Enhancements

1. **Advanced ML**
   - Graph neural networks for topology
   - Reinforcement learning for recommendations
   - Transfer learning

2. **Integration**
   - Webhooks for external systems
   - Bi-directional Slack integration
   - JIRA ticket creation
   - PagerDuty integration

3. **Automation**
   - Auto-remediation execution
   - Capacity planning
   - Predictive scaling

4. **Observability**
   - Distributed tracing
   - Log aggregation
   - APM integration

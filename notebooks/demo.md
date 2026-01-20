# Agentic AI Observability Platform - Demo Notebook

This notebook demonstrates the key features of the Agentic AI-Driven Observability & Anomaly Detection Platform.

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import asyncio

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
%matplotlib inline
```

## 1. Generate Sample Time Series Data

```python
# Generate synthetic metrics with anomalies
def generate_sample_data(n_points=1000, inject_anomalies=True):
    """Generate time series with optional anomalies."""
    t = np.linspace(0, 100, n_points)
    
    # Normal pattern
    trend = 50 + 0.1 * t
    seasonal = 15 * np.sin(2 * np.pi * t / 24)
    noise = np.random.normal(0, 3, len(t))
    data = trend + seasonal + noise
    
    if inject_anomalies:
        # Spike anomaly
        data[500:510] += 40
        # Drop anomaly
        data[700:720] -= 30
        # Gradual drift
        data[850:] += np.linspace(0, 20, len(data[850:]))
    
    timestamps = [datetime.now() - timedelta(minutes=i) for i in range(n_points-1, -1, -1)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'value': data
    })

df = generate_sample_data()

# Plot
plt.figure(figsize=(15, 5))
plt.plot(df['timestamp'], df['value'])
plt.title('Sample Metric: CPU Usage (%)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.tight_layout()
plt.show()
```

## 2. Test Anomaly Detection Models

```python
from src.models import ZScoreDetector, STLESDDetector, IsolationForestDetector, EnsembleDetector

# Prepare data
values = df['value'].values

# Test Z-Score Detector
print("=" * 50)
print("Z-Score Detector")
print("=" * 50)
zscore = ZScoreDetector(threshold=3.0, use_mad=True)
zscore.fit(values[:800])
result = zscore.detect(values)
print(f"Detected {np.sum(result.predictions)} anomalies")
print(f"Max anomaly score: {result.anomaly_scores.max():.3f}")

# Test STL+ESD Detector
print("\n" + "=" * 50)
print("STL+ESD Detector")
print("=" * 50)
stl = STLESDDetector(period=24, alpha=0.05)
stl.fit(values)
result_stl = stl.detect(values)
print(f"Detected {np.sum(result_stl.predictions)} anomalies")

# Test Isolation Forest
print("\n" + "=" * 50)
print("Isolation Forest")
print("=" * 50)
iforest = IsolationForestDetector(contamination=0.05)
X = values.reshape(-1, 1)
iforest.fit(X)
result_if = iforest.detect(X)
print(f"Detected {np.sum(result_if.predictions)} anomalies")

# Visualize results
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Z-Score
axes[0].plot(df['timestamp'], values, label='Metric', alpha=0.7)
axes[0].scatter(df['timestamp'][result.predictions], 
                values[result.predictions], 
                color='red', label='Anomalies', s=50, zorder=5)
axes[0].set_title('Z-Score Detection')
axes[0].legend()

# STL+ESD
axes[1].plot(df['timestamp'], values, label='Metric', alpha=0.7)
axes[1].scatter(df['timestamp'][result_stl.predictions], 
                values[result_stl.predictions], 
                color='orange', label='Anomalies', s=50, zorder=5)
axes[1].set_title('STL+ESD Detection')
axes[1].legend()

# Isolation Forest
axes[2].plot(df['timestamp'], values, label='Metric', alpha=0.7)
axes[2].scatter(df['timestamp'][result_if.predictions], 
                values[result_if.predictions], 
                color='purple', label='Anomalies', s=50, zorder=5)
axes[2].set_title('Isolation Forest Detection')
axes[2].legend()

plt.tight_layout()
plt.show()
```

## 3. Feature Engineering Pipeline

```python
from src.features import FeatureExtractor, StatisticalFeatures, SeasonalDecomposer, TrendAnalyzer

# Extract features
extractor = FeatureExtractor()
features = extractor.extract(values)

print("Extracted Features:")
print("=" * 50)
for key, value in features.items():
    if isinstance(value, (int, float)):
        print(f"{key:30s}: {value:.3f}")
    elif isinstance(value, np.ndarray) and value.size == 1:
        print(f"{key:30s}: {value.item():.3f}")

# Visualize seasonal decomposition
decomposer = SeasonalDecomposer(period=24)
seasonal_features = decomposer.transform(values)

if 'trend' in seasonal_features and 'seasonal' in seasonal_features:
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    
    axes[0].plot(values)
    axes[0].set_title('Original')
    
    axes[1].plot(seasonal_features['trend'])
    axes[1].set_title('Trend Component')
    
    axes[2].plot(seasonal_features['seasonal'])
    axes[2].set_title('Seasonal Component')
    
    axes[3].plot(seasonal_features['residual'])
    axes[3].set_title('Residual')
    
    plt.tight_layout()
    plt.show()
```

## 4. Ensemble Model with Adaptive Weighting

```python
from src.models import AdaptiveEnsembleDetector

# Create ensemble
detectors = {
    'zscore': ZScoreDetector(use_mad=True),
    'stl_esd': STLESDDetector(period=24),
    'iforest': IsolationForestDetector(),
}

ensemble = AdaptiveEnsembleDetector(detectors, learning_rate=0.1)

# Fit on training data
X_train = values[:800].reshape(-1, 1)
ensemble.fit(X_train)

# Detect anomalies
X_test = values.reshape(-1, 1)
result_ensemble = ensemble.detect(X_test)

print(f"Ensemble detected {np.sum(result_ensemble.predictions)} anomalies")
print(f"\nModel weights:")
for name, weight in ensemble.weights.items():
    print(f"  {name:15s}: {weight:.3f}")

# Visualize
plt.figure(figsize=(15, 6))
plt.plot(df['timestamp'], values, label='Metric', alpha=0.7)
plt.scatter(df['timestamp'][result_ensemble.predictions], 
            values[result_ensemble.predictions], 
            color='red', label='Anomalies', s=80, zorder=5, marker='*')
plt.title('Adaptive Ensemble Detection')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

# Plot anomaly scores over time
plt.figure(figsize=(15, 4))
plt.plot(df['timestamp'], result_ensemble.anomaly_scores, label='Anomaly Score')
plt.axhline(y=0.7, color='r', linestyle='--', label='Threshold')
plt.title('Anomaly Scores Over Time')
plt.xlabel('Time')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.show()
```

## 5. Agent System Demo

```python
from src.agents import create_default_orchestrator
from src.types import Anomaly, Severity, AnomalyType, FeedbackType

# Create orchestrator (async context required)
async def demo_agents():
    orchestrator = await create_default_orchestrator()
    await orchestrator.start()
    
    # Create sample anomaly
    anomaly = Anomaly(
        id="demo-anomaly-1",
        metric_name="cpu_usage",
        labels={"service": "api-gateway", "env": "production"},
        anomaly_type=AnomalyType.SPIKE,
        severity=Severity.HIGH,
        ensemble_score=0.92,
        confidence=0.88,
        value=95.5,
        expected_value=45.2,
        deviation=0.527,
        timestamp=datetime.utcnow(),
    )
    
    # Process anomaly through agent pipeline
    print("Processing anomaly through agent pipeline...")
    result = await orchestrator.process_anomaly(anomaly)
    
    print(f"\nDetection Agent Decision: {result['detection'].action}")
    print(f"Anomaly Type: {result['detection'].metadata.get('classified_type')}")
    
    if result.get('incident'):
        print(f"\nIncident Created: {result['incident'].title}")
        print(f"Affected Services: {result['incident'].affected_services}")
    
    if result.get('root_causes'):
        print(f"\nIdentified {len(result['root_causes'])} potential root causes:")
        for cause in result['root_causes'][:3]:
            print(f"  - {cause.description} (probability: {cause.probability:.2f})")
    
    if result.get('recommendations'):
        print(f"\nGenerated {len(result['recommendations'])} recommendations:")
        for rec in result['recommendations'][:3]:
            print(f"  - {rec.action_type}: {rec.description}")
            print(f"    Risk: {rec.risk_level}, Impact: {rec.estimated_impact}")
    
    # Submit feedback
    print("\nSubmitting positive feedback...")
    feedback_result = await orchestrator.submit_feedback(
        anomaly_id="demo-anomaly-1",
        feedback_type=FeedbackType.TRUE_POSITIVE,
        comment="Confirmed CPU spike during deployment",
    )
    print(f"Feedback processed: {feedback_result['decision'].action}")
    
    # Get agent health
    health = orchestrator.get_agent_health()
    print("\nAgent Health Status:")
    for agent_name, agent_health in health.items():
        print(f"  {agent_name:20s}: {agent_health.status} "
              f"(processed: {agent_health.messages_processed}, "
              f"errors: {agent_health.error_count})")
    
    await orchestrator.stop()

# Run demo
await demo_agents()
```

## 6. Explainability Demo

```python
from src.explainability import SHAPExplainer, NaturalLanguageExplainer

# Feature importance with SHAP
feature_names = ['mean', 'std', 'trend_slope', 'seasonal_strength', 'recent_change']
feature_values = np.array([[55.2, 12.3, 0.15, 0.72, 8.5]])

# Try SHAP explanation
explainer = SHAPExplainer(model=ensemble, feature_names=feature_names)
try:
    importance = explainer.explain(feature_values[0])
    
    print("Feature Importance:")
    sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, imp in sorted_features:
        print(f"  {feat:25s}: {imp:+.4f}")
    
    # Visualize
    features, importances = zip(*sorted_features)
    plt.figure(figsize=(10, 5))
    plt.barh(features, importances)
    plt.xlabel('SHAP Value')
    plt.title('Feature Importance for Anomaly Detection')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"SHAP explanation failed (using fallback): {e}")

# Natural language explanation
nl_explainer = NaturalLanguageExplainer()

explanation_text = nl_explainer.generate_summary(
    anomaly,
    top_features=[
        ('recent_change', 0.35),
        ('std', 0.28),
        ('mean', 0.22),
    ]
)

print("\n" + "=" * 70)
print("NATURAL LANGUAGE EXPLANATION")
print("=" * 70)
print(explanation_text)
```

## 7. API Integration Example

```python
import httpx

async def test_api():
    """Test API endpoints."""
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        # Health check
        response = await client.get(f"{base_url}/api/v1/health")
        print("Health Check:")
        print(response.json())
        
        # Ingest metrics
        metrics_data = {
            "metrics": [
                {
                    "metric_name": "cpu_usage",
                    "value": 85.5,
                    "timestamp": datetime.utcnow().isoformat(),
                    "labels": {"service": "api", "env": "prod"}
                }
            ]
        }
        
        response = await client.post(
            f"{base_url}/api/v1/anomalies/ingest",
            json=metrics_data
        )
        print("\nMetric Ingestion:")
        print(response.json())
        
        # Get recent anomalies
        response = await client.get(f"{base_url}/api/v1/anomalies/recent?limit=5")
        print("\nRecent Anomalies:")
        print(response.json())

# Note: Run this when the API server is running
# await test_api()
print("API test code ready. Start the server with 'python main.py' and uncomment the last line.")
```

## Summary

This notebook demonstrated:
1. ✅ Generating synthetic time series data with anomalies
2. ✅ Testing multiple anomaly detection models
3. ✅ Feature engineering pipeline
4. ✅ Adaptive ensemble with model weighting
5. ✅ Multi-agent orchestration system
6. ✅ Explainability with SHAP and natural language
7. ✅ API integration patterns

The platform provides production-ready anomaly detection with:
- **Multiple ML models** (statistical, tree-based, deep learning)
- **Intelligent agents** for correlation, root cause analysis, and recommendations
- **Explainable AI** with SHAP and natural language generation
- **REST API** for integration
- **Docker & Kubernetes** deployment

-- Initialize Database for Agentic Observability Platform

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create tables

-- Metrics table (hypertable for time-series)
CREATE TABLE IF NOT EXISTS metrics (
    time TIMESTAMPTZ NOT NULL,
    metric_name TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    labels JSONB DEFAULT '{}',
    service TEXT,
    instance TEXT
);

-- Convert to hypertable
SELECT create_hypertable('metrics', 'time', if_not_exists => TRUE);

-- Create index for fast queries
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics (metric_name, time DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_service ON metrics (service, time DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_labels ON metrics USING GIN (labels);

-- Anomalies table
CREATE TABLE IF NOT EXISTS anomalies (
    id TEXT PRIMARY KEY,
    metric_name TEXT NOT NULL,
    labels JSONB DEFAULT '{}',
    anomaly_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    ensemble_score DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    value DOUBLE PRECISION,
    expected_value DOUBLE PRECISION,
    deviation DOUBLE PRECISION,
    timestamp TIMESTAMPTZ NOT NULL,
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    correlated_anomaly_ids TEXT[],
    model_scores JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_anomalies_time ON anomalies (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_anomalies_severity ON anomalies (severity, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_anomalies_metric ON anomalies (metric_name, timestamp DESC);

-- Incidents table
CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    severity TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    anomaly_ids TEXT[],
    affected_services TEXT[],
    root_cause_ids TEXT[],
    recommendation_ids TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents (severity, created_at DESC);

-- Root causes table
CREATE TABLE IF NOT EXISTS root_causes (
    id TEXT PRIMARY KEY,
    incident_id TEXT REFERENCES incidents(id),
    category TEXT NOT NULL,
    description TEXT,
    probability DOUBLE PRECISION,
    evidence TEXT[],
    affected_components TEXT[],
    suggested_investigation TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    was_correct BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_root_causes_incident ON root_causes (incident_id);

-- Recommendations table
CREATE TABLE IF NOT EXISTS recommendations (
    id TEXT PRIMARY KEY,
    incident_id TEXT REFERENCES incidents(id),
    title TEXT NOT NULL,
    description TEXT,
    action_type TEXT NOT NULL,
    risk_level TEXT,
    confidence DOUBLE PRECISION,
    expected_impact TEXT,
    status TEXT DEFAULT 'pending',
    parameters JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    approved_at TIMESTAMPTZ,
    executed_at TIMESTAMPTZ,
    outcome_success BOOLEAN,
    time_to_resolve INTERVAL
);

CREATE INDEX IF NOT EXISTS idx_recommendations_incident ON recommendations (incident_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_status ON recommendations (status);

-- Feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    anomaly_id TEXT REFERENCES anomalies(id),
    feedback_type TEXT NOT NULL,
    comment TEXT,
    labeler TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_anomaly ON feedback (anomaly_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback (feedback_type);

-- Labeled samples table (for model training)
CREATE TABLE IF NOT EXISTS labeled_samples (
    id TEXT PRIMARY KEY,
    anomaly_id TEXT,
    metric TEXT,
    features JSONB NOT NULL,
    label BOOLEAN NOT NULL,
    confidence DOUBLE PRECISION,
    labeler TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_samples_metric ON labeled_samples (metric);
CREATE INDEX IF NOT EXISTS idx_samples_label ON labeled_samples (label);

-- Model metadata table
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    version TEXT,
    parameters JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE
);

-- Agent decisions table (audit log)
CREATE TABLE IF NOT EXISTS agent_decisions (
    id TEXT PRIMARY KEY,
    agent_type TEXT NOT NULL,
    decision TEXT NOT NULL,
    reasoning TEXT,
    confidence DOUBLE PRECISION,
    input_data JSONB,
    output_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_decisions_agent ON agent_decisions (agent_type, created_at DESC);

-- Correlation matrix table
CREATE TABLE IF NOT EXISTS correlations (
    metric_pair TEXT PRIMARY KEY,
    metric_a TEXT NOT NULL,
    metric_b TEXT NOT NULL,
    correlation_score DOUBLE PRECISION,
    sample_count INTEGER DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_correlations_metrics ON correlations (metric_a, metric_b);

-- Create retention policies (keep 30 days of metrics, 90 days of anomalies)
SELECT add_retention_policy('metrics', INTERVAL '30 days', if_not_exists => TRUE);

-- Create continuous aggregates for faster queries
CREATE MATERIALIZED VIEW IF NOT EXISTS metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    metric_name,
    service,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value,
    COUNT(*) as sample_count
FROM metrics
GROUP BY bucket, metric_name, service
WITH NO DATA;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('metrics_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO observability_user;

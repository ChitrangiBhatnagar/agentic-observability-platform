# Models Directory

This directory stores trained ML models and their artifacts.

## Structure

- `checkpoints/` - Model checkpoints during training
- `production/` - Production-ready models
- `experiments/` - Experimental model versions
- `metadata/` - Model metadata and performance metrics

## Naming Convention

Models should follow this naming pattern:
```
{model_type}_{version}_{timestamp}.pkl
```

Examples:
- `ensemble_v1_20240119.pkl`
- `lstm_autoencoder_v2_20240119.pth`
- `isolation_forest_v1_20240119.pkl`

## Note

Model files are excluded from version control via `.gitignore`.
Use model versioning and registry for production deployments.

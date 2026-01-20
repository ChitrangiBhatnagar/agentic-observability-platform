# Contributing Guidelines

Thank you for considering contributing to the Agentic AI-Driven Observability Platform! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)
8. [Release Process](#release-process)

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing viewpoints and experiences

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Git
- Poetry (recommended) or pip

### Setup Development Environment

1. Fork the repository on GitHub

2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/agentic-observability-platform.git
cd agentic-observability-platform
```

3. Add upstream remote:
```bash
git remote add upstream https://github.com/ChitrangiBhatnagar/agentic-observability-platform.git
```

4. Install dependencies:
```bash
poetry install --with dev
```

5. Set up pre-commit hooks:
```bash
pre-commit install
```

6. Start development services:
```bash
docker-compose up -d postgres redis prometheus
```

## Development Workflow

### 1. Create a Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

### 2. Make Changes

- Write clear, concise commit messages
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Commit Your Changes

```bash
git add .
git commit -m "feat: add new anomaly detection model"
```

Commit message format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Build process or auxiliary tool changes

### 4. Keep Your Branch Updated

```bash
git fetch upstream
git rebase upstream/main
```

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 6. Create a Pull Request

- Go to the repository on GitHub
- Click "New Pull Request"
- Select your branch
- Fill out the PR template
- Request review

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Maximum line length: 100 characters
# String quotes: Double quotes preferred
# Imports: Organized and sorted

# Example:
from typing import List, Optional
import asyncio

from src.models.base import BaseAnomalyDetector
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MyDetector(BaseAnomalyDetector):
    """
    Brief description.
    
    Longer description with more details about the class,
    its purpose, and usage examples.
    
    Attributes:
        threshold: Anomaly detection threshold
        window_size: Size of the sliding window
    """
    
    def __init__(self, threshold: float = 0.7, window_size: int = 100):
        """
        Initialize the detector.
        
        Args:
            threshold: Anomaly detection threshold (0-1)
            window_size: Number of points in sliding window
        """
        self.threshold = threshold
        self.window_size = window_size
    
    async def detect(self, data: np.ndarray) -> DetectionResult:
        """
        Detect anomalies in the data.
        
        Args:
            data: Input time series data
            
        Returns:
            DetectionResult with anomaly predictions
            
        Raises:
            ValueError: If data is invalid
        """
        if len(data) == 0:
            raise ValueError("Input data cannot be empty")
        
        # Implementation
        ...
```

### Type Hints

Always use type hints:

```python
from typing import List, Dict, Optional, Union

def process_metrics(
    metrics: List[Dict[str, float]],
    threshold: Optional[float] = None
) -> List[str]:
    """Process metrics and return anomaly IDs."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description of the function.
    
    More detailed description if needed. Can span multiple
    lines and include code examples.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
        TypeError: When param1 is not a string
        
    Example:
        >>> result = example_function("test", 5)
        >>> print(result)
        True
    """
    ...
```

### Async/Await

Use async/await for I/O-bound operations:

```python
async def fetch_data(url: str) -> dict:
    """Fetch data from URL asynchronously."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

async def main():
    """Main async entry point."""
    data = await fetch_data("http://example.com/api")
    # Process data
```

### Error Handling

Handle errors appropriately:

```python
try:
    result = await risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    # Handle or re-raise
```

### Logging

Use structured logging:

```python
from src.utils.logging import get_logger

logger = get_logger(__name__)

logger.info("Processing started", metric_name=metric, count=len(data))
logger.warning("Threshold exceeded", value=value, threshold=threshold)
logger.error("Failed to process", error=str(e), exc_info=True)
```

## Testing

### Writing Tests

- Use pytest for testing
- Organize tests to mirror source structure
- Use fixtures for common setup
- Aim for >80% code coverage

```python
import pytest
import numpy as np

from src.models import ZScoreDetector


class TestZScoreDetector:
    """Tests for Z-Score detector."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return ZScoreDetector(threshold=3.0)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        return np.random.randn(1000)
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.threshold == 3.0
        assert detector.model is None
    
    def test_fit(self, detector, sample_data):
        """Test model fitting."""
        detector.fit(sample_data)
        assert detector.model is not None
        assert detector.metadata["n_samples"] == 1000
    
    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async functionality."""
        result = await some_async_function()
        assert result is not None
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run specific test
pytest tests/test_models.py::TestZScoreDetector::test_fit

# Run with verbose output
pytest -v

# Run in parallel
pytest -n auto
```

### Test Coverage

Maintain high test coverage:

```bash
# Generate coverage report
pytest --cov=src --cov-report=term-missing

# View HTML report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## Documentation

### Code Documentation

- All public functions/classes must have docstrings
- Include type hints
- Provide examples where helpful
- Document exceptions

### README Updates

Update README.md when:
- Adding new features
- Changing installation process
- Modifying API endpoints
- Updating dependencies

### Architecture Documentation

Update `docs/architecture.md` for:
- New components
- System design changes
- Integration points

### API Documentation

FastAPI auto-generates API docs, but ensure:
- Pydantic models have descriptions
- Endpoint docstrings are clear
- Request/response examples are provided

## Pull Request Process

### Before Submitting

1. Ensure all tests pass:
```bash
pytest
```

2. Check code style:
```bash
black src tests
isort src tests
flake8 src tests
mypy src
```

3. Update documentation

4. Add entry to CHANGELOG.md

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

- At least one approval required
- All CI checks must pass
- Address review comments
- Keep PR scope focused

### After Merge

- Delete your branch
- Update your local repository:
```bash
git checkout main
git pull upstream main
```

## Release Process

### Versioning

We follow Semantic Versioning (SemVer):

- MAJOR.MINOR.PATCH
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Creating a Release

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release branch:
```bash
git checkout -b release/v1.2.0
```

4. Commit changes:
```bash
git commit -m "chore: prepare release v1.2.0"
```

5. Create tag:
```bash
git tag -a v1.2.0 -m "Release v1.2.0"
```

6. Push tag:
```bash
git push upstream v1.2.0
```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues and PRs first

Thank you for contributing! 🎉

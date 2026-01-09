# Sentiment Analyzer MLOps

**Author:** Aina Tiavina Ratefiarivony

A complete MLOps pipeline for sentiment analysis of French movie reviews, featuring model versioning, automated testing, deployment, and monitoring capabilities.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [CLI Commands](#cli-commands)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Development](#development)
- [Configuration](#configuration)
- [Project Structure](#project-structure)

## Overview

This project implements a full ML lifecycle for binary sentiment classification (positive/negative) on French movie reviews. It demonstrates MLOps best practices including:

- Experiment tracking with MLflow
- Model versioning and registry
- Automated testing before production promotion
- Containerized deployment with Docker
- Prediction logging and monitoring with MongoDB

## Features

- **Model Training & Tracking**: Logistic Regression model with MLflow experiment tracking
- **Model Registry**: Version control and stage management (Staging/Production)
- **CLI Tools**: Commands for prediction, model promotion, and model retrieval
- **REST API**: FastAPI backend for real-time predictions
- **Web UI**: Streamlit frontend for interactive usage
- **Prediction Logging**: MongoDB storage for prediction history and monitoring
- **Containerization**: Docker Compose orchestration for all services

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Compose                           │
├─────────────────┬─────────────────────────┬─────────────────────┤
│   Frontend      │       WebApp            │     Database        │
│   (Streamlit)   │      (FastAPI)          │    (MongoDB)        │
│   :8501         │       :8001             │     :27017          │
│                 │                         │                     │
│   User Input ──►│──► /predict ──► Model   │                     │
│                 │         │               │                     │
│   ◄── Display ──│◄── Response ◄───────────│◄── Log predictions  │
│                 │                         │                     │
│                 │    /history ────────────│──► Retrieve logs    │
└─────────────────┴─────────────────────────┴─────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     MLflow Server (:5000)                       │
├─────────────────────────────────────────────────────────────────┤
│  Experiment Tracking  │  Model Registry  │  Model Artifacts     │
│  - Metrics            │  - Versions      │  - model.pkl         │
│  - Parameters         │  - Stages        │  - MLmodel           │
│  - Runs               │  - Aliases       │  - requirements.txt  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- MLflow server (for model registry features)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mlops_project
```

2. Install the package:
```bash
pip install -e .
```

3. Start MLflow server (optional, for registry features):
```bash
mlflow server --host 0.0.0.0 --port 5000
```

### Running with Docker Compose

Start all services:
```bash
docker-compose up -d
```

This starts:
- **Frontend**: http://localhost:8501 (Streamlit UI)
- **WebApp API**: http://localhost:8001 (FastAPI)
- **MongoDB**: localhost:27017

Stop services:
```bash
docker-compose down
```

## CLI Commands

The project provides four CLI commands:

### predict

Run sentiment predictions on text or CSV files.

```bash
# Single text prediction
predict --text "Ce film était excellent !" --model_name sentiment_model --model_version 1

# Batch prediction from CSV
predict --input_file reviews.csv --output_file predictions.csv --model_name sentiment_model --model_version 1
```

**Options:**
| Option | Description |
|--------|-------------|
| `--text` | Single text to predict |
| `--input_file` | CSV file with "review" column |
| `--output_file` | Output CSV path |
| `--model_name` | Model name in MLflow registry (required) |
| `--model_version` | Model version (required) |
| `--mlflow_url` | MLflow server URL (default: http://localhost:5000) |

### promote

Promote a model to Staging or Production with automatic testing.

```bash
# Promote to Staging
promote --model_name sentiment_model --model_version 1 --status Staging

# Promote to Production (runs tests automatically)
promote --model_name sentiment_model --model_version 1 --status Production --test_set test_data.csv
```

**Options:**
| Option | Description |
|--------|-------------|
| `--model_name` | Model name in registry (required) |
| `--model_version` | Version to promote (required) |
| `--status` | Target stage: "Staging" or "Production" (required) |
| `--test_set` | Path to test CSV (required for Production) |
| `--mlflow_url` | MLflow server URL |

### get_model

Download a model from MLflow registry.

```bash
get_model --mlflow_server_uri http://localhost:5000 --model_name sentiment_model --model_version 1 --target_path ./model
```

### retrain

Retrain a model with new data (WIP).

```bash
retrain --model_name sentiment_model --model_version 1 --training_set new_data.csv --register_updated_model
```

## API Reference

### POST /predict

Predict sentiment for a list of reviews.

**Request:**
```json
{
  "reviews": ["Ce film était génial !", "Film décevant et ennuyeux."]
}
```

**Response:**
```json
{
  "sentiments": ["positif", "negatif"]
}
```

### GET /history

Retrieve prediction history.

**Parameters:**
- `n` (int, default=5): Number of records to retrieve

**Response:**
```json
[
  {
    "_id": "...",
    "reviews": ["Ce film était génial !"],
    "sentiments": ["positif"],
    "timestamp": "2025-12-02T13:37:16.393279"
  }
]
```

### API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8001/docs
- OpenAPI spec: http://localhost:8001/openapi.json

## Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest src/sentiment_analyzer/tests/

# Run with verbose output
pytest -v src/sentiment_analyzer/tests/test_model.py
```

### Test Suite

| Test | Description |
|------|-------------|
| `test_model_output_type_with_simple_input` | Validates output type is numpy.int64 |
| `test_model_work_with_unusual_input` | Tests special character handling |
| `test_model_work_with_empty_input` | Validates error handling for empty input |
| `test_model_prediction_with_obvious_input` | Checks sentiment correctness |
| `test_model_accuracy` | Evaluates accuracy > 0.8 threshold |
| `test_model_better_than_baseline` | Compares against baseline model |

### Environment Variables for Tests

```bash
export TEST_MODEL_NAME=sentiment_model
export TEST_MODEL_VERSION=1
export TEST_TEST_SET=path/to/test.csv
export TEST_BASELINE_MODEL=baseline_model
export TEST_BASELINE_VERSION=1
```

## Development

### Jupyter Notebooks

Notebooks for exploration and model development are in `notebooks/`:

| Notebook | Purpose |
|----------|---------|
| `exploratory_analysis.ipynb` | Data exploration and analysis |
| `model_design.ipynb` | Initial model design |
| `model_design_2.ipynb` | Model iterations |
| `model_design_3.ipynb` | Further refinements |
| `llm_mlops.ipynb` | LLM integration experiments |

### Model Development Workflow

1. **Explore data** in `exploratory_analysis.ipynb`
2. **Design model** in `model_design*.ipynb`
3. **Track experiments** with MLflow
4. **Register model** in MLflow registry
5. **Promote to Staging** with `promote` CLI
6. **Run tests** and validate
7. **Promote to Production**
8. **Deploy** with Docker Compose

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server URL | http://localhost:5000 |
| `SENTIMENT_ANALYZER_MODEL_PATH` | Model path in container | . |
| `WEBAPP_URL` | WebApp URL for frontend | webapp:8000 |

### MongoDB Configuration

| Setting | Value |
|---------|-------|
| Host | mongodb (docker) / localhost |
| Port | 27017 |
| Database | sentiment_db |
| Collection | logs |
| Username | admin |
| Password | secretpassword |

## Project Structure

```
mlops_project/
├── notebooks/                    # Jupyter notebooks
│   ├── exploratory_analysis.ipynb
│   ├── model_design.ipynb
│   ├── model_design_2.ipynb
│   ├── model_design_3.ipynb
│   └── llm_mlops.ipynb
├── src/
│   ├── sentiment_analyzer/       # Core ML module
│   │   ├── model_manager.py      # MLflow integration
│   │   ├── predict.py            # Prediction CLI
│   │   ├── promote.py            # Model promotion CLI
│   │   ├── retrain.py            # Retraining CLI (WIP)
│   │   ├── get_mlflow_model.py   # Model download utility
│   │   └── tests/
│   │       └── test_model.py     # Test suite
│   ├── webapp/                   # FastAPI backend
│   │   ├── app.py                # REST API
│   │   ├── Dockerfile
│   │   └── model/                # Packaged model
│   └── frontend/                 # Streamlit UI
│       ├── app.py
│       └── Dockerfile
├── mlruns/                       # MLflow tracking data
├── docker-compose.yml            # Container orchestration
├── requirements.txt              # Dependencies
├── setup.py                      # Package configuration
└── README.md
```

## Tech Stack

| Category | Technologies |
|----------|-------------|
| ML Framework | scikit-learn, MLflow |
| NLP | spaCy |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Database | MongoDB |
| Containerization | Docker, Docker Compose |
| Testing | pytest |
| CLI | Click |
| Logging | Loguru |

## License

This project is for educational and demonstration purposes.

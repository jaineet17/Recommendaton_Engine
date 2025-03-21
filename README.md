# Amazon Recommendation Engine

A scalable recommendation system for e-commerce products featuring multiple recommendation algorithms, real-time processing, and comprehensive monitoring.

## Overview

This project implements a complete recommendation system pipeline:

1. **Data Processing**: Ingest and prepare Amazon review data
2. **Model Training**: Train multiple recommendation models
3. **API Service**: Serve recommendations via a RESTful API
4. **Monitoring**: Track system performance metrics
5. **Real-time Processing**: Update recommendations based on user activity

## Quick Start for New Users

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- 8GB+ RAM for training models

### First-Time Setup

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/amazon-recommendation-engine.git
   cd amazon-recommendation-engine
   ```

2. Set up the environment
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up environment variables (optional)
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

3. Run the API server
   ```bash
   python run_api.py --debug
   ```

4. Test that everything works
   ```bash
   # Check API health
   curl http://localhost:5050/api/health
   
   # Get sample recommendations (if demo data is loaded)
   curl "http://localhost:5050/api/recommend/lightgcn/A1B2C3D4E5?count=5"
   ```

## Using Docker

For a containerized setup:

```bash
# Start all services in development mode
docker-compose up -d

# For production deployment
ENVIRONMENT=production docker-compose up -d
```

## Documentation

All documentation has been consolidated into a single comprehensive document:

- [Unified Documentation](docs/UNIFIED_DOCUMENTATION.md) - Complete guide to all aspects of the system

This document includes:
- Getting Started guide
- Project structure and architecture
- API reference
- Model training and evaluation
- Configuration and deployment
- Monitoring and performance tuning
- Project history

## Key Features

- **Multiple Recommendation Models**: LightGCN, NCF, Matrix Factorization, Content-Based, and Ensemble
- **Unified Configuration System**: Centralized configuration with environment-specific settings
- **Comprehensive Monitoring**: Prometheus metrics and Grafana dashboards
- **Dockerized Deployment**: Production-ready containerization

## Project Structure - Updated

```
recommendation_engine/
├── config/                 # Configuration files
├── data/                   # Data storage
│   ├── models/             # Trained model files
│   ├── processed/          # Processed data files
│   └── raw/                # Raw data files
├── docs/                   # Documentation
├── logs/                   # Log files
├── monitoring/             # Monitoring configuration
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── api/                # API implementation
│   ├── caching/            # Caching mechanisms
│   ├── data/               # Data processing
│   ├── evaluation/         # Model evaluation
│   ├── feedback/           # Feedback loop system (moved from root)
│   ├── frontend/           # Frontend server (consolidated)
│   ├── models/             # Recommendation models
│   └── utils/              # Utility modules
│       ├── config/         # Centralized configuration system
│       ├── logging_config.py # Logging configuration
│       └── utility.py      # Consolidated utility functions
├── tests/                  # Test suite
├── docker-compose.yml      # Docker setup
├── Dockerfile              # API service container
├── .env.example            # Example environment variables
├── requirements.txt        # Python dependencies
└── run_api.py              # API startup script
```

## Code Organization Principles

### 1. Module Organization

The codebase follows a modular organization pattern:

- **Core Components**: Each major functional area has its own directory under `src/`
- **Utility Functions**: Shared code is consolidated in `src/utils/utility.py`
- **Configuration**: A centralized configuration system in `src/utils/config/`
- **Logging**: Standardized logging setup in `src/utils/logging_config.py`

### 2. Feedback Loop System

The feedback loop system has been moved from the project root to `src/feedback/` for better organization. A compatibility layer is maintained in the root directory for backward compatibility.

### 3. Frontend Serving

The frontend server implementation has been consolidated into a single module with:  
- Auto-detection of the frontend directory
- Configurable port via environment variables or command line arguments
- Optional browser auto-launch
- Support for API routes

## Common Tasks

### Training Models

```bash
# Train the LightGCN model
python scripts/train_lightgcn.py --data-path data/processed/ratings.csv

# Train the NCF model
python scripts/train_ncf.py --data-path data/processed/ratings.csv
```

### Running the Full Pipeline

```bash
# Process data, train models, and start API
python scripts/run_full_pipeline.py --sample-size 10000
```

### Monitoring

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- MLflow: http://localhost:5001

## License

This project is licensed under the MIT License

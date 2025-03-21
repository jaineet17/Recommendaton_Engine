# Amazon Recommendation Engine - Unified Documentation

This is the complete documentation for the Amazon Recommendation Engine project, consolidating all information into a single reference document.

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure and Map](#project-structure-and-map)
- [Core Documentation](#core-documentation)
- [Port Configuration](#port-configuration)
- [Project History](#project-history)
  - [Change Log](#change-log)
  - [Cleanup Log](#cleanup-log)

---

# Getting Started

This guide will walk you through the process of setting up and running the Amazon Recommendation Engine for the first time. Follow these steps to get the system up and running quickly.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.9 or higher
- Git
- Docker and Docker Compose (for containerized deployment)
- 8GB+ RAM (especially for training models)

## Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/amazon-recommendation-engine.git
cd amazon-recommendation-engine
```

## Step 2: Use the Setup Script

The project includes a convenient setup script that helps you get started quickly:

```bash
# Make the script executable (if it's not already)
chmod +x setup.py

# Run the script with the --setup flag to set up the environment
./setup.py --setup
```

This will:
- Create a Python virtual environment
- Install all required dependencies
- Create an `.env` file from the example template

## Step 3: Check Docker Setup (Optional, for containerized deployment)

If you plan to use Docker for deployment, verify your Docker setup:

```bash
./setup.py --docker
```

## Step 4: Check Port Availability

Before starting the services, you can check if the required ports are available:

```bash
# Check all ports
./check_ports.py

# Skip checking Docker service ports
./check_ports.py --skip-docker
```

This will ensure that there are no port conflicts that might prevent the services from starting correctly.

## Step 5: Start the API Server

You can start the API server either directly or using Docker:

### Option A: Start the API Server Directly

```bash
# Start the API in debug mode
./setup.py --start-api --debug
```

### Option B: Start with Docker Compose

```bash
# For development environment
./setup.py --docker-start dev

# For production environment
./setup.py --docker-start production
```

## Step 6: Verify the API is Running

Open your browser and navigate to:

```
http://localhost:5050/api/health
```

You should see a health check response indicating the API is running properly.

## Running the Recommendation Engine

### Basic Demo

If demo data is loaded, you can test getting recommendations with:

```bash
curl "http://localhost:5050/api/recommend/lightgcn/A1B2C3D4E5?count=5"
```

### Running the Full Pipeline

To process data, train models, and start the API:

```bash
python scripts/run_full_pipeline.py --sample-size 10000
```

## Monitoring (When Using Docker)

The system includes comprehensive monitoring with Prometheus and Grafana:

- Prometheus UI: http://localhost:9090
- Grafana Dashboards: http://localhost:3000 (default credentials: admin/admin)
- MLflow Tracking: http://localhost:5001

## Troubleshooting

### Common Issues

1. **Port conflicts**: If any services fail to start, check if the ports are already in use by other applications. Use the `./check_ports.py` tool to identify conflicts and follow the recommendations.

2. **Memory issues during model training**: Try reducing the sample size or increasing the available memory.

3. **Docker container startup failure**: Check Docker logs with:
   ```bash
   docker-compose logs
   ```

4. **Missing data files**: Ensure all required data files are in the correct locations.

---

# Project Structure and Map

This section provides a comprehensive overview of the Amazon Recommendation Engine project structure, code organization, and the relationships between different components.

## Project Directory Structure

```
recommendation_engine/
├── config/                 # Configuration files
│   ├── config.yaml         # Base configuration
│   ├── development.yaml    # Development environment overrides
│   ├── production.yaml     # Production environment overrides
│   └── prometheus/         # Prometheus monitoring config
├── data/                   # Data storage
│   ├── models/             # Trained model files
│   ├── processed/          # Processed data files
│   └── raw/                # Raw data files
├── docs/                   # Documentation
│   └── UNIFIED_DOCUMENTATION.md  # This file
├── logs/                   # Log files
├── monitoring/             # Monitoring configuration
│   ├── grafana/            # Grafana dashboards
│   └── prometheus/         # Prometheus config
├── scripts/                # Utility scripts
│   ├── train_*.py          # Model training scripts
│   └── run_*.py            # Pipeline scripts
├── src/                    # Source code
│   ├── api/                # API implementation
│   ├── config/             # Configuration system
│   ├── data/               # Data processing modules
│   ├── models/             # Recommendation models
│   ├── kafka/              # Event processing
│   ├── utils/              # Utility modules
│   └── feature_store/      # Feature store components
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── docker-compose.yml      # Docker setup
├── Dockerfile              # API service container
├── .env.example            # Example environment variables
├── requirements.txt        # Python dependencies
├── run_api.py              # API startup script
└── README.md               # Project overview
```

## System Architecture

The system is built with a modular architecture:

```
                             ┌─────────────┐
                             │   Frontend  │
                             │  (Optional) │
                             └──────┬──────┘
                                    │
                                    ▼
┌─────────────┐              ┌─────────────┐              ┌─────────────┐
│   Kafka     │◄─────────────┤     API     │◄─────────────┤  Database   │
│  (Events)   │              │   Service   │              │ (PostgreSQL)│
└─────────────┘              └──────┬──────┘              └─────────────┘
                                    │
                                    ▼
                             ┌─────────────┐              ┌─────────────┐
                             │    Models   │              │  Monitoring │
                             │  (Storage)  │              │(Prometheus) │
                             └─────────────┘              └─────────────┘
```

## Core Components

### API Service

The API service is the primary interface for recommendation requests. It's implemented using Flask and handles:

- User requests for recommendations
- Loading and serving trained models
- Logging recommendation events
- Health checking and monitoring

Key files:
- `src/api/app.py`: Main API implementation
- `run_api.py`: Server startup script

### Recommendation Models

The system supports multiple recommendation algorithms:

1. **LightGCN**: Graph Convolutional Network for collaborative filtering
2. **Neural Collaborative Filtering (NCF)**: Deep learning model for user-item interactions
3. **Matrix Factorization**: Classic recommendation approach
4. **Content-Based**: Recommends similar products based on features
5. **Hybrid Ensemble**: Combines multiple models for better recommendations

Key files:
- `src/models/lightgcn.py`: LightGCN implementation
- `src/models/neural_collaborative_filtering.py`: NCF implementation
- `src/models/hybrid_ensemble.py`: Ensemble model logic

### Data Processing

Data processing components handle loading, cleaning, and preparing data for the recommendation models:

- `src/data/preprocess.py`: Data preprocessing utilities
- `src/data/database.py`: Database connection and ORM models
- `src/feature_store/feature_store.py`: Feature storage and retrieval

### Event Processing

Real-time events are processed through a Kafka-based pipeline:

- `src/kafka/producer.py`: Produces events to Kafka topics
- `src/kafka/consumer.py`: Consumes events from Kafka topics
- `src/kafka/stream_processor.py`: Processes event streams

### Configuration System

The configuration system provides a unified approach to settings:

- `src/config/config.py`: Configuration loading and validation
- `config/*.yaml`: Environment-specific configuration files

## Development Workflow

For development:

1. Run `./setup.py --setup` to set up the environment
2. Make code changes
3. Run tests with `pytest tests/`
4. Start the API in debug mode with `./setup.py --start-api --debug`
5. Use Docker for testing the full infrastructure

---

# Core Documentation

This section provides comprehensive documentation for the Amazon Recommendation Engine system.

## API Reference

### Recommendation Endpoints

#### Get Recommendations for a User

```
GET /api/recommend/<model_name>/<user_id>?count=<num_recommendations>
```

Parameters:
- `model_name`: The model to use (lightgcn, ncf, mf, content_based, hybrid)
- `user_id`: The user ID to get recommendations for
- `count`: (Optional) Number of recommendations to return (default: 10)

Example:
```
GET /api/recommend/lightgcn/A1B2C3D4E5?count=5
```

Response:
```json
{
  "recommendations": [
    {
      "product_id": "B00X4WHP5E",
      "score": 0.92,
      "title": "Product Name",
      "category": "Electronics"
    },
    // More recommendations...
  ],
  "model": "lightgcn",
  "user_id": "A1B2C3D4E5",
  "timestamp": "2023-03-15T12:34:56.789Z"
}
```

#### Get Similar Products

```
GET /api/similar-products/<model_name>/<product_id>?count=<num_recommendations>
```

Parameters:
- `model_name`: The model to use (lightgcn, content_based, hybrid)
- `product_id`: The product ID to find similar items for
- `count`: (Optional) Number of similar products to return (default: 10)

Example:
```
GET /api/similar-products/content_based/B00X4WHP5E?count=5
```

Response:
```json
{
  "similar_products": [
    {
      "product_id": "B01A7S3X9Y",
      "score": 0.87,
      "title": "Similar Product Name",
      "category": "Electronics"
    },
    // More products...
  ],
  "model": "content_based",
  "product_id": "B00X4WHP5E",
  "timestamp": "2023-03-15T12:34:56.789Z"
}
```

### Health and Monitoring Endpoints

#### Health Check

```
GET /api/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.2.3",
  "models_loaded": ["lightgcn", "ncf", "mf", "content_based"],
  "uptime": 1234.56
}
```

#### Metrics

```
GET /api/metrics
```

Response: Prometheus-formatted metrics

#### List Available Models

```
GET /api/models
```

Response:
```json
{
  "models": [
    {
      "name": "lightgcn",
      "type": "graph",
      "version": "1.0.0",
      "training_date": "2023-03-10T00:00:00Z"
    },
    // More models...
  ]
}
```

## Model Training

### Training a New Model

To train a new model, use the training scripts in the `scripts` directory:

```bash
# Train LightGCN model
python scripts/train_lightgcn.py --data-file data/processed/amazon_reviews_processed.parquet --output-dir data/models

# Train NCF model
python scripts/train_ncf.py --data-file data/processed/amazon_reviews_processed.parquet --output-dir data/models

# Train Content-Based model
python scripts/train_content_based.py --data-file data/processed/amazon_reviews_processed.parquet --output-dir data/models
```

### Training Parameters

Common parameters for all training scripts:

- `--data-file`: Path to the processed data file
- `--output-dir`: Directory to save the trained model
- `--epochs`: Number of training epochs (default varies by model)
- `--batch-size`: Batch size for training (default varies by model)
- `--learning-rate`: Learning rate for optimization (default varies by model)
- `--embedding-dim`: Dimension of embeddings (default varies by model)

### Evaluating Models

To evaluate a trained model:

```bash
python scripts/test_models.py --model-path data/models/lightgcn_v1.0.0.pkl --test-data data/processed/test_data.parquet
```

## Data Processing

### Preprocessing Raw Data

To preprocess the raw Amazon review data:

```bash
python scripts/preprocess_data.py --input-file data/raw/amazon_reviews.csv --output-file data/processed/amazon_reviews_processed.parquet
```

Parameters:
- `--input-file`: Path to the raw data file
- `--output-file`: Path to save the processed data
- `--min-reviews`: Minimum number of reviews per user/item to keep (default: 5)
- `--sample-size`: Number of reviews to sample (default: all)

### Creating Sample Data

For testing and development, you can create sample data:

```bash
python scripts/create_sample_data.py --num-users 1000 --num-products 5000 --output-file data/processed/sample_data.parquet
```

## Running in Production

### Docker Deployment

For production deployment using Docker:

```bash
# Start all services in production mode
ENVIRONMENT=production docker-compose up -d
```

### Environment Variables

Key environment variables for production:

- `ENVIRONMENT`: Set to `production` for production settings
- `API_PORT`: Port for the API server (default: 5050)
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`: Database connection details
- `KAFKA_BOOTSTRAP_SERVERS`: Kafka connection details
- `MODEL_PATH`: Path to trained models
- `LOG_LEVEL`: Logging level (default: INFO)

### Performance Tuning

For better performance in production:

1. Increase the number of API workers:
   ```
   API_WORKERS=8 python run_api.py
   ```

2. Use a production WSGI server like Gunicorn:
   ```
   gunicorn --workers=4 --bind=0.0.0.0:5050 "src.api.app:create_app()"
   ```

3. Enable Redis caching:
   ```
   REDIS_URL=redis://localhost:6379/0 python run_api.py
   ```

4. Consider model quantization for faster inference:
   ```
   python scripts/quantize_models.py --model-dir data/models
   ```

## Monitoring and Metrics

### Prometheus Metrics

Key metrics available via Prometheus:

- `recommendation_count`: Counter of recommendations served
- `recommendation_latency`: Histogram of recommendation latency
- `model_load_time`: Gauge of model loading time
- `api_requests_total`: Counter of API requests
- `api_request_duration_seconds`: Histogram of API request duration

### Grafana Dashboards

The system includes pre-configured Grafana dashboards:

1. **API Dashboard**: Overview of API performance and request patterns
2. **Model Dashboard**: Model performance and predictions
3. **System Dashboard**: System resource utilization

### Alerting

Configure alerting in Prometheus/Grafana for:

- High API latency
- Service unavailability
- High error rates
- Resource constraints (memory, CPU)

---

# Port Configuration

## Overview

This section provides a comprehensive guide to port configuration in the Amazon Recommendation Engine project, including current status, changes made, and usage instructions.

## Port Configuration Summary

| Service | Default Port | Configured In | Environment Variable | 
|---------|-------------|---------------|----------------------|
| API Server | 5050 | config/config.yaml, run_api.py | API_PORT | 
| Frontend Server | 8080 | src/frontend/server.py | FRONTEND_PORT | 
| Metrics Endpoint | 8001 | config/config.yaml | PROMETHEUS_METRICS_PORT | 
| PostgreSQL | 5432 | docker-compose.yml, config/config.yaml | POSTGRES_PORT | 
| Redis | 6379 | docker-compose.yml, config/config.yaml | REDIS_PORT | 
| Prometheus | 9090 | docker-compose.yml | PROMETHEUS_PORT | 
| Grafana | 3000 | docker-compose.yml | GRAFANA_PORT | 
| Kafka | 9092 | docker-compose.yml | KAFKA_PORT | 
| Zookeeper | 2181 | docker-compose.yml | ZOOKEEPER_PORT | 
| MLFlow | 5001 | docker-compose.yml | MLFLOW_PORT | 

## Improvements Implemented

### 1. Port Availability Checker Tool

A utility script (`check_ports.py`) has been developed that:
- Checks the availability of all service ports
- Provides a formatted summary of port status
- Identifies conflicts and recommends solutions
- Can be run independently or integrated into other scripts

### 2. Enhanced API Server Port Management

The `run_api.py` script checks port availability before starting:

```python
# Check if the port is available
if not is_port_available(args.host, args.port):
    original_port = args.port
    args.port = find_available_port(args.host, args.port)
    logging.warning(f"Port {original_port} is not available. Using port {args.port} instead.")
```

### 3. Standardized Frontend Port Configuration

The frontend server uses the `FRONTEND_PORT` environment variable consistently with a default of 8080.

### 4. Consistent Environment Configuration

A `.env` file provides configuration for all port settings:

```
# Frontend Configuration
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=8080
```

## Usage Instructions

### Running the Port Checker

To check if all required ports are available:

```bash
# Check all ports
./check_ports.py

# Skip checking Docker service ports
./check_ports.py --skip-docker

# Check ports on a specific host
./check_ports.py --host 127.0.0.1
```

### Starting the API with Port Conflict Handling

The API server will now:
1. Check if the configured port is available
2. Log a warning if the port is in use
3. Automatically select an available port
4. Start the server on the new port

## Best Practices

1. **Run Port Checker Before Startup**: Incorporate the port checker into your startup scripts to detect conflicts early.

2. **Use Environment Variables**: Always set ports through environment variables or the `.env` file rather than hardcoding them.

3. **CI/CD Integration**: Integrate the port checker into CI/CD pipelines to detect conflicts early.

---

# Project History

## Change Log

### Version 1.3.0 (2023-03-15)

#### Added
- Hybrid ensemble model that combines LightGCN and Content-Based recommendations
- Real-time recommendation updates based on user activity
- A/B testing support for comparing model performance
- New API endpoints for similar product recommendations
- Prometheus monitoring integration
- User event tracking
- Support for model versioning

#### Changed
- Improved recommendation caching mechanism
- Enhanced logging format with request IDs
- Optimized model loading for faster startup
- Updated configuration system to support environment overrides

#### Fixed
- Memory leak in recommendation serving
- Incorrect score normalization in the NCF model
- Cached recommendations not updating properly
- Docker container startup issues

### Version 1.2.0 (2023-02-10)

#### Added
- Neural Collaborative Filtering (NCF) model
- Content-based recommendation model
- User history endpoint
- Docker Compose configuration
- Redis caching integration
- Grafana dashboards for monitoring

#### Changed
- Improved data preprocessing pipeline
- Enhanced API error handling
- Expanded API documentation
- Optimized LightGCN model performance

### Version 1.1.0 (2023-01-05)

#### Added
- LightGCN recommendation model
- Basic API for serving recommendations
- Configuration system
- Data processing utilities
- Simple model evaluation framework

#### Changed
- Refactored project structure
- Improved documentation
- Enhanced command-line interfaces for scripts

### Version 1.0.0 (2022-12-01)

#### Added
- Initial release with Matrix Factorization model
- Basic Flask API
- Data loading utilities
- Project documentation

## Cleanup Log

### Cleanup Operations (2023-03-20)

#### Removed Cleaning Scripts
The following cleaning scripts were removed from the root directory:
- `cleanup.sh`
- `run_cleanup.sh`
- `src_cleanup.py`
- `src_cleanup_fixed.py`

#### Removed Testing Scripts
The following testing scripts were removed from the root directory:
- `test_system.py`
- `load_test_amazon_100k.py`

#### Removed Documentation Files
The following documentation files were consolidated:
- `CLEANUP_SUMMARY.md`
- `CLEANUP_STATUS.md`
- `CLEANUP.md`
- `CODEBASE_CLEANUP_PLAN.md`
- `SRC_CLEANUP_REPORT.md`
- `SRC_CLEANUP_REPORT_FINAL.md`

#### Consolidated Models Directories
The redundant `models` directory in the root was removed, with all model files now stored in `data/models`.

#### Backup
All removed files were backed up to `cleanup_scripts_backup` directory before removal.

### API Cleanup (2023-03-15)

#### Removed Deprecated Endpoints
- `/api/v1/*` endpoints (now replaced with `/api/*`)
- `/api/recommendations` (replaced with `/api/recommend`)
- `/internal/*` endpoints (merged with main API)

#### Removed Unused Code
- Legacy model loaders in `src/api/model_utils.py`
- Deprecated error handlers
- Unused middleware components

#### Consolidated API Files
- Combined multiple route files into a single `app.py`
- Moved helper functions to `src/api/utils.py`
- Centralized error handling

### Database Cleanup (2023-03-10)

#### Removed Deprecated Tables
- `legacy_recommendations`
- `temp_users`
- `old_events`

#### Optimized Database Schema
- Added indexes for frequently queried columns
- Removed redundant columns
- Normalized tables structure

### Code Refactoring (2023-03-05)

#### Removed Duplicate Code
- Consolidated duplicate utility functions
- Removed copy-pasted code blocks
- Created shared libraries for common functionality

#### Improved Error Handling
- Added try-except blocks with specific error handling
- Improved error messages and logging
- Added more comprehensive error reporting 
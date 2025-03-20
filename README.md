# Amazon Recommendation Engine

This project implements a comprehensive recommendation system for Amazon product recommendations, featuring state-of-the-art algorithms and a production-grade architecture.

## Overview

The Amazon Recommendation Engine is designed to provide personalized product recommendations based on user behavior and product attributes. The system integrates with Kafka for real-time event processing, uses PostgreSQL for data storage, and includes monitoring with Prometheus and Grafana.

## Features

- **Advanced Recommendation Algorithms**: Implements state-of-the-art algorithms like Neural Collaborative Filtering (NCF) and LightGCN
- **Real-time Processing**: Stream processors handle Kafka messages for real-time recommendations
- **Comprehensive Evaluation**: Tools for measuring recommendation quality with metrics like precision, recall, and NDCG
- **Scalable Architecture**: Designed for high throughput with proper database indices and efficient algorithms
- **Monitoring & Observability**: Integration with Prometheus and Grafana for system monitoring
- **API & Frontend Interface**: FastAPI-based REST API and lightweight web interface for interaction

## Architecture

The system is organized into the following layers:

1. **Data Layer**: Manages data storage and retrieval using PostgreSQL and Parquet files
2. **Model Layer**: Implements recommendation algorithms (NCF and LightGCN)
3. **API Layer**: Exposes RESTful endpoints for serving recommendations
4. **Stream Processing Layer**: Processes real-time user events using Kafka
5. **Monitoring Layer**: Tracks system performance using Prometheus and Grafana

![Architecture Diagram](docs/images/architecture.png)

### Stream Processing with Kafka

The recommendation system uses Apache Kafka for real-time event processing:

- **User Interaction Events**: View, click, and purchase events are captured and sent to Kafka topics
- **Real-time Recommendations**: The stream processor consumes events and updates recommendations in real-time
- **Model Performance Monitoring**: Metrics are collected and published to Kafka for monitoring

Key Kafka components:

- **Topics**: Dedicated topics for different event types (reviews, product views, purchases)
- **Producers**: Components that publish events to Kafka topics
- **Consumers**: Components that process events from Kafka topics
- **Stream Processors**: Stateful processors that transform event streams into recommendations

## Recommendation Algorithms

### Neural Collaborative Filtering (NCF)

Neural Collaborative Filtering combines matrix factorization with neural networks to capture non-linear user-item interactions. It outperforms traditional matrix factorization methods by leveraging deep learning to model complex patterns.

Key features:
- Neural network architecture with embedding layers
- Combination of Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
- Support for implicit and explicit feedback
- Negative sampling for improved training
- Early stopping to prevent overfitting

### LightGCN

LightGCN is a simplified graph convolutional network specifically designed for recommendations. It leverages the user-item interaction graph structure to learn high-quality representations while being more efficient than traditional GCNs.

Key features:
- Light-weight graph convolution operations without feature transformation and non-linear activation
- Layer combination for improved performance
- Neighborhood aggregation to capture collaborative signals
- BPR loss function for ranking optimization
- Efficient implementation for large-scale recommendation

## Getting Started

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- PyTorch (for model training)
- Recommended: CUDA-capable GPU for faster training

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/amazon-recommendation-engine.git
cd amazon-recommendation-engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the infrastructure services:
```bash
docker-compose up -d
```

4. Initialize the database:
```bash
python -m src.data.database
```

### Training Models

For CPU training (small datasets):
```bash
python -m src.models.train --model-type ncf --data-path data/processed/sample.parquet
```

For GPU training (large datasets):
```bash
# Navigate to the notebooks directory
cd notebooks

# Start Jupyter Notebook
jupyter notebook

# Open and run one of the training notebooks:
# - train_ncf_model.ipynb
# - train_lightgcn_model.ipynb
```

### Making Recommendations

```python
from src.models.model_factory import model_factory

# Load a trained model
model = model_factory.load_model("models/ncf_model_production.pkl")

# Get recommendations for a user
recommendations = model.predict(user_id="user_123", n=10)

# Print recommendations
for item_id, score in recommendations:
    print(f"Item: {item_id}, Score: {score:.4f}")
```

## Project Structure

```
amazon-recommendation-engine/
├── config/                  # Configuration files
├── data/                    # Data directory
│   ├── raw/                 # Raw data files
│   ├── processed/           # Processed data files
│   └── interim/             # Intermediate data files
├── models/                  # Saved model files
├── notebooks/               # Jupyter notebooks
│   ├── train_ncf_model.ipynb     # NCF training notebook
│   └── train_lightgcn_model.ipynb # LightGCN training notebook
├── scripts/                 # Utility scripts
├── src/                     # Source code
│   ├── api/                 # API endpoints
│   ├── data/                # Data processing modules
│   ├── kafka/               # Kafka integration
│   ├── models/              # Recommendation models
│   └── utils/               # Utility functions
├── tests/                   # Test modules
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Training Deep Learning Models

The deep learning-based recommendation models (NCF and LightGCN) are implemented in PyTorch and can be trained using either the provided Python modules or the Jupyter notebooks.

For training on cloud GPUs:
1. Use the provided notebooks in the `notebooks/` directory
2. Set up a cloud environment with GPU support (AWS, GCP, etc.)
3. Train the models using the notebooks
4. Download the trained models for use in the production system

## Performance Evaluation

The recommendation models are evaluated using several metrics:

- **RMSE/MAE**: For rating prediction accuracy
- **Precision@k and Recall@k**: For recommendation relevance
- **NDCG@k**: For ranking quality
- **Coverage**: For recommendation diversity
- **Latency**: For system responsiveness

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The NCF implementation is based on the paper "Neural Collaborative Filtering" by He et al. (2017)
- The LightGCN implementation is based on the paper "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" by He et al. (2020)

### Running the API and Frontend

To start both the API server and the frontend:

```bash
# Run both API and frontend
python scripts/run_app.py

# Run only the API server
python scripts/run_app.py --api-only

# Run only the frontend server
python scripts/run_app.py --frontend-only

# Specify custom ports
python scripts/run_app.py --api-port 9000 --frontend-port 9080
```

Once running, you can access:
- API documentation: http://localhost:8000/docs
- Frontend interface: http://localhost:8080

### API Endpoints

The API provides the following main endpoints:

- `GET /health`: API health check
- `GET /system-info`: System information including available models
- `GET /models`: List available recommendation models
- `POST /recommendations`: Get recommendations for a user
- `GET /products/{product_id}`: Get product details
- `GET /users/{user_id}/history`: Get user interaction history

### Frontend Features

The web interface provides:

- Selection of recommendation models
- User-based recommendation queries
- Viewing and filtering of recommendations
- Exploration of user history
- Product detail viewing

## Production Deployment

### Docker Compose Setup

The system can be deployed using Docker Compose, which sets up:

- PostgreSQL database
- Kafka and Zookeeper
- Schema Registry
- Stream processor
- API service
- Frontend service
- Monitoring stack (Prometheus, Grafana)

### Deployment Instructions

To deploy the system in production mode:

1. Make sure Docker and Docker Compose are installed
2. Clone the repository
3. Configure environment variables in `.env` (copy from `.env.example`)
4. Run the deployment script:

```bash
python scripts/start_production.py
```

The script will:
- Start all required services
- Set up Kafka topics
- Initialize the database
- Verify service health

### Scaling Considerations

- **Horizontal Scaling**: The API and stream processor can be scaled horizontally using Docker Swarm or Kubernetes
- **Kafka Partitioning**: Kafka topics are partitioned to allow parallel processing
- **Database Scaling**: PostgreSQL can be configured with read replicas for scaling reads

### Monitoring

The system includes comprehensive monitoring:

- **Prometheus**: Collects metrics from all components
- **Grafana**: Visualizes metrics in dashboards
- **Kafka Metrics**: Tracks message throughput, consumer lag, and more
- **Application Metrics**: Monitors recommendation quality, response times, and error rates

## Recent Improvements

1. Enhanced feedback loop for better recommendation updates
2. Improved error handling and connection management 
3. Fixed load testing script to handle 100k users
4. Optimized batch processing for large datasets
5. Better quality metrics for recommendation evaluation
6. Documentation for load testing process

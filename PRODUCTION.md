# Amazon Recommendation Engine - Production Guide

This document provides detailed instructions for deploying and operating the Amazon Recommendation Engine in a production environment, with a focus on the Kafka integration for real-time event processing.

## Production Architecture

The system follows a microservices architecture with the following components:

- **API Service**: Flask-based API for serving recommendations
- **Stream Processing Service**: Kafka-based service for real-time event processing 
- **PostgreSQL Database**: For storing user, product, and interaction data
- **Kafka Cluster**: For real-time event streaming
- **Frontend Service**: Web interface for demonstrating the recommendation system
- **Monitoring Stack**: Prometheus and Grafana for metrics and dashboards

## Kafka Integration

### Kafka Components

1. **Kafka Producers**:
   - `ReviewProducer`: Sends user reviews to Kafka
   - `ProductEventProducer`: Tracks user interactions with products
   - `RecommendationRequestProducer`: Logs recommendation requests
   - `MetricsProducer`: Publishes system and model metrics

2. **Kafka Consumers**:
   - `ReviewConsumer`: Processes incoming reviews
   - `ProductEventConsumer`: Processes user interaction events
   - `RecommendationRequestConsumer`: Handles recommendation requests
   - `MetricsConsumer`: Consumes metrics for monitoring

3. **Stream Processors**:
   - `UserEventProcessor`: Aggregates user behavior in real-time windows
   - `RealtimeRecommender`: Updates recommendations based on real-time events
   - `DataDriftDetector`: Monitors for data distribution changes

### Kafka Topics

The system uses the following Kafka topics:

- `amazon-user-events`: User events (logins, profile updates)
- `amazon-product-views`: Product view events
- `amazon-purchases`: Purchase events
- `amazon-reviews`: Product review events
- `amazon-recommendation-requests`: Requests for recommendations
- `amazon-recommendation-responses`: Recommendation responses
- `amazon-model-updates`: Model update events
- `amazon-model-metrics`: Model performance metrics
- `amazon-system-metrics`: System performance metrics
- `amazon-data-drift`: Data drift alerts

### Real-time Recommendation Flow

1. User interactions are captured by the frontend and API
2. Events are sent to Kafka using the appropriate producer
3. Stream processors consume events to update user profiles and preferences
4. Real-time recommender generates updated recommendations
5. Updated recommendations are stored in the database and cache
6. API serves the latest recommendations to users

## Deployment

### Prerequisites

- Docker and Docker Compose
- PostgreSQL 14+
- Apache Kafka 3.1+
- Python 3.9+

### Initial Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and configure environment variables
3. Run the deployment script:

```bash
python scripts/start_production.py
```

### Manual Deployment Steps

If you prefer to deploy manually:

1. Start the PostgreSQL database:
```bash
docker-compose up -d postgres
```

2. Start the Kafka cluster:
```bash
docker-compose up -d zookeeper kafka schema-registry kafka-connect
```

3. Set up Kafka topics:
```bash
python scripts/setup_kafka_topics.py
```

4. Initialize the database:
```bash
python -m src.data.database
```

5. Start the API and stream processor services:
```bash
docker-compose up -d api stream-processor
```

6. Start the frontend and monitoring services:
```bash
docker-compose up -d frontend prometheus grafana
```

## Monitoring and Maintenance

### Health Checks

- API Health: `http://localhost:5050/api/health`
- Kafka Control Center: `http://localhost:9021`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

### Common Operations

#### Scaling Services

To scale a service:

```bash
docker-compose up -d --scale api=3
```

#### Updating Models

1. Train a new model version
2. Place the model files in the `models` directory
3. Reload the models via the API:
```bash
curl -X POST http://localhost:5050/api/reload-models
```

#### Clearing Cache

```bash
curl -X POST http://localhost:5050/api/clear-cache
```

### Troubleshooting

#### Kafka Issues

- Check Kafka logs: `docker logs amazon-rec-kafka`
- Verify topics: `docker exec amazon-rec-kafka kafka-topics --list --bootstrap-server localhost:9092`
- Check consumer groups: `docker exec amazon-rec-kafka kafka-consumer-groups --bootstrap-server localhost:9092 --list`

#### Database Issues

- Check PostgreSQL logs: `docker logs amazon-rec-postgres`
- Verify connectivity: `docker exec amazon-rec-postgres pg_isready -U postgres`

#### Stream Processor Issues

- Check stream processor logs: `docker logs amazon-rec-stream-processor`
- Restart the stream processor: `docker restart amazon-rec-stream-processor`

## Performance Tuning

### Kafka Tuning

- Adjust the number of partitions for high-volume topics
- Configure retention policies based on data volume
- Tune producer batch settings for throughput

### API Service Tuning

- Adjust worker count based on expected load
- Configure recommendation cache size and TTL
- Implement rate limiting for high-traffic endpoints

### Stream Processor Tuning

- Adjust window sizes for event processing
- Configure batch processing settings
- Optimize resource allocation based on event volume

## Security Considerations

### Network Security

- Use TLS for all Kafka connections in production
- Configure proper network isolation between services
- Implement API authentication and authorization

### Data Security

- Encrypt sensitive data in Kafka topics
- Implement proper access controls for the database
- Regularly rotate credentials and keys

## Backup and Recovery

### Data Backup

- Set up regular PostgreSQL database backups
- Configure Kafka Connect for topic backup
- Archive model files regularly

### Disaster Recovery

- Document recovery procedures for each component
- Implement automated health checks and alerts
- Test recovery procedures regularly

## Conclusion

This production setup provides a scalable, real-time recommendation system that can handle high volumes of user interactions and deliver personalized recommendations with low latency. The Kafka integration enables event-driven architecture and real-time processing, making the system responsive to user behavior changes. 
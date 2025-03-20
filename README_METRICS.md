# Recommendation Metrics Visualization

This document outlines how to use and interpret the recommendation metrics visualization in Grafana.

## Overview

The recommendation system collects and exposes various metrics that help evaluate the quality and performance of recommendations. These metrics are visualized in a Grafana dashboard for easy monitoring.

## Setup

1. Ensure all services are running via Docker Compose:
   ```bash
   docker-compose up -d
   ```

2. Access Grafana at http://localhost:3000 (username: admin, password: admin)

3. Navigate to the "Recommendation System Metrics" dashboard

## Metrics Explanation

### Performance Metrics

- **Average Response Time**: The average time taken to generate recommendations, measured in seconds.
- **Throughput**: Number of recommendations served per second.

### Quality Metrics

- **Precision@K**: The proportion of recommended items that are relevant to the user.
  - Precision@5: Calculated for top 5 recommendations
  - Precision@10: Calculated for top 10 recommendations

- **Recall@K**: The proportion of relevant items that are successfully recommended.
  - Recall@5: Calculated for top 5 recommendations
  - Recall@10: Calculated for top 10 recommendations

- **Diversity**: Measures how varied the recommendations are across the product catalog.
  - Higher values indicate more diverse recommendations
  - Calculated as unique products recommended / total recommendations

- **Coverage**: The proportion of the product catalog that is being recommended across all users.
  - Higher values indicate the system is exposing more of the catalog
  - Calculated as unique products recommended / total products in catalog

- **Personalization**: Measures how different recommendations are between users.
  - Higher values indicate more personalized recommendations
  - Calculated by comparing the similarity of recommendation sets between users

## Metric Collection

Metrics are collected in two ways:

1. **API Endpoint**: The API exposes metrics at `/api/metrics` which provides real-time quality metrics.

2. **Metrics Exporter**: A dedicated service (metrics-exporter) collects metrics and exposes them to Prometheus in a standardized format.

## Interpreting Dashboard

- **Timeframe**: All metrics can be filtered by time range (last hour, day, week, etc.)
- **Refresh Rate**: Dashboard auto-refreshes every 5 seconds
- **Alert Thresholds**: Visual indicators show when metrics fall below acceptable values

## Troubleshooting

If metrics are not displaying:

1. Check that the metrics-exporter service is running: 
   ```bash
   docker ps | grep metrics-exporter
   ```

2. Verify Prometheus is receiving data:
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```

3. Check the API metrics endpoint is accessible:
   ```bash
   curl http://localhost:5050/api/metrics
   ```

## Extending Metrics

To add new metrics:

1. Update the API's `/api/metrics` endpoint in `src/api/app.py`
2. Add the metric to the Prometheus exporter in `export_metrics.py`
3. Update the Grafana dashboard JSON in `config/grafana/dashboards/recommendation_metrics.json` 
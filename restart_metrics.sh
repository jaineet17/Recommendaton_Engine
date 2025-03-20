#!/bin/bash

# Script to restart the monitoring system with metrics exporter

echo "Stopping Grafana and Prometheus containers..."
docker-compose stop grafana prometheus
sleep 2

echo "Removing Grafana and Prometheus containers..."
docker-compose rm -f grafana prometheus
sleep 2

echo "Creating necessary directories..."
mkdir -p config/grafana/dashboards
mkdir -p config/grafana/provisioning/dashboards
mkdir -p config/grafana/provisioning/datasources

echo "Starting Grafana, Prometheus, and metrics-exporter..."
docker-compose up -d grafana prometheus metrics-exporter

echo "Waiting for services to start..."
sleep 10

echo "Checking service status..."
docker-compose ps grafana prometheus metrics-exporter

echo "Verifying API metrics endpoint..."
curl -s http://localhost:5050/api/metrics | jq

echo "Setup complete! Grafana dashboard is available at http://localhost:3000"
echo "Login with username: admin, password: admin"
echo "Navigate to Dashboards -> Recommendation System Metrics" 
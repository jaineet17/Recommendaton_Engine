#!/bin/bash

# Start System Script
# Starts all required services for the Amazon Recommendation Engine, including MLFlow

set -e  # Exit on any error

echo "==========================================="
echo "Starting Amazon Recommendation Engine System"
echo "==========================================="

# Function to check if a port is in use
check_port() {
    local port=$1
    local service=$2
    
    # Check if the port is already in use
    if lsof -i:$port > /dev/null 2>&1; then
        echo "✅ $service is already running on port $port"
        return 0
    else
        return 1
    fi
}

# Create necessary directories
create_directories() {
    echo "Creating necessary directories..."
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p models
    mkdir -p config/grafana/dashboards
    mkdir -p config/grafana/provisioning/dashboards
    mkdir -p config/grafana/provisioning/datasources
    mkdir -p monitoring/grafana
    mkdir -p monitoring/prometheus
    mkdir -p logs
}

# Start Kafka and related services if not already running
start_kafka() {
    if check_port 9092 "Kafka"; then
        return
    fi
    
    echo "Starting Kafka and related services..."
    docker-compose up -d zookeeper kafka schema-registry
    
    # Wait for Kafka to be ready
    echo "Waiting for Kafka to be ready..."
    for i in {1..30}; do
        if docker-compose exec kafka kafka-topics --bootstrap-server kafka:9092 --list > /dev/null 2>&1; then
            echo "✅ Kafka is ready"
            break
        fi
        echo -n "."
        sleep 2
        
        if [ $i -eq 30 ]; then
            echo "❌ Kafka did not start in time"
            exit 1
        fi
    done
}

# Start PostgreSQL if not already running
start_postgres() {
    if check_port 5432 "PostgreSQL"; then
        return
    fi
    
    echo "Starting PostgreSQL..."
    docker-compose up -d postgres
    
    # Wait for PostgreSQL to be ready
    echo "Waiting for PostgreSQL to be ready..."
    for i in {1..15}; do
        if docker-compose exec postgres pg_isready -U postgres > /dev/null 2>&1; then
            echo "✅ PostgreSQL is ready"
            break
        fi
        echo -n "."
        sleep 2
        
        if [ $i -eq 15 ]; then
            echo "❌ PostgreSQL did not start in time"
            exit 1
        fi
    done
}

# Start MLFlow if not already running
start_mlflow() {
    if check_port 5000 "MLFlow"; then
        return
    fi
    
    # Check if MinIO is running
    if ! check_port 9000 "MinIO"; then
        echo "Starting MinIO..."
        docker-compose up -d minio
        
        # Wait for MinIO to be ready
        echo "Waiting for MinIO to be ready..."
        for i in {1..15}; do
            if curl -s http://localhost:9000/minio/health/live > /dev/null 2>&1; then
                echo "✅ MinIO is ready"
                break
            fi
            echo -n "."
            sleep 2
            
            if [ $i -eq 15 ]; then
                echo "❌ MinIO did not start in time"
                exit 1
            fi
        done
        
        # Initialize MinIO
        docker-compose up -d minio-init
    fi
    
    # Start MLFlow
    echo "Starting MLFlow..."
    docker-compose up -d mlflow
    
    # Wait for MLFlow to be ready
    echo "Waiting for MLFlow to be ready..."
    for i in {1..15}; do
        if curl -s http://localhost:5000/api/2.0/mlflow/experiments/list > /dev/null 2>&1; then
            echo "✅ MLFlow is ready"
            break
        fi
        echo -n "."
        sleep 2
        
        if [ $i -eq 15 ]; then
            echo "❌ MLFlow did not start in time"
            # Continue anyway as it might just be slow to start
        fi
    done
}

# Start Prometheus if not already running
start_prometheus() {
    if check_port 9090 "Prometheus"; then
        return
    fi
    
    echo "Starting Prometheus..."
    docker-compose up -d prometheus
}

# Start Grafana if not already running
start_grafana() {
    if check_port 3000 "Grafana"; then
        return
    fi
    
    echo "Starting Grafana..."
    docker-compose up -d grafana
}

# Start API if not already running
start_api() {
    if check_port 5050 "API"; then
        return
    fi
    
    echo "Starting API..."
    # We'll run this directly instead of with docker-compose to have more control
    python -m src.api.app &
    
    # Wait for API to be ready
    echo "Waiting for API to be ready..."
    for i in {1..15}; do
        if curl -s http://localhost:5050/api/health > /dev/null 2>&1; then
            echo "✅ API is ready"
            break
        fi
        echo -n "."
        sleep 2
        
        if [ $i -eq 15 ]; then
            echo "❌ API did not start in time"
            # Continue anyway as the API might be slow to start
        fi
    done
}

# Start metrics exporter if not already running
start_metrics_exporter() {
    if check_port 8001 "Metrics Exporter"; then
        return
    fi
    
    echo "Starting Metrics Exporter..."
    python export_metrics.py --api-url http://localhost:5050/api &
    
    # Give it a second to start
    sleep 3
    
    if check_port 8001 "Metrics Exporter"; then
        echo "✅ Metrics Exporter is ready"
    else
        echo "❌ Metrics Exporter did not start properly"
        # Continue anyway
    fi
}

# Main
create_directories
start_postgres
start_kafka
start_mlflow
start_prometheus
start_grafana
start_api
start_metrics_exporter

echo ""
echo "==========================================="
echo "System Started Successfully"
echo "==========================================="
echo ""
echo "Services:"
echo "- API: http://localhost:5050"
echo "- Grafana: http://localhost:3000 (admin/admin)"
echo "- MLFlow: http://localhost:5000"
echo "- Prometheus: http://localhost:9090"
echo "- MinIO: http://localhost:9001 (minioadmin/minioadmin)"
echo ""
echo "To download the Amazon dataset and train models:"
echo "python scripts/download_and_train.py"
echo ""
echo "To check metrics:"
echo "curl http://localhost:5050/api/metrics"
echo ""
echo "To stop the system:"
echo "docker-compose down"
echo "===========================================" 
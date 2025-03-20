#!/bin/bash

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
until kafka-topics --bootstrap-server kafka:9092 --list 2>/dev/null; do
  echo "Kafka not yet ready, waiting 5 seconds..."
  sleep 5
done
echo "Kafka is ready!"

# Create topics
echo "Creating Kafka topics..."

# Raw reviews topic
kafka-topics --bootstrap-server kafka:9092 --create --if-not-exists \
  --topic amazon-reviews \
  --partitions 3 \
  --replication-factor 1 \
  --config retention.ms=2592000000

# Product events topic (views, purchases)
kafka-topics --bootstrap-server kafka:9092 --create --if-not-exists \
  --topic product-events \
  --partitions 3 \
  --replication-factor 1 \
  --config retention.ms=2592000000

# Recommendation requests topic
kafka-topics --bootstrap-server kafka:9092 --create --if-not-exists \
  --topic recommendation-requests \
  --partitions 2 \
  --replication-factor 1 \
  --config retention.ms=604800000

# Model metrics topic
kafka-topics --bootstrap-server kafka:9092 --create --if-not-exists \
  --topic model-metrics \
  --partitions 1 \
  --replication-factor 1 \
  --config retention.ms=7776000000

# System metrics topic
kafka-topics --bootstrap-server kafka:9092 --create --if-not-exists \
  --topic system-metrics \
  --partitions 1 \
  --replication-factor 1 \
  --config retention.ms=1209600000

echo "Kafka topics created successfully!"

# List all topics
echo "List of Kafka topics:"
kafka-topics --bootstrap-server kafka:9092 --list 
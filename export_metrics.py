#!/usr/bin/env python3
"""
Script to export recommendation metrics to Prometheus for visualization in Grafana.
"""

import os
import time
import argparse
import requests
import logging
import numpy as np
from prometheus_client import start_http_server, Gauge

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('metrics_exporter')

# Define Prometheus metrics
PRECISION_5 = Gauge('recommendation_precision_5', 'Precision at 5 recommendations')
PRECISION_10 = Gauge('recommendation_precision_10', 'Precision at 10 recommendations') 
RECALL_5 = Gauge('recommendation_recall_5', 'Recall at 5 recommendations')
RECALL_10 = Gauge('recommendation_recall_10', 'Recall at 10 recommendations')
DIVERSITY = Gauge('recommendation_diversity', 'Diversity of recommendations')
COVERAGE = Gauge('recommendation_coverage', 'Coverage of product catalog')
PERSONALIZATION = Gauge('recommendation_personalization', 'Personalization score')
RESPONSE_TIME = Gauge('recommendation_response_time', 'Average response time in seconds')

# Base API URL
BASE_API_URL = "http://localhost:5050/api"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export recommendation metrics to Prometheus')
    parser.add_argument('--port', type=int, default=8001, help='Port to expose metrics on')
    parser.add_argument('--interval', type=int, default=30, help='Metrics collection interval in seconds')
    parser.add_argument('--api-url', type=str, default=BASE_API_URL, help='Base API URL')
    return parser.parse_args()

def get_metrics():
    """Collect metrics from the recommendation API."""
    try:
        # Get system metrics
        response = requests.get(f"{BASE_API_URL}/metrics", timeout=10)
        metrics = response.json()
        
        # Update Prometheus metrics
        if 'precision@5' in metrics:
            PRECISION_5.set(metrics['precision@5'])
        if 'precision@10' in metrics:
            PRECISION_10.set(metrics['precision@10'])
        if 'recall@5' in metrics:
            RECALL_5.set(metrics['recall@5'])
        if 'recall@10' in metrics:
            RECALL_10.set(metrics['recall@10'])
        if 'diversity' in metrics:
            DIVERSITY.set(metrics['diversity'])
        if 'coverage' in metrics:
            COVERAGE.set(metrics['coverage'])
        if 'personalization' in metrics:
            PERSONALIZATION.set(metrics['personalization'])
        if 'avg_response_time' in metrics:
            RESPONSE_TIME.set(metrics['avg_response_time'])
        
        logger.info("Successfully updated metrics")
        return True
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        return False

def main():
    """Main entry point."""
    args = parse_args()
    
    # Start metrics server
    start_http_server(args.port)
    logger.info(f"Metrics server started on port {args.port}")
    
    # Initial metrics
    get_metrics()
    
    # Metrics loop
    while True:
        time.sleep(args.interval)
        get_metrics()

if __name__ == "__main__":
    main() 
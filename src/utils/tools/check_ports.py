#!/usr/bin/env python3
"""
Port Configuration Checker for Amazon Recommendation Engine

This script checks the availability of ports used by various services
in the Amazon Recommendation Engine and reports any conflicts.
"""

import os
import sys
import socket
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Key services and their default ports
SERVICES = {
    "API Server": 5050,
    "Frontend Server": 8080,
    "Metrics Endpoint": 8001,
    "PostgreSQL": 5432,
    "Redis": 6379,
    "Prometheus": 9090,
    "Grafana": 3000,
    "Kafka": 9092,
    "Zookeeper": 2181,
    "MLFlow": 5001
}

# Environment variables for each service
PORT_ENV_VARS = {
    "API Server": "API_PORT",
    "Frontend Server": "FRONTEND_PORT",
    "Metrics Endpoint": "PROMETHEUS_METRICS_PORT",
    "PostgreSQL": "POSTGRES_PORT",
    "Redis": "REDIS_PORT",
    "Prometheus": "PROMETHEUS_PORT",
    "Grafana": "GRAFANA_PORT",
    "Kafka": "KAFKA_PORT",
    "Zookeeper": "ZOOKEEPER_PORT",
    "MLFlow": "MLFLOW_PORT"
}

def is_port_available(host, port):
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except socket.error:
        return False

def check_port(service, port, host="0.0.0.0"):
    """Check if a port for a specific service is available."""
    logger.info(f"Checking {service} port {port}...")
    if is_port_available(host, port):
        logger.info(f"✓ Port {port} for {service} is available")
        return True
    else:
        logger.error(f"✗ Port {port} for {service} is already in use")
        return False

def check_all_ports(host="0.0.0.0", skip_docker=False):
    """Check all service ports."""
    results = {}
    conflicts = []
    
    # Load environment variables
    load_dotenv()
    
    for service, default_port in SERVICES.items():
        # Skip Docker services if requested
        if skip_docker and service in ["PostgreSQL", "Redis", "Prometheus", "Grafana", "Kafka", "Zookeeper", "MLFlow"]:
            logger.info(f"Skipping {service} (Docker service)")
            results[service] = "Skipped"
            continue
            
        # Get port from environment variable or use default
        env_var = PORT_ENV_VARS.get(service)
        port = int(os.environ.get(env_var, default_port))
        
        # Check port availability
        available = check_port(service, port, host)
        results[service] = "Available" if available else "In Use"
        
        if not available:
            conflicts.append((service, port))
    
    return results, conflicts

def print_summary(results, conflicts):
    """Print a summary of port availability."""
    print("\n" + "="*60)
    print(" PORT CONFIGURATION SUMMARY ".center(60, "="))
    print("="*60)
    
    print("\nService Port Status:")
    print("-" * 50)
    max_service_len = max(len(service) for service in results.keys())
    for service, status in results.items():
        print(f"{service.ljust(max_service_len)} | {status}")
    
    print("\nConflicts:")
    print("-" * 50)
    if conflicts:
        for service, port in conflicts:
            print(f"✗ {service} on port {port} has a conflict")
        print("\nRecommendations:")
        print("1. Check for running processes using these ports")
        print("2. Modify .env file to use different ports")
        print("3. Stop conflicting services before starting the recommendation engine")
    else:
        print("No port conflicts detected! All services can start normally.")
    
    print("\n" + "="*60 + "\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check port availability for Amazon Recommendation Engine services")
    parser.add_argument("--host", default="0.0.0.0", help="Host to check ports on")
    parser.add_argument("--skip-docker", action="store_true", help="Skip checking Docker service ports")
    args = parser.parse_args()
    
    try:
        results, conflicts = check_all_ports(args.host, args.skip_docker)
        print_summary(results, conflicts)
        
        # Return appropriate exit code
        if conflicts:
            return 1
        return 0
    except Exception as e:
        logger.error(f"Error checking ports: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
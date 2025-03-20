#!/usr/bin/env python3
"""
Script to start the Amazon Recommendation Engine in production mode.

This script:
1. Checks if all required services are running
2. Sets up Kafka topics if needed
3. Initializes the database if needed
4. Starts the API and stream processor
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Start the Amazon Recommendation Engine in production mode')
    parser.add_argument('--config-path', type=str, default='config/config.yaml',
                        help='Path to the config file')
    parser.add_argument('--no-kafka-check', action='store_true',
                        help='Skip Kafka availability check')
    parser.add_argument('--no-postgres-check', action='store_true',
                        help='Skip PostgreSQL availability check')
    parser.add_argument('--skip-topic-setup', action='store_true',
                        help='Skip Kafka topic setup')
    parser.add_argument('--skip-db-init', action='store_true',
                        help='Skip database initialization')
    parser.add_argument('--compose-file', type=str, default='docker-compose.yml',
                        help='Docker Compose file to use')
    parser.add_argument('--docker-env-file', type=str, default='.env',
                        help='Docker environment file')
    return parser.parse_args()

def check_docker():
    """Check if Docker is installed and running."""
    try:
        output = subprocess.check_output(['docker', 'info'], stderr=subprocess.STDOUT)
        logger.info("Docker is installed and running")
        return True
    except subprocess.CalledProcessError:
        logger.error("Docker is not running. Please start Docker")
        return False
    except FileNotFoundError:
        logger.error("Docker is not installed. Please install Docker")
        return False

def check_docker_compose():
    """Check if Docker Compose is installed."""
    try:
        output = subprocess.check_output(['docker-compose', '--version'], stderr=subprocess.STDOUT)
        logger.info("Docker Compose is installed")
        return True
    except subprocess.CalledProcessError:
        logger.error("Error checking Docker Compose version")
        return False
    except FileNotFoundError:
        logger.error("Docker Compose is not installed. Please install Docker Compose")
        return False

def check_services(compose_file):
    """Check if required services are defined in the Docker Compose file."""
    required_services = ['postgres', 'kafka', 'zookeeper', 'api', 'frontend', 'stream-processor']
    
    if not os.path.exists(compose_file):
        logger.error(f"Docker Compose file not found: {compose_file}")
        return False
    
    # Simple check for service names in the compose file
    with open(compose_file, 'r') as f:
        content = f.read()
    
    missing_services = []
    for service in required_services:
        if f"{service}:" not in content:
            missing_services.append(service)
    
    if missing_services:
        logger.error(f"Missing required services in {compose_file}: {', '.join(missing_services)}")
        return False
    
    logger.info(f"All required services found in {compose_file}")
    return True

def check_config_file(config_path):
    """Check if the config file exists."""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return False
    
    logger.info(f"Config file found: {config_path}")
    return True

def check_kafka_availability():
    """Check if Kafka is available."""
    try:
        # Use kafka-topics command to check if Kafka is available
        cmd = ['docker', 'exec', 'amazon-rec-kafka', 'kafka-topics', '--list', '--bootstrap-server', 'localhost:9092']
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10)
        logger.info("Kafka is available")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"Kafka is not available: {e}")
        return False

def check_postgres_availability():
    """Check if PostgreSQL is available."""
    try:
        # Use pg_isready to check if PostgreSQL is available
        cmd = ['docker', 'exec', 'amazon-rec-postgres', 'pg_isready', '-U', 'postgres']
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
        logger.info("PostgreSQL is available")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"PostgreSQL is not available: {e}")
        return False

def setup_kafka_topics():
    """Set up Kafka topics."""
    try:
        logger.info("Setting up Kafka topics...")
        # Wait for Kafka to be ready
        for _ in range(30):
            if check_kafka_availability():
                break
            logger.info("Waiting for Kafka to be ready...")
            time.sleep(2)
        else:
            logger.error("Kafka is not available after waiting")
            return False
        
        # Run the topic setup script
        cmd = ['docker', 'exec', 'amazon-rec-api', 'python', '-m', 'scripts.setup_kafka_topics']
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        logger.info("Kafka topics setup completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to set up Kafka topics: {e}")
        logger.error(f"Output: {e.output.decode('utf-8')}")
        return False

def init_database():
    """Initialize the database."""
    try:
        logger.info("Initializing database...")
        # Wait for PostgreSQL to be ready
        for _ in range(30):
            if check_postgres_availability():
                break
            logger.info("Waiting for PostgreSQL to be ready...")
            time.sleep(2)
        else:
            logger.error("PostgreSQL is not available after waiting")
            return False
        
        # Run the database initialization
        cmd = ['docker', 'exec', 'amazon-rec-api', 'python', '-m', 'src.data.database']
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        logger.info("Database initialization completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to initialize database: {e}")
        logger.error(f"Output: {e.output.decode('utf-8')}")
        return False

def start_services(compose_file, env_file=None):
    """Start all services using Docker Compose."""
    env_args = []
    if env_file and os.path.exists(env_file):
        env_args = ['--env-file', env_file]
    
    try:
        logger.info(f"Starting services using {compose_file}...")
        cmd = ['docker-compose', '-f', compose_file] + env_args + ['up', '-d']
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        logger.info("Services started")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start services: {e}")
        logger.error(f"Output: {e.output.decode('utf-8')}")
        return False

def check_service_status(compose_file):
    """Check the status of all services."""
    try:
        logger.info("Checking service status...")
        cmd = ['docker-compose', '-f', compose_file, 'ps']
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        logger.info(f"Service status:\n{output.decode('utf-8')}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to check service status: {e}")
        return False

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set the config path in environment
    os.environ['CONFIG_PATH'] = args.config_path
    
    # Check prerequisites
    if not check_docker():
        return 1
    
    if not check_docker_compose():
        return 1
    
    if not check_config_file(args.config_path):
        return 1
    
    if not check_services(args.compose_file):
        return 1
    
    # Start services
    if not start_services(args.compose_file, args.docker_env_file):
        return 1
    
    # Check service status
    if not check_service_status(args.compose_file):
        return 1
    
    # Set up Kafka topics if not skipped
    if not args.skip_topic_setup and not args.no_kafka_check:
        if not setup_kafka_topics():
            logger.warning("Failed to set up Kafka topics, but continuing...")
    
    # Initialize database if not skipped
    if not args.skip_db_init and not args.no_postgres_check:
        if not init_database():
            logger.warning("Failed to initialize database, but continuing...")
    
    logger.info("Amazon Recommendation Engine started successfully in production mode")
    
    # Print URL information
    logger.info("\nAccess points:")
    logger.info("- API: http://localhost:5050/api/health")
    logger.info("- Frontend: http://localhost:8000")
    logger.info("- Kafka Control Center: http://localhost:9021")
    logger.info("- Prometheus: http://localhost:9090")
    logger.info("- Grafana: http://localhost:3000 (admin/admin)")
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 
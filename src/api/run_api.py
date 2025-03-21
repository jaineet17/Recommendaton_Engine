#!/usr/bin/env python3
"""
Unified API Startup Script for Amazon Recommendation Engine

This script provides a unified way to start the recommendation API with:
- Centralized configuration system
- Enhanced logging
- Graceful error handling
- Support for different environments
- Mock Kafka/ZooKeeper when needed

Usage:
    python run_api_unified.py [--port PORT] [--host HOST] [--log-level LEVEL] [--config CONFIG]
"""

import os
import sys
import time
import signal
import logging
import argparse
import traceback
import socket
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
import subprocess
import threading

# Add the project root to the Python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# Import configuration and utilities
from src.utils.config import get_config
from src.utils.logging_utils import setup_logging

# Define constants
DEFAULT_CONFIG_PATH = "config/config.yaml"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5050

# Handle Kafka imports
try:
    from confluent_kafka import Producer, Consumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    # Mock Kafka classes if not available
    class MockProducer:
        def __init__(self, *args, **kwargs):
            pass
        def produce(self, *args, **kwargs):
            pass
        def flush(self, *args, **kwargs):
            pass
        def poll(self, *args, **kwargs):
            return 0
    
    class MockConsumer:
        def __init__(self, *args, **kwargs):
            pass
        def subscribe(self, *args, **kwargs):
            pass
        def poll(self, *args, **kwargs):
            return None
        def close(self, *args, **kwargs):
            pass

    Producer = MockProducer
    Consumer = MockConsumer
    print("Warning: Kafka packages not installed. Using mock implementations.")

# Function to check if a port is available
def is_port_available(host, port):
    """Check if a port is available for the API to use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except socket.error:
        return False

# Function to find an available port starting from the specified port
def find_available_port(host, start_port, max_tries=10):
    """Find an available port starting from start_port."""
    port = start_port
    for _ in range(max_tries):
        if is_port_available(host, port):
            return port
        port += 1
    raise RuntimeError(f"Could not find an available port after {max_tries} attempts")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Start the Amazon Recommendation API")
    parser.add_argument("--port", type=int, default=os.getenv("API_PORT", DEFAULT_PORT),
                      help="Port to run the API on")
    parser.add_argument("--host", type=str, default=os.getenv("API_HOST", DEFAULT_HOST),
                      help="Host to bind the API to")
    parser.add_argument("--config", type=str, default=os.getenv("CONFIG_FILE", DEFAULT_CONFIG_PATH),
                      help="Path to the configuration file")
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"),
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    return parser.parse_args()

@contextmanager
def graceful_shutdown():
    """Context manager for handling graceful shutdown on signals."""
    # Set up signal handlers
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    
    def handle_shutdown(signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        # Restore original handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
    
    # Set up handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    try:
        yield
    finally:
        # Restore original handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        print("Shutdown complete")

def run_api(args):
    """Initialize and run the API server."""
    try:
        # Set up signal handling
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
        signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))
        
        # Load configuration
        config = get_config(args.config)
        
        # Set up logging
        log_config = config.get("logging", {})
        log_file = log_config.get("file", "logs/api.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        setup_logging(
            level=args.log_level.upper(), 
            log_file=log_file,
            log_format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        
        # Log startup information
        logging.info("Starting Amazon Recommendation API")
        logging.info(f"Environment: {config.get('environment', 'development')}")
        logging.info(f"Host: {args.host}, Port: {args.port}")
        
        # Check if the port is available
        if not is_port_available(args.host, args.port):
            original_port = args.port
            args.port = find_available_port(args.host, args.port)
            logging.warning(f"Port {original_port} is not available. Using port {args.port} instead.")
        
        # Import API module here to ensure logging is configured first
        try:
            from src.api.app import create_app
            app = create_app(config)
            
            # Run the application
            if config.get("environment") == "development" or args.debug:
                logging.info("Running in development mode with Flask development server")
                app.run(host=args.host, port=args.port, debug=True)
            else:
                logging.info("Running in production mode")
                try:
                    import waitress
                    logging.info("Using Waitress production server")
                    waitress.serve(app, host=args.host, port=args.port, threads=config.get("api", {}).get("threads", 8))
                except ImportError:
                    logging.warning("Waitress not installed, falling back to Flask server")
                    app.run(host=args.host, port=args.port, debug=False)
                    
        except ImportError as e:
            logging.error(f"Failed to import API module: {e}")
            return 1
            
    except Exception as e:
        print(f"Error starting API: {e}")
        traceback.print_exc()
        return 1
        
    return 0

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup base configurations
    os.makedirs("logs", exist_ok=True)
    
    # Run the port checker first if available
    try:
        import importlib.util
        checker_spec = importlib.util.find_spec("check_ports")
        if checker_spec:
            from check_ports import check_port
            # Only check the API port, as that's what we're about to use
            port_available = check_port("API Server", args.port, args.host)
            if not port_available:
                print("WARNING: API port is already in use. The application will try to find an available port.")
        else:
            print("Port checker not found. Skipping port availability check.")
    except ImportError:
        # Continue if the port checker is not available
        print("Port checker module not available. Continuing without port checks.")
    
    # Run the API with graceful shutdown handling
    with graceful_shutdown():
        return run_api(args)

if __name__ == "__main__":
    sys.exit(main()) 
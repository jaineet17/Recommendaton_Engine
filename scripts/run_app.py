#!/usr/bin/env python3
"""
Run script for the Amazon Recommendation Engine.

This script starts both the API and frontend servers.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from multiprocessing import Process

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Default ports
API_PORT = 8000
FRONTEND_PORT = 8080

# Path to the scripts
API_SCRIPT = 'src/api/run.py'
FRONTEND_SCRIPT = 'src/frontend/server.py'

# Processes
api_process = None
frontend_process = None

def signal_handler(sig, frame):
    """Handle signals to stop the servers gracefully."""
    logger.info("Stopping servers...")
    
    # Stop the API server
    if api_process and api_process.is_alive():
        logger.info("Stopping API server...")
        api_process.terminate()
        api_process.join(timeout=5)
    
    # Stop the frontend server
    if frontend_process and frontend_process.is_alive():
        logger.info("Stopping frontend server...")
        frontend_process.terminate()
        frontend_process.join(timeout=5)
    
    logger.info("Servers stopped")
    sys.exit(0)

def start_api_server(port):
    """
    Start the API server.
    
    Args:
        port: Port to run the API server on
    """
    # Use subprocess to run the API server
    try:
        cmd = [sys.executable, API_SCRIPT]
        
        # Set environment variable for port
        env = os.environ.copy()
        env['API_PORT'] = str(port)
        
        proc = subprocess.Popen(cmd, env=env)
        proc.wait()
    except KeyboardInterrupt:
        # Handle gracefully
        pass
    except Exception as e:
        logger.error(f"Error starting API server: {e}")

def start_frontend_server(port):
    """
    Start the frontend server.
    
    Args:
        port: Port to run the frontend server on
    """
    # Use subprocess to run the frontend server
    try:
        cmd = [sys.executable, FRONTEND_SCRIPT, str(port)]
        proc = subprocess.Popen(cmd)
        proc.wait()
    except KeyboardInterrupt:
        # Handle gracefully
        pass
    except Exception as e:
        logger.error(f"Error starting frontend server: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run the Amazon Recommendation Engine')
    parser.add_argument('--api-port', type=int, default=API_PORT,
                        help=f'Port for the API server (default: {API_PORT})')
    parser.add_argument('--frontend-port', type=int, default=FRONTEND_PORT,
                        help=f'Port for the frontend server (default: {FRONTEND_PORT})')
    parser.add_argument('--api-only', action='store_true',
                        help='Run only the API server')
    parser.add_argument('--frontend-only', action='store_true',
                        help='Run only the frontend server')
    
    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the API server if requested
        global api_process
        if not args.frontend_only:
            logger.info(f"Starting API server on port {args.api_port}...")
            api_process = Process(target=start_api_server, args=(args.api_port,))
            api_process.start()
            time.sleep(1)  # Give the API server time to start
        
        # Start the frontend server if requested
        global frontend_process
        if not args.api_only:
            logger.info(f"Starting frontend server on port {args.frontend_port}...")
            frontend_process = Process(target=start_frontend_server, args=(args.frontend_port,))
            frontend_process.start()
        
        # Wait for the servers to finish
        if api_process:
            api_process.join()
        if frontend_process:
            frontend_process.join()
    
    except KeyboardInterrupt:
        # Handle gracefully using the signal handler
        pass
    except Exception as e:
        logger.error(f"Error: {e}")
        signal_handler(None, None)  # Stop servers
        sys.exit(1)

if __name__ == "__main__":
    main() 
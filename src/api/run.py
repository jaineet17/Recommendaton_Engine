"""
Script to run the FastAPI server for the Amazon recommendation system.
"""

import logging
import os
import sys

import uvicorn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.database import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
api_config = config.get('api', {})

def run_api_server():
    """Run the FastAPI server."""
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)
    log_level = api_config.get("log_level", "info").lower()
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=api_config.get("debug", False)
    )

if __name__ == "__main__":
    run_api_server() 
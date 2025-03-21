"""
Script to run the FastAPI server for the Amazon recommendation system.
(Compatibility Module)

This module has been replaced by run_api.py and is maintained for backward compatibility only.
"""

import warnings
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Issue deprecation warning
warnings.warn(
    "This module has been replaced by run_api.py. "
    "Please update your references accordingly.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the new location
from src.api.run_api import run_api, main

def run_api_server():
    """Run the FastAPI server for the recommendation system.
    
    This is a compatibility wrapper around the main function from run_api.py
    """
    # Parse default arguments for backward compatibility
    import argparse
    parser = argparse.ArgumentParser(description="Run the API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--log-level", default="info", help="Logging level")
    args = parser.parse_args()
    
    # Call the main function which will run the API
    return main()

if __name__ == "__main__":
    sys.exit(main()) 
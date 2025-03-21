"""
Frontend server for the Recommendation Engine.

This module provides a web server to serve the frontend application and static files.
Features:
- Auto-detection of the frontend directory
- Configurable port via environment variables or command line
- Optional browser auto-launch
- Support for handling API routes and static files
- Detailed logging
"""

import http.server
import socketserver
import os
import sys
import webbrowser
import logging
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Add project root to the Python path to resolve imports when run directly
project_root = Path(__file__).resolve().parents[2]  # Up two directories from /src/frontend/
sys.path.append(str(project_root))

# Import the standardized logging setup
from src.utils.logging_config import get_logger, configure_logging

# Set up basic logging configuration
configure_logging(
    logger_name="frontend",
    log_level=os.environ.get("LOG_LEVEL", "INFO"),
    log_to_console=True,
    log_to_file=False
)

# Get module logger
logger = get_logger("frontend.server")

# Configuration
PORT = int(os.environ.get("FRONTEND_PORT", 8080))
DIRECTORY = Path(__file__).parent.absolute()
AUTO_OPEN_BROWSER = os.environ.get("AUTO_OPEN_BROWSER", "true").lower() in ["true", "1", "yes"]

# Custom request handler
class FrontendRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for the frontend server with support for API routing."""
    
    def __init__(self, *args, **kwargs):
        # Set the directory to serve files from
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        # Parse the URL
        parsed_url = urlparse(self.path)
        
        # If the path is '/', serve index.html
        if parsed_url.path == '/':
            self.path = '/index.html'
        
        # Let the parent class handle the file serving
        return super().do_GET()
    
    def log_message(self, format, *args):
        """Override log_message to use our logger."""
        logger.info("%s - %s", self.address_string(), format % args)

def run_server(port=PORT, auto_open=AUTO_OPEN_BROWSER):
    """
    Run the frontend server.
    
    Args:
        port: Port to listen on
        auto_open: Whether to automatically open the browser
    """
    server_address = ("", port)
    url = f"http://localhost:{port}/"
    
    try:
        # Create the server with address reuse to avoid "Address already in use" errors
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(server_address, FrontendRequestHandler) as httpd:
            logger.info(f"Frontend server running at {url}")
            logger.info(f"Using environment: {os.environ.get('ENVIRONMENT', 'development')}")
            logger.info("Press Ctrl+C to stop")
            
            # Open browser if requested
            if auto_open:
                logger.info("Opening browser...")
                webbrowser.open(url)
            
            # Serve until interrupted
            httpd.serve_forever()
    
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Frontend server for the Recommendation Engine")
    parser.add_argument(
        "-p", "--port", 
        type=int, 
        default=PORT,
        help=f"Port to listen on (default: {PORT})"
    )
    parser.add_argument(
        "--no-browser", 
        action="store_true",
        help="Don't open the browser automatically"
    )
    
    args = parser.parse_args()
    
    # Run the server
    run_server(port=args.port, auto_open=not args.no_browser)
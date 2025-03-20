"""
Simple HTTP server for the Amazon Recommendation Engine frontend.

This script serves the frontend static files on a specified port.
"""

import http.server
import logging
import os
import socketserver
import sys
from urllib.parse import urlparse, parse_qs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Default port
PORT = 8080

# Custom request handler
class FrontendRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for the frontend server."""
    
    def __init__(self, *args, **kwargs):
        # Set the directory to serve files from
        self.directory = os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, directory=self.directory, **kwargs)
    
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

def run_server(port=PORT):
    """
    Run the frontend server.
    
    Args:
        port: Port to listen on
    """
    try:
        # Create the HTTP server
        with socketserver.TCPServer(("", port), FrontendRequestHandler) as httpd:
            logger.info(f"Frontend server running at http://localhost:{port}")
            logger.info("Press Ctrl+C to stop")
            
            # Serve until interrupted
            httpd.serve_forever()
    
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Get port from command line if provided
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid port: {sys.argv[1]}")
            sys.exit(1)
    else:
        port = PORT
    
    # Run the server
    run_server(port) 
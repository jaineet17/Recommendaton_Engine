"""
Simple HTTP server to serve the frontend files.

Run this script to start a web server that serves the frontend files.
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

# Configuration
PORT = 8000
DIRECTORY = Path(__file__).parent.absolute()

class Handler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for serving files from the frontend directory."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def log_message(self, format, *args):
        """Override to customize logging format."""
        print(f"[Frontend Server] {self.address_string()} - {format % args}")

def main():
    """Start the HTTP server and open the browser."""
    print(f"Starting frontend server at http://localhost:{PORT}")
    print(f"Serving files from: {DIRECTORY}")
    
    # Create a TCP server
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("Server started. Press Ctrl+C to stop.")
        
        # Open browser
        webbrowser.open(f"http://localhost:{PORT}")
        
        # Serve until interrupted
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped.")

if __name__ == "__main__":
    main() 
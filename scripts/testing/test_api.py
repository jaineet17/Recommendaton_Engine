#!/usr/bin/env python3
"""
Simple script to test the API connectivity
"""

import requests
import time
import sys

def test_endpoint(url, timeout=5):
    print(f"Testing endpoint: {url}")
    try:
        start = time.time()
        response = requests.get(url, timeout=timeout)
        end = time.time()
        
        print(f"Status code: {response.status_code}")
        print(f"Response time: {end - start:.2f} seconds")
        
        if response.status_code < 400:
            try:
                print(f"Response content: {response.json()}")
                return True
            except:
                print(f"Response not JSON: {response.text[:100]}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    # List of endpoints to test
    base_urls = [
        "http://localhost:5050",
        "http://127.0.0.1:5050",
        "http://0.0.0.0:5050"
    ]
    
    paths = [
        "/health",
        "/api/health",
        "/",
        "/api"
    ]
    
    success = False
    
    print("Starting API connection test...")
    
    for base in base_urls:
        for path in paths:
            url = f"{base}{path}"
            if test_endpoint(url):
                success = True
                print(f"Successfully connected to {url}")
    
    if not success:
        print("Failed to connect to any API endpoint")
        sys.exit(1)
    else:
        print("API connection successful!")
        sys.exit(0)

if __name__ == "__main__":
    main() 
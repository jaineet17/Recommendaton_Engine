#!/usr/bin/env python
"""
Setup script for Amazon Recommendation Engine.
This script helps new users to set up and run the recommendation engine with minimal effort.
"""

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path


def run_command(command, error_message="Command failed"):
    """Run a shell command and handle errors."""
    try:
        subprocess.run(command, check=True, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {error_message}")
        print(f"Command '{command}' failed with exit code {e.returncode}")
        return False


def setup_environment():
    """Set up the Python virtual environment and install dependencies."""
    print("\n🔧 Setting up environment...")
    
    # Check if virtual environment exists
    venv_dir = "venv"
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        if not run_command(f"python -m venv {venv_dir}", "Failed to create virtual environment"):
            return False
    
    # Determine the activate script based on OS
    if sys.platform == "win32":
        activate_script = f"{venv_dir}\\Scripts\\activate"
    else:
        activate_script = f"source {venv_dir}/bin/activate"
    
    # Install dependencies
    print("Installing dependencies...")
    if not run_command(f"{activate_script} && pip install -r requirements.txt", 
                     "Failed to install dependencies"):
        return False
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        print("Creating .env file from example...")
        if not run_command("cp .env.example .env", "Failed to create .env file"):
            return False
    
    print("✅ Environment setup completed successfully!")
    return True


def setup_docker():
    """Set up Docker environment."""
    print("\n🐳 Setting up Docker environment...")
    
    # Check if Docker is installed
    if not run_command("docker --version", "Docker not found. Please install Docker first."):
        return False
    
    # Check if Docker Compose is installed
    if not run_command("docker-compose --version", 
                      "Docker Compose not found. Please install Docker Compose first."):
        return False
    
    print("✅ Docker environment check completed successfully!")
    return True


def start_api(debug=False):
    """Start the API server."""
    print("\n🚀 Starting API server...")
    
    debug_flag = "--debug" if debug else ""
    cmd = f"python run_api.py {debug_flag}"
    
    print(f"Running command: {cmd}")
    subprocess.Popen(cmd, shell=True)
    
    print("API server started! You can access it at: http://localhost:5050")
    return True


def start_docker_services(mode="dev"):
    """Start the services using Docker Compose."""
    print(f"\n🐳 Starting services with Docker Compose ({mode} mode)...")
    
    env_prefix = f"ENVIRONMENT={mode}" if mode == "production" else ""
    cmd = f"{env_prefix} docker-compose up -d"
    
    if not run_command(cmd, "Failed to start Docker services"):
        return False
    
    print("✅ Docker services started successfully!")
    return True


def open_docs():
    """Open the documentation in a web browser."""
    docs_path = Path("docs/DOCUMENTATION.md")
    if docs_path.exists():
        print("\n📖 Opening documentation...")
        webbrowser.open(f"file://{os.path.abspath(docs_path)}")
    else:
        print("Documentation file not found.")
        return False
    return True


def main():
    """Main function to parse arguments and run commands."""
    parser = argparse.ArgumentParser(description="Amazon Recommendation Engine Setup Script")
    
    parser.add_argument("--setup", action="store_true", help="Set up the environment")
    parser.add_argument("--docker", action="store_true", help="Set up and check Docker environment")
    parser.add_argument("--start-api", action="store_true", help="Start the API server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for API")
    parser.add_argument("--docker-start", choices=["dev", "production"], 
                        help="Start services with Docker Compose")
    parser.add_argument("--docs", action="store_true", help="Open documentation")
    
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nQuick Start Guide:")
        print("1. Run './setup.py --setup' to set up the environment")
        print("2. Run './setup.py --start-api --debug' to start the API server in debug mode")
        print("3. Run './setup.py --docs' to open the documentation")
        return
    
    # Run commands based on arguments
    if args.setup:
        setup_environment()
    
    if args.docker:
        setup_docker()
    
    if args.start_api:
        start_api(args.debug)
    
    if args.docker_start:
        start_docker_services(args.docker_start)
    
    if args.docs:
        open_docs()


if __name__ == "__main__":
    main() 
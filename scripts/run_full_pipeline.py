#!/usr/bin/env python3
"""
Full Pipeline for Amazon Recommendation System.

This script runs the complete pipeline:
1. Download and process Amazon review data
2. Train multiple recommendation models
3. Create an ensemble model
4. Test the models and ensemble

This is a comprehensive script for end-to-end execution.
"""

import os
import sys
import time
import logging
import argparse
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('pipeline')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the complete recommendation pipeline')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip data download and use existing processed data')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training and use existing models')
    parser.add_argument('--sample-size', type=int, default=100000,
                       help='Number of reviews to sample for training')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--embedding-dim', type=int, default=32,
                       help='Embedding dimension size')
    return parser.parse_args()

def run_command(command, description):
    """
    Run a shell command and log the output.
    
    Args:
        command: Command to run
        description: Description of the command
        
    Returns:
        True if command was successful, False otherwise
    """
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Log output in real-time
        for line in process.stdout:
            logger.info(line.strip())
        
        process.wait()
        
        if process.returncode == 0:
            logger.info(f"Successfully completed: {description}")
            return True
        else:
            logger.error(f"Failed with exit code {process.returncode}: {description}")
            return False
    
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False

def run_data_download_and_processing(args):
    """
    Run the data download and processing step.
    
    Args:
        args: Command line arguments
        
    Returns:
        True if successful, False otherwise
    """
    if args.skip_download:
        logger.info("Skipping data download and processing step")
        return True
    
    command = f"python scripts/download_and_train.py --sample-size {args.sample_size} --skip-training"
    return run_command(command, "Data download and processing")

def train_models(args):
    """
    Train all recommendation models.
    
    Args:
        args: Command line arguments
        
    Returns:
        True if successful, False otherwise
    """
    if args.skip_training:
        logger.info("Skipping model training step")
        return True
    
    # Train all models using the combined script
    command = (
        f"python scripts/download_and_train.py --skip-download "
        f"--epochs {args.epochs} --embedding-dim {args.embedding_dim}"
    )
    if not run_command(command, "Training all models"):
        logger.error("Failed to train models, but continuing with the pipeline")
    
    # Train content-based model separately with a reasonable number of components
    # Use a smaller number than embedding_dim to avoid dimensionality issues
    n_components = min(30, args.embedding_dim - 2)
    command = (
        f"python scripts/train_content_based.py "
        f"--n-components {n_components}"
    )
    return run_command(command, "Training content-based model")

def create_ensemble(args):
    """
    Create ensemble model from trained models.
    
    Args:
        args: Command line arguments
        
    Returns:
        True if successful, False otherwise
    """
    command = "python scripts/ensemble_recommender.py --num-users 3"
    return run_command(command, "Creating ensemble model")

def test_models(args):
    """
    Test all models and compare their performance.
    
    Args:
        args: Command line arguments
        
    Returns:
        True if successful, False otherwise
    """
    command = "python scripts/test_models.py --num-users 5"
    return run_command(command, "Testing all models")

def main():
    """Main function to run the complete pipeline."""
    args = parse_args()
    
    start_time = time.time()
    logger.info("Starting the complete recommendation pipeline")
    
    # Step 1: Download and process data
    if not run_data_download_and_processing(args):
        logger.error("Failed at data download and processing step")
        return
    
    # Step 2: Train models
    if not train_models(args):
        logger.error("Failed at model training step")
        return
    
    # Step 3: Create ensemble
    if not create_ensemble(args):
        logger.error("Failed at ensemble creation step")
        return
    
    # Step 4: Test models
    if not test_models(args):
        logger.error("Failed at model testing step")
        return
    
    # Done!
    total_time = time.time() - start_time
    logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
    
    # Summary
    logger.info("\nRecommendation system is now ready with the following models:")
    logger.info("- LightGCN: Graph-based collaborative filtering model")
    logger.info("- NCF: Neural collaborative filtering model")
    logger.info("- Simple MF: Simple matrix factorization model")
    logger.info("- Content-based: Content-based filtering model")
    logger.info("- Ensemble: Weighted combination of all models")
    logger.info("\nUse the API to serve recommendations or the test script to evaluate models")

if __name__ == '__main__':
    main() 
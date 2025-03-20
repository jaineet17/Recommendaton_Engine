#!/usr/bin/env python3
"""
Script to train only the LightGCN model with detailed debugging.
"""

import os
import sys
import json
import logging
import argparse
import pickle
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging with more details
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to get more information
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_lightgcn')

# Import model-related modules
try:
    from src.models.lightgcn import LightGCN
except ImportError as e:
    logger.error(f"Failed to import LightGCN model: {e}")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train LightGCN model')
    parser.add_argument('--data-file', type=str, default='data/processed/amazon_reviews_processed.parquet',
                        help='Processed data file for training')
    parser.add_argument('--output-dir', type=str, default='data/models',
                        help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--embedding-dim', type=int, default=32,
                        help='Embedding dimension size')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of GCN layers')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                        help='Learning rate for optimizer')
    parser.add_argument('--negative-samples', type=int, default=1,
                        help='Number of negative samples per positive sample')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional logging')
    return parser.parse_args()

def main():
    """Main function to train LightGCN model."""
    args = parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        for handler in logging.root.handlers:
            handler.setLevel(logging.DEBUG)
    
    # Check if the data file exists
    if not os.path.exists(args.data_file):
        logger.error(f"Data file {args.data_file} does not exist")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load processed data
    logger.info(f"Loading data from {args.data_file}")
    try:
        df = pd.read_parquet(args.data_file)
        logger.info(f"Loaded {len(df)} records with columns: {df.columns.tolist()}")
        
        # Check for required columns
        required_columns = ['user_id', 'product_id', 'rating']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column {col} not in data")
                return
        
        # Print data statistics
        logger.info(f"Data contains {len(df['user_id'].unique())} unique users")
        logger.info(f"Data contains {len(df['product_id'].unique())} unique products")
        logger.info(f"Rating range: {df['rating'].min()} to {df['rating'].max()}")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Create and train LightGCN model
    logger.info("Initializing LightGCN model")
    try:
        # Import torch here to catch any torch-specific errors
        import torch
        logger.info(f"Using PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Initialize model with provided parameters
        model = LightGCN(
            name="lightgcn",
            version="1.0.0",
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            negative_samples=args.negative_samples,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Print model hyperparameters
        logger.info(f"Model hyperparameters: {model.hyperparameters}")
        
        # Train the model
        logger.info("Starting model training")
        metrics = model.train(df)
        
        # Print training metrics
        logger.info(f"Training metrics: {metrics}")
        
        # Save model
        model_path = os.path.join(args.output_dir, 'lightgcn_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'name': 'lightgcn',
                'version': '1.0.0',
                'training_date': datetime.now(),
                'hyperparameters': model.hyperparameters,
                'metrics': metrics,
                'user_map': model.user_map,
                'product_map': model.product_map,
                'user_factors': model.user_factors,
                'item_factors': model.item_factors,
            }, f)
        
        logger.info(f"Model saved to {model_path}")
        
        # Test the model with a sample prediction
        logger.info("Testing model with sample prediction")
        user_id = df['user_id'].iloc[0]
        recs = model.predict(user_id, n=5)
        logger.info(f"Top 5 recommendations for user {user_id}: {recs}")
        
    except Exception as e:
        logger.error(f"Error training LightGCN model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
if __name__ == '__main__':
    main() 
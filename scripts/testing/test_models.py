#!/usr/bin/env python3
"""
Script to test trained recommendation models.

This script will:
1. Load the trained recommendation models
2. Make sample predictions for users
3. Print evaluation metrics if test data is available
"""

import os
import sys
import pickle
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_models')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test trained recommendation models')
    parser.add_argument('--model-dir', type=str, default='data/models',
                        help='Directory containing trained models')
    parser.add_argument('--data-file', type=str, default='data/processed/amazon_reviews_processed.parquet',
                        help='Processed data file (for test users)')
    parser.add_argument('--num-recs', type=int, default=10,
                        help='Number of recommendations to generate per user')
    parser.add_argument('--num-users', type=int, default=5,
                        help='Number of users to generate recommendations for')
    return parser.parse_args()

def load_model(model_path):
    """Load a trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        logger.info(f"Loaded model from {model_path}")
        return model_data
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None

def make_predictions(model_data, user_ids, num_recs=10):
    """Make predictions for a list of users."""
    predictions = {}
    
    # Extract model components
    user_map = model_data.get('user_map', {})
    product_map = model_data.get('product_map', {})
    user_factors = model_data.get('user_factors')
    item_factors = model_data.get('item_factors')
    model_name = model_data.get('name', '')
    
    if user_factors is None or item_factors is None:
        logger.error("Model does not contain user or item factors")
        return predictions
    
    # Reverse product map for predictions
    reverse_product_map = {idx: prod_id for prod_id, idx in product_map.items()}
    
    # Get item factors in correct orientation for matrix multiplication
    # Different models store factors differently
    if model_name == 'simple_mf':
        # For Simple MF, item factors are (n_components, n_items)
        item_factors_for_pred = item_factors
    elif model_name == 'ncf':
        # For NCF, special handling based on model type
        model_type = model_data.get('hyperparameters', {}).get('model_type', 'MLP')
        logger.info(f"NCF model type: {model_type}")
        
        # Ensure item factors are properly transposed
        if item_factors.shape[0] == len(product_map):
            # Item factors are (n_items, n_components)
            item_factors_for_pred = item_factors.T
        else:
            # Item factors already transposed 
            item_factors_for_pred = item_factors
    elif model_name == 'content_based':
        # For content-based, item factors are typically (n_components, n_items)
        item_factors_for_pred = item_factors
    else:
        # For other models (LightGCN, etc.)
        item_factors_for_pred = item_factors.T
    
    # Generate predictions for each user
    for user_id in user_ids:
        if user_id not in user_map:
            logger.warning(f"User {user_id} not found in model")
            continue
        
        # Get user index
        user_idx = user_map[user_id]
        
        # Get user vector
        user_vector = user_factors[user_idx]
        
        # Calculate scores for all items
        try:
            # Try matrix multiplication
            scores = np.dot(user_vector, item_factors_for_pred)
        except ValueError as e:
            logger.info(f"Matrix multiplication failed: {e}. Trying alternative approach.")
            # Try alternative approaches
            if model_name in ['simple_mf', 'content_based']:
                scores = np.dot(user_vector, item_factors)
            else:
                # Try dot product with item factors directly
                try:
                    scores = np.array([np.dot(user_vector, item_factors[i]) 
                                      for i in range(len(product_map))])
                except:
                    # Last resort: transpose user vector
                    scores = np.dot(item_factors, user_vector)
        
        # Debug info for scores
        logger.info(f"Model: {model_name} - Scores stats: min={np.min(scores):.6f}, max={np.max(scores):.6f}, mean={np.mean(scores):.6f}, std={np.std(scores):.6f}")
        logger.info(f"User vector norm: {np.linalg.norm(user_vector):.6f}, Item factors norm: {np.linalg.norm(item_factors_for_pred):.6f}")
        
        # Apply post-processing for different models to improve scores
        if model_name == 'ncf':
            # NCF often produces very small values, scale them up significantly
            if np.max(np.abs(scores)) < 0.1:
                logger.info("Scaling up NCF scores which are very small")
                # Use a much larger scale factor for NCF
                scale_factor = 10000.0
                scores = scores * scale_factor
            
            # NCF may have negative values, use sigmoid to normalize
            if np.min(scores) < 0:
                logger.info("Applying sigmoid to NCF scores to handle negative values")
                scores = 1.0 / (1.0 + np.exp(-scores))
        
        # Get top N items
        top_indices = np.argsort(scores)[::-1][:num_recs]
        
        # Map to product IDs and scores
        user_predictions = [
            (reverse_product_map[idx], float(scores[idx]))
            for idx in top_indices
        ]
        
        predictions[user_id] = user_predictions
    
    return predictions

def main():
    """Main function to test recommendation models."""
    args = parse_args()
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory {args.model_dir} does not exist")
        return
    
    # Find all model files
    model_files = [f for f in os.listdir(args.model_dir) if f.endswith('.pkl')]
    
    if not model_files:
        logger.error(f"No model files found in {args.model_dir}")
        return
    
    logger.info(f"Found {len(model_files)} model files: {', '.join(model_files)}")
    
    # Load data to get test users
    try:
        df = pd.read_parquet(args.data_file)
        unique_users = df['user_id'].unique()
        test_users = unique_users[:args.num_users]
        logger.info(f"Selected {len(test_users)} test users from data")
    except Exception as e:
        logger.error(f"Failed to load data file: {e}")
        logger.info("Using dummy test users")
        test_users = [f"user_{i}" for i in range(args.num_users)]
    
    # Process each model
    for model_file in model_files:
        model_path = os.path.join(args.model_dir, model_file)
        model_data = load_model(model_path)
        
        if model_data:
            logger.info(f"\nTesting model: {model_data.get('name', model_file)}")
            
            # Display model info
            logger.info(f"Model version: {model_data.get('version', 'unknown')}")
            logger.info(f"Training date: {model_data.get('training_date', 'unknown')}")
            
            # Display model metrics if available
            if 'metrics' in model_data:
                logger.info("Model metrics:")
                for metric, value in model_data['metrics'].items():
                    logger.info(f"  {metric}: {value}")
            
            # Make predictions
            predictions = make_predictions(model_data, test_users, args.num_recs)
            
            # Display predictions
            logger.info("\nSample predictions:")
            for user_id, user_preds in predictions.items():
                logger.info(f"\nTop {len(user_preds)} recommendations for user {user_id}:")
                for prod_id, score in user_preds:
                    logger.info(f"  {prod_id}: {score:.4f}")
            
            logger.info("\n" + "="*80)

if __name__ == '__main__':
    main() 
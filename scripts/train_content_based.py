#!/usr/bin/env python3
"""
Script to train a content-based filtering model for the recommendation system.

This script will:
1. Load the processed data
2. Train a content-based model using TF-IDF and SVD
3. Save the model to the specified directory
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_content_based')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a content-based filtering model')
    parser.add_argument('--data-file', type=str, default='data/processed/amazon_reviews_processed.parquet',
                       help='Processed data file for training')
    parser.add_argument('--output-dir', type=str, default='data/models',
                       help='Directory to save trained model')
    parser.add_argument('--n-components', type=int, default=30,
                       help='Number of components for SVD')
    parser.add_argument('--model-name', type=str, default='content_based',
                       help='Name for the saved model')
    return parser.parse_args()

def train_content_based_model(df, output_dir, n_components=30, model_name='content_based'):
    """Train a content-based filtering model using TF-IDF and SVD on product metadata."""
    logger.info(f"Training content-based model with {n_components} components...")
    
    # Create user and product maps
    user_map = {user_id: i for i, user_id in enumerate(df['user_id'].unique())}
    product_map = {prod_id: i for i, prod_id in enumerate(df['product_id'].unique())}
    
    logger.info(f"Created mappings for {len(user_map)} users and {len(product_map)} products")
    
    # For this example, we'll use product IDs as a proxy for product features
    # In a real system, use product metadata like title, description, category
    product_ids = list(product_map.keys())
    
    # Create a simple representation
    logger.info("Creating product feature vectors using TF-IDF on product IDs")
    vectorizer = TfidfVectorizer(analyzer=lambda x: [c for c in x])
    product_features = vectorizer.fit_transform([str(p) for p in product_ids])
    
    # Get the number of features created by TF-IDF
    n_features = product_features.shape[1]
    logger.info(f"TF-IDF created {n_features} features")
    
    # Adjust n_components if needed
    if n_components > n_features:
        logger.warning(f"Reducing n_components from {n_components} to {n_features-1} (max possible)")
        n_components = n_features - 1
    
    # Reduce dimensionality
    logger.info(f"Reducing dimensionality to {n_components} components using SVD")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    item_factors = svd.fit_transform(product_features)
    
    # Create user factors by aggregating items they've interacted with
    logger.info("Creating user factors based on their interactions")
    user_factors = np.zeros((len(user_map), n_components))
    
    # For each user, average the factors of items they've rated highly
    for user_id, user_idx in user_map.items():
        # Get items rated by this user
        user_items = df[df['user_id'] == user_id]
        # Focus on highly rated items (rating >= 4)
        highly_rated = user_items[user_items['rating'] >= 4]
        
        if len(highly_rated) > 0:
            # Get the item factors for these items
            item_indices = [product_map[prod_id] for prod_id in highly_rated['product_id']]
            # Average the factors
            user_factors[user_idx] = np.mean(item_factors[item_indices], axis=0)
        else:
            # If no highly rated items, use all items
            item_indices = [product_map[prod_id] for prod_id in user_items['product_id']]
            if item_indices:
                user_factors[user_idx] = np.mean(item_factors[item_indices], axis=0)
    
    # Calculate similarity score (dot product) for a sample user
    sample_user_idx = 0
    sample_scores = np.dot(user_factors[sample_user_idx], item_factors.T)
    logger.info(f"Sample scores for user {list(user_map.keys())[sample_user_idx]}: " 
               f"min={np.min(sample_scores):.4f}, max={np.max(sample_scores):.4f}, "
               f"mean={np.mean(sample_scores):.4f}")
    
    # Save the model
    model_path = os.path.join(output_dir, f"{model_name}_model.pkl")
    
    # Calculate average score as a baseline metric
    # In a real system, compute proper evaluation metrics
    all_scores = np.dot(user_factors, item_factors.T)
    avg_score = np.mean(all_scores)
    
    # Ensure factors have good magnitude (not too small)
    logger.info(f"User factors norm: {np.linalg.norm(user_factors):.4f}")
    logger.info(f"Item factors norm: {np.linalg.norm(item_factors):.4f}")
    
    # Save model data
    model_data = {
        'name': model_name,
        'version': '1.0.0',
        'training_date': datetime.now(),
        'hyperparameters': {
            'n_components': n_components,
        },
        'metrics': {
            'avg_score': float(avg_score),
        },
        'user_map': user_map,
        'product_map': product_map,
        'user_factors': user_factors,
        'item_factors': item_factors.T,  # Transpose for prediction compatibility
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Saved content-based model to {model_path}")
    
    return model_data

def main():
    """Main function to train content-based model."""
    args = parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        logger.error(f"Data file {args.data_file} does not exist")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.data_file}")
    try:
        df = pd.read_parquet(args.data_file)
        logger.info(f"Loaded {len(df)} records")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Train model
    start_time = datetime.now()
    model_data = train_content_based_model(
        df, 
        args.output_dir,
        n_components=args.n_components,
        model_name=args.model_name
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Display some model information
    logger.info("\nModel Information:")
    logger.info(f"Model name: {model_data['name']}")
    logger.info(f"Version: {model_data['version']}")
    logger.info(f"Training date: {model_data['training_date']}")
    logger.info(f"Number of users: {len(model_data['user_map'])}")
    logger.info(f"Number of products: {len(model_data['product_map'])}")
    logger.info(f"Components: {model_data['hyperparameters']['n_components']}")
    logger.info(f"Average score: {model_data['metrics']['avg_score']:.4f}")

if __name__ == "__main__":
    main() 
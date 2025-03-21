#!/usr/bin/env python3
"""
Ensemble Recommender for the Amazon recommendation system.

This script creates and tests an ensemble model that combines recommendations from multiple trained models:
- LightGCN: Graph-based collaborative filtering
- NCF: Neural collaborative filtering  
- Simple MF: Simple matrix factorization
- Content-based: Content-based filtering

The ensemble uses a weighted combination of these models to provide high-quality recommendations.
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
logger = logging.getLogger('ensemble_recommender')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create and test ensemble recommender')
    parser.add_argument('--models-dir', type=str, default='data/models',
                        help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default='data/models',
                        help='Directory to save ensemble model')
    parser.add_argument('--data-file', type=str, default='data/processed/amazon_reviews_processed.parquet',
                        help='Processed data file for evaluation')
    parser.add_argument('--num-users', type=int, default=5,
                        help='Number of users to generate recommendations for in testing')
    parser.add_argument('--num-recommendations', type=int, default=10,
                        help='Number of recommendations to generate')
    parser.add_argument('--weights', type=str, default='0.5,0.2,0.2,0.1',
                        help='Comma-separated weights for models in order: lightgcn,content_based,simple_mf,ncf')
    return parser.parse_args()

def load_models(models_dir):
    """
    Load all available trained models from directory.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Dictionary of model_name -> model_data
    """
    models = {}
    
    # Define model files to look for
    model_files = {
        'lightgcn': 'lightgcn_model.pkl',
        'ncf': 'ncf_model.pkl',
        'simple_mf': 'simple_mf_model.pkl',
        'content_based': 'content_based_model.pkl'
    }
    
    # Load each model if available
    for model_name, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                logger.info(f"Loaded {model_name} model with {len(models[model_name]['user_map'])} users and {len(models[model_name]['product_map'])} items")
            except Exception as e:
                logger.error(f"Failed to load {model_name} model: {e}")
        else:
            logger.warning(f"{model_name} model not found at {model_path}")
    
    return models

def create_ensemble_model(models, weights=None):
    """
    Create an ensemble model from the loaded models.
    
    Args:
        models: Dictionary of model_name -> model_data
        weights: Dictionary of model_name -> weight
        
    Returns:
        Ensemble model data
    """
    if not models:
        raise ValueError("No models available for ensemble")
    
    # Default to equal weights if not provided
    if weights is None:
        weights = {name: 1.0 / len(models) for name in models}
    
    # Choose primary model for user and product mappings
    # Prefer models in this order: lightgcn, content_based, simple_mf, ncf
    primary_model_name = None
    for name in ['lightgcn', 'content_based', 'simple_mf', 'ncf']:
        if name in models:
            primary_model_name = name
            break
    
    if primary_model_name is None:
        raise ValueError("No suitable primary model found")
    
    primary_model = models[primary_model_name]
    
    # Create ensemble model data
    ensemble = {
        'name': 'ensemble',
        'version': '1.0.0',
        'training_date': datetime.now(),
        'user_map': primary_model['user_map'],
        'product_map': primary_model['product_map'],
        'components': list(models.keys()),
        'weights': weights,
        'primary_model': primary_model_name
    }
    
    # Add metadata and metrics
    ensemble['metadata'] = {
        'models': list(models.keys()),
        'num_users': len(primary_model['user_map']),
        'num_items': len(primary_model['product_map']),
        'created_date': datetime.now()
    }
    
    # Add user and item factors from the primary model to make it compatible with test_models.py
    ensemble['user_factors'] = primary_model.get('user_factors', None)
    ensemble['item_factors'] = primary_model.get('item_factors', None)
    
    # Calculate normalization factors for each model's scores
    # This helps normalize scores from different models to similar ranges
    ensemble['normalization'] = {}
    
    return ensemble

def get_model_recommendations(model_data, user_id, top_n=10):
    """
    Get recommendations from a single model.
    
    Args:
        model_data: Model data dictionary
        user_id: User ID to get recommendations for
        top_n: Number of recommendations to return
        
    Returns:
        List of (product_id, score) tuples
    """
    # Check if user exists in model
    user_map = model_data.get('user_map', {})
    if user_id not in user_map:
        logger.warning(f"User {user_id} not found in model {model_data.get('name', 'unknown')}")
        return []
    
    try:
        # Get user index and vector
        user_idx = user_map[user_id]
        user_factors = model_data.get('user_factors')
        item_factors = model_data.get('item_factors')
        product_map = model_data.get('product_map', {})
        
        user_vector = user_factors[user_idx]
        
        # Different models have different item factor layouts
        model_name = model_data.get('name', '')
        if model_name in ['content_based', 'simple_mf']:
            # item_factors is (n_components, n_items)
            scores = np.dot(user_vector, item_factors)
        else:
            # item_factors is (n_items, n_components) 
            scores = np.dot(user_vector, item_factors.T)
        
        # Get top N items
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        # Map to product IDs
        reverse_product_map = {idx: prod_id for prod_id, idx in product_map.items()}
        recommendations = [(reverse_product_map[idx], float(scores[idx])) for idx in top_indices]
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Error getting recommendations from model {model_data.get('name', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_ensemble_recommendations(ensemble, models, user_id, top_n=10):
    """
    Get recommendations from the ensemble model.
    
    Args:
        ensemble: Ensemble model data
        models: Dictionary of model_name -> model_data
        user_id: User ID to get recommendations for
        top_n: Number of recommendations to return
        
    Returns:
        List of product IDs
    """
    # Get weights
    weights = ensemble.get('weights', {})
    
    # Initialize score map
    scores = {}
    
    # Get recommendations from each component model
    for model_name, model_data in models.items():
        # Skip if weight is zero
        if model_name not in weights or weights[model_name] == 0:
            continue
        
        # Get recommendations from this model
        model_recs = get_model_recommendations(model_data, user_id, top_n * 2)  # Get more for diversity
        
        if not model_recs:
            logger.warning(f"No recommendations from model {model_name} for user {user_id}")
            continue
        
        # Add weighted scores to the map
        weight = weights[model_name]
        
        # Normalize scores from this model to [0, 1] range
        min_score = min(rec[1] for rec in model_recs)
        max_score = max(rec[1] for rec in model_recs)
        score_range = max_score - min_score
        
        for product_id, score in model_recs:
            # Normalize score to [0, 1]
            normalized_score = (score - min_score) / score_range if score_range > 0 else 0.5
            
            # Add to scores map
            if product_id not in scores:
                scores[product_id] = 0
            
            # Apply weight and add to total score
            scores[product_id] += normalized_score * weight
    
    # Sort products by final score
    sorted_products = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N product IDs with scores
    return [(prod_id, score) for prod_id, score in sorted_products[:top_n]]

def save_ensemble_model(ensemble, output_path):
    """
    Save the ensemble model to disk.
    
    Args:
        ensemble: Ensemble model data
        output_path: Path to save the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(ensemble, f)
        logger.info(f"Saved ensemble model to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save ensemble model: {e}")
        return False

def main():
    """Main function to create and test ensemble recommender."""
    args = parse_args()
    
    # Parse weights
    try:
        weight_values = [float(w) for w in args.weights.split(',')]
        model_names = ['lightgcn', 'content_based', 'simple_mf', 'ncf']
        
        if len(weight_values) != len(model_names):
            logger.warning(f"Expected {len(model_names)} weights, got {len(weight_values)}. Using equal weights.")
            weights = {name: 1.0 / len(model_names) for name in model_names}
        else:
            weights = {name: weight for name, weight in zip(model_names, weight_values)}
        
        logger.info(f"Using weights: {weights}")
    except ValueError as e:
        logger.error(f"Invalid weights format: {e}")
        logger.info("Using equal weights for all models")
        weights = None
    
    # Load all available models
    models = load_models(args.models_dir)
    
    if not models:
        logger.error("No models loaded. Please train models first.")
        return
    
    # Create ensemble model
    ensemble = create_ensemble_model(models, weights)
    
    # Save ensemble model
    ensemble_path = os.path.join(args.output_dir, 'ensemble_model.pkl')
    save_ensemble_model(ensemble, ensemble_path)
    
    # Test on some users
    try:
        # Load data for testing
        df = pd.read_parquet(args.data_file)
        # Select a few users
        test_users = df['user_id'].unique()[:args.num_users]
        
        logger.info(f"Testing ensemble on {len(test_users)} users")
        
        # Get recommendations for each test user
        for user_id in test_users:
            # Get recommendations from ensemble
            ensemble_recs = get_ensemble_recommendations(ensemble, models, user_id, args.num_recommendations)
            
            if not ensemble_recs:
                logger.warning(f"No ensemble recommendations for user {user_id}")
                continue
            
            logger.info(f"\nTop {len(ensemble_recs)} ensemble recommendations for user {user_id}:")
            for i, (prod_id, score) in enumerate(ensemble_recs, 1):
                logger.info(f"  {i}. {prod_id}: {score:.4f}")
            
            # Compare with individual model recommendations
            logger.info("\nIndividual model recommendations:")
            for model_name, model_data in models.items():
                model_recs = get_model_recommendations(model_data, user_id, 5)  # Just top 5 for comparison
                
                if not model_recs:
                    logger.warning(f"  No recommendations from {model_name} for user {user_id}")
                    continue
                
                logger.info(f"  {model_name.upper()}:")
                for i, (prod_id, score) in enumerate(model_recs, 1):
                    logger.info(f"    {i}. {prod_id}: {score:.4f}")
    
    except Exception as e:
        logger.error(f"Error testing ensemble: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 
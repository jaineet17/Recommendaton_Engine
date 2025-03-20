#!/usr/bin/env python3
"""
Script to retrain the NCF model with fixes.

This script will:
1. Load the processed data
2. Train an NCF model with the MLP model type
3. Save the model with proper factors for prediction
"""

import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import NCF model
try:
    from src.models.neural_collaborative_filtering import NeuralCollaborativeFiltering
except ImportError as e:
    print(f"Failed to import NeuralCollaborativeFiltering: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('retrain_ncf')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Retrain NCF model with fixes')
    parser.add_argument('--data-file', type=str, default='data/processed/amazon_reviews_processed.parquet',
                       help='Processed data file for training')
    parser.add_argument('--output-dir', type=str, default='data/models',
                       help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--embedding-dim', type=int, default=32,
                       help='Embedding dimension size')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--model-type', type=str, default='MLP',
                       help='NCF model type: MLP, GMF, or NeuMF')
    return parser.parse_args()

def main():
    """Main function to retrain NCF model."""
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
    
    # Initialize and train NCF model
    logger.info(f"Initializing NCF model with {args.model_type} type")
    try:
        # MLP layer sizes
        mlp_layers = [64, 32, 16]
        
        # Initialize model
        model = NeuralCollaborativeFiltering(
            name="ncf",
            version="1.0.0",
            embedding_dim=args.embedding_dim,
            mlp_layers=mlp_layers,
            model_type=args.model_type,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs
        )
        
        # Train model
        logger.info(f"Starting training for {args.epochs} epochs")
        start_time = datetime.now()
        
        # Split data for validation (10%)
        train_data, val_data = train_test_split(df, test_size=0.1, random_state=42)
        logger.info(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")
        
        metrics = model.train(train_data, val_data)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Add additional metrics
        metrics['training_time'] = training_time
        metrics['num_epochs'] = args.epochs
        metrics['num_users'] = len(model.user_map)
        metrics['num_items'] = len(model.product_map)
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Metrics: {metrics}")
        
        # Save model
        model_path = os.path.join(args.output_dir, 'ncf_model.pkl')
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Generate sample predictions
        sample_user = df['user_id'].iloc[0]
        logger.info(f"Sample predictions for user {sample_user}:")
        recs = model.predict(sample_user, n=5)
        for i, (item, score) in enumerate(recs, 1):
            logger.info(f"  {i}. {item}: {score:.4f}")
        
    except Exception as e:
        logger.error(f"Error training NCF model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

if __name__ == '__main__':
    main() 
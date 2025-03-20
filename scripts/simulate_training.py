"""
Simulate model training for the recommendation engine.

This script simulates training NCF and LightGCN models and creates dummy model files.
"""

import os
import time
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Create output directory if it doesn't exist
os.makedirs('models', exist_ok=True)

print("Loading training data...")
try:
    features_df = pd.read_parquet('data/processed/features.parquet')
    print(f"Loaded {len(features_df)} training examples")
except Exception as e:
    print(f"Error loading training data: {e}")
    print("Creating dummy training data instead...")
    features_df = pd.DataFrame({
        'user_id': [f"user_{i}" for i in range(100) for _ in range(20)],
        'product_id': [f"product_{j}" for _ in range(100) for j in range(20)],
        'rating': np.random.randint(1, 6, size=2000)
    })

# Create user and item mappings
print("Creating user and item mappings...")
users = features_df['user_id'].unique()
items = features_df['product_id'].unique()

user_mapping = {user: idx for idx, user in enumerate(users)}
item_mapping = {item: idx for idx, item in enumerate(items)}

print(f"Found {len(user_mapping)} unique users and {len(item_mapping)} unique items")

# Simulate NCF training
print("\n----- Neural Collaborative Filtering (NCF) -----")
print("Starting training...")
ncf_start_time = time.time()

# Simulate training delay based on data size
delay = min(len(features_df) / 1000, 10)  # Cap at 10 seconds
time.sleep(delay)

print(f"Training completed in {delay:.2f} seconds")

# Create dummy model data
ncf_model = {
    'name': 'NCF',
    'version': '0.1.0',
    'training_date': datetime.now(),
    'hyperparameters': {
        'embedding_dim': 32,
        'mlp_layers': [64, 32, 16],
        'learning_rate': 0.001,
        'batch_size': 256,
        'epochs': 20
    },
    'metrics': {
        'train_loss': 0.123,
        'valid_loss': 0.234,
        'rmse': 0.456,
        'mae': 0.345
    },
    'user_map': user_mapping,
    'product_map': item_mapping,
    # Generate random user and item factors for recommendation simulation
    'user_factors': np.random.normal(0, 0.1, size=(len(user_mapping), 32)),
    'item_factors': np.random.normal(0, 0.1, size=(len(item_mapping), 32)),
}

# Save NCF model
ncf_path = 'models/ncf_model.pkl'
with open(ncf_path, 'wb') as f:
    pickle.dump(ncf_model, f)

print(f"Model saved to {ncf_path}")

# Simulate LightGCN training
print("\n----- LightGCN -----")
print("Starting training...")
lightgcn_start_time = time.time()

# Simulate training delay based on data size
delay = min(len(features_df) / 800, 12)  # Cap at 12 seconds
time.sleep(delay)

print(f"Training completed in {delay:.2f} seconds")

# Create dummy model data
lightgcn_model = {
    'name': 'LightGCN',
    'version': '0.1.0',
    'training_date': datetime.now(),
    'hyperparameters': {
        'embedding_dim': 64,
        'num_layers': 3,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 100
    },
    'metrics': {
        'train_loss': 0.111,
        'valid_loss': 0.222,
        'rmse': 0.333,
        'mae': 0.222,
        'precision@10': 0.45,
        'recall@10': 0.32
    },
    'user_map': user_mapping,
    'product_map': item_mapping,
    # Generate random user and item factors for recommendation simulation
    'user_factors': np.random.normal(0, 0.1, size=(len(user_mapping), 64)),
    'item_factors': np.random.normal(0, 0.1, size=(len(item_mapping), 64)),
}

# Save LightGCN model
lightgcn_path = 'models/lightgcn_model.pkl'
with open(lightgcn_path, 'wb') as f:
    pickle.dump(lightgcn_model, f)

print(f"Model saved to {lightgcn_path}")

print("\nModel training simulation complete!")
print(f"- NCF model saved to {ncf_path}")
print(f"- LightGCN model saved to {lightgcn_path}")
print("These models can now be used with the recommendation API.") 
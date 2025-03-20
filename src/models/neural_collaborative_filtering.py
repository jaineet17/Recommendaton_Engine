"""
Neural Collaborative Filtering (NCF) model for the Amazon recommendation system.

This module implements the Neural Collaborative Filtering model, which combines
matrix factorization with deep neural networks to capture non-linear user-item
interactions.

Reference: He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017).
Neural collaborative filtering. In Proceedings of the 26th international conference
on world wide web (pp. 173-182).
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base_model import MatrixFactorizationBase

# Set up logging
logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. NCF model will not work.")
    TORCH_AVAILABLE = False


class GMF(nn.Module):
    """
    Generalized Matrix Factorization module for NCF.
    
    This module implements the Generalized Matrix Factorization (GMF) component
    of the Neural Collaborative Filtering model.
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        """
        Initialize GMF module.
        
        Args:
            num_users: Number of users
            num_items: Number of items
            embedding_dim: Dimension of embeddings
        """
        super(GMF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with normal distribution."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.kaiming_uniform_(self.output_layer.weight, a=1)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GMF.
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            
        Returns:
            Model output
        """
        user_embeddings = self.user_embedding(user_indices)
        item_embeddings = self.item_embedding(item_indices)
        
        # Element-wise product
        element_product = torch.mul(user_embeddings, item_embeddings)
        
        # Output layer
        output = self.output_layer(element_product)
        
        return output.view(-1)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron module for NCF.
    
    This module implements the Multi-Layer Perceptron (MLP) component
    of the Neural Collaborative Filtering model.
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, layer_sizes: List[int]):
        """
        Initialize MLP module.
        
        Args:
            num_users: Number of users
            num_items: Number of items
            embedding_dim: Dimension of embeddings
            layer_sizes: List of layer sizes for MLP
        """
        super(MLP, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        self.mlp_layers = nn.ModuleList()
        input_size = 2 * embedding_dim
        
        # Create MLP layers
        for size in layer_sizes:
            self.mlp_layers.append(nn.Linear(input_size, size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(p=0.2))
            input_size = size
        
        # Output layer
        self.output_layer = nn.Linear(layer_sizes[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with normal distribution."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)
        
        nn.init.kaiming_uniform_(self.output_layer.weight, a=1)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP.
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            
        Returns:
            Model output
        """
        user_embeddings = self.user_embedding(user_indices)
        item_embeddings = self.item_embedding(item_indices)
        
        # Concatenate embeddings
        vector = torch.cat([user_embeddings, item_embeddings], dim=-1)
        
        # MLP layers
        for layer in self.mlp_layers:
            vector = layer(vector)
        
        # Output layer
        output = self.output_layer(vector)
        
        return output.view(-1)


class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering model.
    
    This model combines Generalized Matrix Factorization (GMF) and Multi-Layer
    Perceptron (MLP) to model user-item interactions.
    """
    
    def __init__(self, num_users: int, num_items: int, 
                 embedding_dim: int = 32, mlp_layer_sizes: List[int] = None,
                 model_type: str = 'NeuMF', dropout: float = 0.2):
        """
        Initialize NCF model.
        
        Args:
            num_users: Number of users
            num_items: Number of items
            embedding_dim: Dimension of embeddings
            mlp_layer_sizes: List of layer sizes for MLP
            model_type: Type of model ('GMF', 'MLP', or 'NeuMF')
            dropout: Dropout rate
        """
        super(NCFModel, self).__init__()
        
        if mlp_layer_sizes is None:
            mlp_layer_sizes = [64, 32, 16]
        
        self.model_type = model_type
        
        if model_type == 'GMF' or model_type == 'NeuMF':
            self.gmf = GMF(num_users, num_items, embedding_dim)
        
        if model_type == 'MLP' or model_type == 'NeuMF':
            self.mlp = MLP(num_users, num_items, embedding_dim, mlp_layer_sizes)
        
        if model_type == 'NeuMF':
            # Output layer for NeuMF
            self.output_layer = nn.Linear(mlp_layer_sizes[-1] + embedding_dim, 1)
            self.dropout = nn.Dropout(p=dropout)
            
            # Initialize weights
            nn.init.kaiming_uniform_(self.output_layer.weight, a=1)
            nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for NCF.
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            
        Returns:
            Model output
        """
        if self.model_type == 'GMF':
            return self.gmf(user_indices, item_indices)
        
        if self.model_type == 'MLP':
            return self.mlp(user_indices, item_indices)
        
        # NeuMF: Combine GMF and MLP
        gmf_output = self.gmf(user_indices, item_indices)
        mlp_output = self.mlp(user_indices, item_indices)
        
        # Get the last layer embeddings from GMF and MLP
        gmf_embedding = torch.mul(
            self.gmf.user_embedding(user_indices),
            self.gmf.item_embedding(item_indices)
        )
        
        mlp_features = None
        for layer in self.mlp.mlp_layers:
            if isinstance(layer, nn.Linear):
                mlp_features = layer(mlp_features) if mlp_features is not None else layer(
                    torch.cat([self.mlp.user_embedding(user_indices), self.mlp.item_embedding(item_indices)], dim=-1)
                )
            else:
                mlp_features = layer(mlp_features)
        
        # Concatenate GMF and MLP features
        combined = torch.cat([gmf_embedding, mlp_features], dim=1)
        combined = self.dropout(combined)
        
        # Final output
        output = self.output_layer(combined)
        
        return output.view(-1)


class NCFDataset(Dataset):
    """Dataset for Neural Collaborative Filtering."""
    
    def __init__(self, user_indices: np.ndarray, item_indices: np.ndarray, ratings: np.ndarray):
        """
        Initialize NCF dataset.
        
        Args:
            user_indices: Array of user indices
            item_indices: Array of item indices
            ratings: Array of ratings
        """
        self.user_indices = user_indices
        self.item_indices = item_indices
        self.ratings = ratings
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.ratings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (user_idx, item_idx, rating)
        """
        return (
            torch.tensor(self.user_indices[idx], dtype=torch.long),
            torch.tensor(self.item_indices[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float)
        )


class NeuralCollaborativeFiltering(MatrixFactorizationBase):
    """
    Neural Collaborative Filtering (NCF) implementation.
    
    This class implements the Neural Collaborative Filtering model, which combines
    collaborative filtering with neural networks to capture non-linear user-item
    interactions.
    """
    
    def __init__(self, name: str = "NCF", version: str = "0.1.0", 
                 embedding_dim: int = 32, mlp_layers: List[int] = None,
                 model_type: str = 'NeuMF', learning_rate: float = 0.001,
                 batch_size: int = 256, epochs: int = 20, 
                 negative_samples: int = 4, early_stopping_patience: int = 3,
                 device: str = None):
        """
        Initialize Neural Collaborative Filtering model.
        
        Args:
            name: Name of the model
            version: Version of the model
            embedding_dim: Dimension of embeddings
            mlp_layers: List of layer sizes for MLP
            model_type: Type of model ('GMF', 'MLP', or 'NeuMF')
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of epochs for training
            negative_samples: Number of negative samples per positive sample
            early_stopping_patience: Patience for early stopping
            device: Device to run model on ('cpu' or 'cuda')
        """
        super().__init__(name, version)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Neural Collaborative Filtering")
        
        self.embedding_dim = embedding_dim
        self.mlp_layers = mlp_layers if mlp_layers else [64, 32, 16]
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.negative_samples = negative_samples
        self.early_stopping_patience = early_stopping_patience
        
        # Set device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Model will be initialized during training
        self.model = None
        self.optimizer = None
        
        # Set hyperparameters
        self.set_hyperparameters(
            embedding_dim=embedding_dim,
            mlp_layers=mlp_layers,
            model_type=model_type,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            negative_samples=negative_samples
        )
    
    def _initialize_model(self, num_users: int, num_items: int) -> None:
        """Initialize the model architecture."""
        if self.model_type == 'MLP':
            self.model = MLP(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=self.embedding_dim,
                layer_sizes=self.mlp_layers
            ).to(self.device)
        elif self.model_type == 'GMF':
            self.model = GMF(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=self.embedding_dim
            ).to(self.device)
        elif self.model_type == 'NeuMF':
            self.model = NCFModel(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=self.embedding_dim,
                mlp_layer_sizes=self.mlp_layers,
                model_type='NeuMF'
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Initialize optimizer with weight decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01  # Add L2 regularization
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
    
    def _prepare_data(self, train_data: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training.
        
        Args:
            train_data: Training data DataFrame with user_id, product_id, and rating
            
        Returns:
            Tuple of (train_loader, valid_loader)
        """
        # Create ID mappings
        self._create_id_mappings(train_data)
        
        # Map external IDs to internal indices
        user_indices = np.array([self.user_map[uid] for uid in train_data['user_id']])
        item_indices = np.array([self.product_map[pid] for pid in train_data['product_id']])
        ratings = train_data['rating'].values.astype(np.float32)
        
        # Normalize ratings to [0, 1]
        ratings = (ratings - train_data['rating'].min()) / (train_data['rating'].max() - train_data['rating'].min())
        
        # Create dataset
        dataset = NCFDataset(user_indices, item_indices, ratings)
        
        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        return train_loader, valid_loader
    
    def _generate_negative_samples(self, train_data: pd.DataFrame, num_negatives: int = 4) -> pd.DataFrame:
        """Generate negative samples for training."""
        # Create a set of all user-item interactions for fast lookup
        user_item_set = set(zip(train_data['user_id'], train_data['product_id']))
        
        # Create a list of all items
        all_items = list(self.product_map.keys())
        
        # Check if all_items is empty
        if not all_items:
            logger.error("No items found in product_map. Cannot generate negative samples.")
            return train_data
        
        # Generate negative samples
        negative_samples = []
        
        for user_id in train_data['user_id'].unique():
            # Get all items this user has interacted with
            interacted_items = set(train_data[train_data['user_id'] == user_id]['product_id'])
            
            # Sample non-interacted items
            non_interacted_items = list(set(all_items) - interacted_items)
            
            # Handle users who have interacted with all items
            if not non_interacted_items:
                logger.warning(f"User {user_id} has interacted with all items. Using random existing items as negatives.")
                # Use random items from the dataset as negatives (not ideal but prevents crashes)
                try:
                    random_items = np.random.choice(all_items, size=num_negatives, replace=True)
                    for item_id in random_items:
                        negative_samples.append({
                            'user_id': user_id,
                            'product_id': item_id,
                            'rating': 0.0  # Negative sample has rating 0
                        })
                except ValueError as e:
                    logger.error(f"Error sampling items: {e}. Skipping negative samples for user {user_id}.")
                continue
            
            # Generate negative samples
            try:
                if len(non_interacted_items) >= num_negatives:
                    samples = np.random.choice(non_interacted_items, size=num_negatives, replace=False)
                else:
                    # If not enough non-interacted items, use replacement
                    samples = np.random.choice(non_interacted_items, size=num_negatives, replace=True)
                
                for item_id in samples:
                    negative_samples.append({
                        'user_id': user_id,
                        'product_id': item_id,
                        'rating': 0.0  # Negative sample has rating 0
                    })
            except ValueError as e:
                logger.error(f"Error sampling items: {e}. Skipping negative samples for user {user_id}.")
        
        # If no negative samples were generated, return original data
        if not negative_samples:
            logger.warning("No negative samples could be generated. Returning original data.")
            return train_data
            
        # Convert to DataFrame and combine with positive samples
        negative_df = pd.DataFrame(negative_samples)
        combined_df = pd.concat([train_data, negative_df], ignore_index=True)
        
        return combined_df
    
    def train(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Train the model."""
        logger.info(f"Training {self.name} model")
        
        # Check for required columns
        required_columns = ['user_id', 'product_id', 'rating']
        for col in required_columns:
            if col not in train_data.columns:
                raise ValueError(f"Required column {col} not in training data")
        
        # Generate negative samples
        if self.negative_samples > 0:
            logger.info(f"Generating {self.negative_samples} negative samples per positive sample")
            train_data = self._generate_negative_samples(train_data, self.negative_samples)
            
            # Skip training if no negative samples were generated
            if len(train_data) == 0:
                logger.warning("No negative samples could be generated. Skipping training.")
                return {}
        
        # Prepare data
        train_loader, valid_loader = self._prepare_data(train_data)
        
        # Initialize model
        num_users = len(self.user_map)
        num_items = len(self.product_map)
        logger.info(f"Initializing model with {num_users} users and {num_items} items")
        self._initialize_model(num_users, num_items)
        
        # Training loop
        start_time = time.time()
        best_valid_loss = float('inf')
        patience_counter = 0
        train_losses = []
        valid_losses = []
        
        logger.info(f"Starting training for {self.epochs} epochs")
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for user_idx, item_idx, rating in train_loader:
                # Move to device
                user_idx = user_idx.to(self.device)
                item_idx = item_idx.to(self.device)
                rating = rating.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                prediction = self.model(user_idx, item_idx)
                
                # Compute loss (MSE)
                loss = F.mse_loss(prediction, rating)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            valid_loss = 0.0
            
            with torch.no_grad():
                for user_idx, item_idx, rating in valid_loader:
                    # Move to device
                    user_idx = user_idx.to(self.device)
                    item_idx = item_idx.to(self.device)
                    rating = rating.to(self.device)
                    
                    # Forward pass
                    prediction = self.model(user_idx, item_idx)
                    
                    # Compute loss
                    loss = F.mse_loss(prediction, rating)
                    valid_loss += loss.item()
                
                valid_loss /= len(valid_loader)
                valid_losses.append(valid_loss)
                
                # Update learning rate
                self.scheduler.step(valid_loss)
            
            # Print progress
            logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}")
            
            # Early stopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                
                # Save user and item factors from PyTorch model
                self.user_factors = self._get_user_factors()
                self.item_factors = self._get_item_factors()
                
                logger.info(f"New best validation loss: {best_valid_loss:.6f}")
            else:
                patience_counter += 1
                logger.info(f"Early stopping patience: {patience_counter}/{self.early_stopping_patience}")
                
                if patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
        
        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Update metadata
        self.metadata['trained'] = True
        self.metadata['training_time'] = training_time
        
        # Metrics
        metrics = {
            'train_loss': train_losses[-1],
            'valid_loss': valid_losses[-1],
            'best_valid_loss': best_valid_loss,
            'training_time': training_time,
            'num_epochs': epoch + 1,
            'num_users': num_users,
            'num_items': num_items,
        }
        
        self.metrics = metrics
        
        return metrics
    
    def _get_user_factors(self) -> np.ndarray:
        """
        Get user latent factors from trained model.
        
        Returns:
            Array of user latent factors
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet")
        
        # Get user embedding weights based on model type
        if self.model_type == 'MLP':
            # For MLP model type, access the embedding directly
            weights = self.model.mlp.user_embedding.weight.detach().cpu().numpy()
        elif self.model_type == 'GMF':
            # For GMF model type, access from GMF model
            weights = self.model.gmf.user_embedding.weight.detach().cpu().numpy()
        elif self.model_type == 'NeuMF':
            # For NeuMF model type, get combined embeddings from both GMF and MLP parts
            gmf_weights = self.model.gmf.user_embedding.weight.detach().cpu().numpy()
            mlp_weights = self.model.mlp.user_embedding.weight.detach().cpu().numpy()
            # Combine the weights (simple concatenation as a starting point)
            weights = np.concatenate([gmf_weights, mlp_weights], axis=1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return weights
    
    def _get_item_factors(self) -> np.ndarray:
        """
        Get item latent factors from trained model.
        
        Returns:
            Array of item latent factors
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet")
        
        # Get item embedding weights based on model type
        if self.model_type == 'MLP':
            # For MLP model type, access the embedding directly
            weights = self.model.mlp.item_embedding.weight.detach().cpu().numpy()
        elif self.model_type == 'GMF':
            # For GMF model type, access from GMF model
            weights = self.model.gmf.item_embedding.weight.detach().cpu().numpy()
        elif self.model_type == 'NeuMF':
            # For NeuMF model type, get combined embeddings from both GMF and MLP parts
            gmf_weights = self.model.gmf.item_embedding.weight.detach().cpu().numpy()
            mlp_weights = self.model.mlp.item_embedding.weight.detach().cpu().numpy()
            # Combine the weights (simple concatenation as a starting point)
            weights = np.concatenate([gmf_weights, mlp_weights], axis=1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return weights
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data DataFrame
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model has not been trained yet")
        
        logger.info(f"Evaluating {self.name} model on {len(test_data)} instances")
        
        # Map user and item IDs to internal indices
        mapped_data = []
        
        for _, row in test_data.iterrows():
            user_idx = self.user_map.get(row['user_id'])
            item_idx = self.product_map.get(row['product_id'])
            
            # Skip if user or item not in training data
            if user_idx is None or item_idx is None:
                continue
            
            mapped_data.append({
                'user_idx': user_idx,
                'item_idx': item_idx,
                'rating': row['rating']
            })
        
        # Convert to DataFrame
        mapped_df = pd.DataFrame(mapped_data)
        
        if len(mapped_df) == 0:
            logger.warning("No test instances could be mapped to training data")
            return {'rmse': float('inf'), 'mae': float('inf')}
        
        # Predict ratings
        predicted_ratings = []
        
        for _, row in mapped_df.iterrows():
            pred = self._predict_score(row['user_idx'], row['item_idx'])
            predicted_ratings.append(pred)
        
        # Calculate metrics
        actual_ratings = mapped_df['rating'].values
        predicted_ratings = np.array(predicted_ratings)
        
        # Scale predictions to original rating scale
        min_rating = test_data['rating'].min()
        max_rating = test_data['rating'].max()
        predicted_ratings = predicted_ratings * (max_rating - min_rating) + min_rating
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((predicted_ratings - actual_ratings) ** 2))
        
        # Calculate MAE
        mae = np.mean(np.abs(predicted_ratings - actual_ratings))
        
        # Other metrics as needed
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'num_test_instances': len(mapped_df)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def _get_model_data(self) -> Dict[str, Any]:
        """
        Get model-specific data for saving.
        
        Returns:
            Dictionary containing model data
        """
        model_data = super()._get_model_data()
        
        # Add PyTorch model state dict if available
        if self.model is not None:
            model_data['torch_state_dict'] = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in self.model.state_dict().items()
            }
        
        return model_data
    
    def _set_model_data(self, data: Dict[str, Any]) -> None:
        """
        Set model-specific data after loading.
        
        Args:
            data: Dictionary containing model data
        """
        super()._set_model_data(data)
        
        # Recreate PyTorch model if state dict is available
        if 'torch_state_dict' in data and len(self.user_map) > 0 and len(self.product_map) > 0:
            self._initialize_model(len(self.user_map), len(self.product_map))
            
            # Convert numpy arrays back to torch tensors
            state_dict = {
                k: torch.tensor(v) if isinstance(v, np.ndarray) else v
                for k, v in data['torch_state_dict'].items()
            }
            
            self.model.load_state_dict(state_dict)
            self.model.eval()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required to run this example")
        exit(1)
    
    # Create synthetic data
    num_users = 100
    num_items = 50
    num_samples = 500
    
    np.random.seed(42)
    
    user_ids = [f"user_{i}" for i in range(num_users)]
    item_ids = [f"item_{i}" for i in range(num_items)]
    
    # Generate random interactions
    train_data = []
    for _ in range(num_samples):
        user_id = np.random.choice(user_ids)
        item_id = np.random.choice(item_ids)
        rating = np.random.randint(1, 6)  # 1-5 rating
        
        train_data.append({
            'user_id': user_id,
            'product_id': item_id,
            'rating': rating
        })
    
    # Convert to DataFrame
    train_df = pd.DataFrame(train_data)
    
    # Create and train NCF model
    ncf_model = NeuralCollaborativeFiltering(
        embedding_dim=16,
        mlp_layers=[32, 16, 8],
        epochs=5  # Small number of epochs for example
    )
    
    # Train model
    metrics = ncf_model.train(train_df)
    
    # Make predictions
    user_id = user_ids[0]
    recommendations = ncf_model.predict(user_id, n=5)
    
    logger.info(f"Top 5 recommendations for {user_id}: {recommendations}") 